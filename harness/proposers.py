"""proposers -- GridProposer (deterministic control) and LlmProposer.

Both implement ``propose(space, ledger_history, last_result) -> dict[str, Any]``
(issue #31 spec section 5.4): a dotted-key overrides dict, already checked
against the declared space and the feasibility predicate. ``space`` is the
``search_space`` module itself, so a proposer never hardcodes anything the
space could change.

After ``propose()`` returns, a proposer's ``last_meta`` attribute carries
what the return value alone cannot: which proposer actually produced it
(relevant when ``LlmProposer`` falls back), the LLM's rationale and model id,
and whether a fallback fired. ``loop.py`` reads it to build the ledger entry.

Criterion 5 (docs/design/2026-07-28-loop-automejorable.md section 2) runs the
loop with both proposers so the LLM's reasoning can be shown to be worth its
cost, or shown not to be, against this deterministic control.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class SearchSpaceExhausted(RuntimeError):
    """Raised when a proposer has no untried, feasible point left to offer.

    Not an error in the everyday sense -- a finite declared space run long
    enough eventually runs out of single-field perturbations. ``loop.py``
    catches this as an additional termination condition, on top of
    ``--max-iterations``/``--patience``.
    """


class OllamaUnreachableError(RuntimeError):
    """The real Ollama call could not reach the server, or timed out."""


def _field_value(config: Any, dotted_key: str) -> Any:
    """``_field_value(config, "retrieval.top_k_final")`` -> ``config.retrieval.top_k_final``."""
    section_name, field_name = dotted_key.split(".", 1)
    return getattr(getattr(config, section_name), field_name)


class GridProposer:
    """Deterministic coordinate descent from the reference configuration.

    Walks ``space.proposal_order()`` (known-reachable keys first -- see
    ``search_space.proposal_order``'s docstring), and for each key, its
    declared values in the order given, proposing a *single*-field change
    away from ``reference`` at a time (skipping the value that already
    equals the reference's own, since that is not a change). Skips any point
    already present in ``ledger_history`` (by its raw ``config_overrides``)
    and any point ``space.is_feasible`` rejects. Reproducible from the
    ledger alone: no randomness, no internal counter surviving between
    ``propose()`` calls -- the walk position is entirely a function of
    ``ledger_history``.
    """

    def __init__(self, reference: Any):
        """Args:
            reference: The ``AppConfig`` every single-field perturbation is
                measured against (typically ``AppConfig.from_env()``, fixed
                for the whole loop run).
        """
        self._reference = reference
        self.last_meta: Dict[str, Any] = {"proposer": "grid", "rationale": None, "model": None,
                                           "fallback": False, "fallback_reason": None}

    def propose(self, space, ledger_history: Sequence[Any], last_result: Any) -> Dict[str, Any]:
        """Args:
            space: The ``search_space`` module.
            ledger_history: Every ``LedgerEntry`` written so far, oldest first.
            last_result: Unused by ``GridProposer`` (it only ever looks at
                ``reference`` and the ledger) -- present for interface
                parity with ``LlmProposer``.

        Returns:
            A single-key overrides dict.

        Raises:
            SearchSpaceExhausted: Every declared, feasible, untried point has
                already been proposed.
        """
        del last_result  # deterministic control: only reference + ledger matter
        tried = {frozenset(entry.config_overrides.items()) for entry in ledger_history}
        for key in space.proposal_order():
            default = _field_value(self._reference, key)
            for value in space.ALLOWED_VALUES[key]:
                if value == default:
                    continue
                overrides = {key: value}
                if frozenset(overrides.items()) in tried:
                    continue
                if not space.is_feasible(overrides, self._reference):
                    continue
                return overrides
        raise SearchSpaceExhausted("GridProposer has walked every declared, feasible point")


class LlmProposer:
    """Ollama-backed proposer, validated against the declared space before use.

    Prompted with the declared space, the ledger history as
    ``(overrides -> objective)`` pairs, and the currently failing search-set
    cases' questions (joined from ``gold_cases.jsonl`` by id -- ``last_result``
    itself carries no question text, only id/case_type/pass/elapsed, see
    ``evaluator.CaseRecord``). Must return strict JSON
    ``{"overrides": {...}, "rationale": "..."}``.

    Validation is the security boundary, not the prompt (issue #31 spec
    section 5.4): every returned overrides dict is checked against
    ``space.DECLARED_KEYS``/``space.ALLOWED_VALUES``/``space.is_feasible``
    before it is trusted. A key outside the space, a value outside the
    declared set, an infeasible combination, or JSON that does not parse to
    the required shape all count as a failed attempt and are retried, up to
    ``max_retries`` times; past that, or the moment Ollama is unreachable,
    this proposer falls back to an internal ``GridProposer`` and records the
    fallback in ``last_meta`` rather than blocking the loop. Because the
    only thing ``propose()`` can ever return is a dict of declared dotted
    keys with declared values, there is no path by which the LLM's output
    reaches ``grade.py``, the gold-case file or the baseline -- the harness
    never evaluates a key or value it did not itself declare.
    """

    DEFAULT_MODEL = "gemma4:e4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        reference: Any,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        call: Optional[Callable[[str], str]] = None,
    ):
        """Args:
            reference: Passed through to the internal fallback ``GridProposer``.
            model: Ollama model id (default matches repo convention, see
                harness/README.md).
            base_url: Ollama server base URL.
            timeout: Bounded HTTP timeout in seconds for the real call --
                this is what stops an unresponsive Ollama from hanging the
                loop.
            max_retries: Attempts before falling back to grid.
            call: Injected ``prompt -> raw_text`` callable, for tests. When
                ``None``, uses the real Ollama HTTP call (lazy ``requests``
                import, so importing this module never requires it).
        """
        self._reference = reference
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._call = call
        self._grid_fallback = GridProposer(reference)
        self.last_meta: Dict[str, Any] = {}

    def propose(self, space, ledger_history: Sequence[Any], last_result: Any) -> Dict[str, Any]:
        prompt = self._build_prompt(space, ledger_history, last_result)
        for _attempt in range(self._max_retries):
            try:
                raw_text = self._call_llm(prompt)
            except OllamaUnreachableError as exc:
                return self._fallback(space, ledger_history, last_result, f"Ollama unreachable: {exc}")
            parsed = self._parse(raw_text)
            if parsed is None:
                continue
            overrides, rationale = parsed
            if not self._validate(overrides, space):
                continue
            self.last_meta = {
                "proposer": "llm", "rationale": rationale, "model": self._model,
                "fallback": False, "fallback_reason": None,
            }
            return overrides
        return self._fallback(
            space, ledger_history, last_result,
            f"no valid proposal in {self._max_retries} attempt(s) (malformed JSON or rejected by the space)",
        )

    def _fallback(self, space, ledger_history: Sequence[Any], last_result: Any, reason: str) -> Dict[str, Any]:
        overrides = self._grid_fallback.propose(space, ledger_history, last_result)
        self.last_meta = {
            "proposer": "llm", "rationale": None, "model": self._model,
            "fallback": True, "fallback_reason": reason,
        }
        return overrides

    def _call_llm(self, prompt: str) -> str:
        call = self._call or self._real_call
        return call(prompt)

    def _real_call(self, prompt: str) -> str:
        """Real Ollama call: ``/api/generate``, non-streaming, JSON-formatted output."""
        import requests  # lazy: keeps this module importable with no engine deps

        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model, "prompt": prompt, "stream": False,
                    "format": "json", "think": False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnreachableError(str(exc)) from exc
        return response.json().get("response", "")

    @staticmethod
    def _parse(raw_text: str) -> Optional[Tuple[Dict[str, Any], str]]:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        overrides = data.get("overrides")
        rationale = data.get("rationale")
        if not isinstance(overrides, dict) or not isinstance(rationale, str):
            return None
        return overrides, rationale

    def _validate(self, overrides: Dict[str, Any], space) -> bool:
        for key, value in overrides.items():
            if key not in space.DECLARED_KEYS:
                return False
            if value not in space.ALLOWED_VALUES[key]:
                return False
        return space.is_feasible(overrides, self._reference)

    def _build_prompt(self, space, ledger_history: Sequence[Any], last_result: Any) -> str:
        """Declared space + ledger history + currently failing questions.

        Failing-case *questions* are joined from ``gold_cases.jsonl`` by id
        (``evaluator._gold_cases``, kept as a lazy import here so this module
        never needs ``evaluator`` at collection time either). What retrieval
        actually surfaced is NOT included: ``evaluator.CaseRecord`` -- the
        contract every ``EvaluatorFn`` (fake or real) returns -- carries only
        id/case_type/passed/elapsed_seconds, not retrieved fragments. A
        richer per-case diagnostic would need widening that contract, which
        this PR does not do (see the PR report).
        """
        from harness import evaluator as evaluator_mod

        lines: List[str] = [
            "You are proposing one configuration change for a retrieval-augmented "
            "generation pipeline, to improve gold-case pass rate without exceeding "
            "a latency ceiling.",
            "",
            "Declared search space (dotted_key: allowed_values -- why):",
        ]
        for key, values, why, _stage in space.SEARCH_SPACE:
            lines.append(f"  {key}: {list(values)} -- {why}")

        lines.append("")
        lines.append("Ledger history (overrides -> objective_adjusted, verdict):")
        if ledger_history:
            for entry in ledger_history:
                lines.append(f"  {entry.config_overrides} -> {entry.objective_adjusted} ({entry.verdict})")
        else:
            lines.append("  (empty -- this is the first iteration)")

        if last_result is not None:
            questions_by_id = {c["id"]: c["question"] for c in evaluator_mod._gold_cases() if "question" in c}
            failing = [r for r in last_result.records if not r.passed]
            lines.append("")
            lines.append("Currently failing search-set cases:")
            for record in failing:
                question = questions_by_id.get(record.id, "(question unavailable)")
                lines.append(f"  {record.id} ({record.case_type}): {question}")

        lines.append("")
        lines.append(
            "Propose ONE change: a dotted key from the declared space above, set to "
            "one of its declared values. Respond with strict JSON only, no prose, "
            'no markdown fences: {"overrides": {"<dotted_key>": <value>}, '
            '"rationale": "<why this change should help>"}'
        )
        return "\n".join(lines)
