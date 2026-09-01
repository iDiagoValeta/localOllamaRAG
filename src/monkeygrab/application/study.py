"""Study -- structured artifacts over retrieved evidence, instead of prose.

Issue #140. ``Answer`` streams prose for one question. This produces artifacts
with parts: a summary now, an outline and a quiz once this shape is proven.
The three share a path -- take fragments the caller already retrieved, ask the
generator for structure, parse it, attach the evidence each part came from --
and differ only in schema and prompt.

Takes its ports and config through the call, reads nothing from module state
(``AGENTS.md`` section 7 rule 2), and hard-fails rather than degrading
(rule 8).

## Why parsing is the boundary

The generator returns text; a summary has structure. Between those two facts
sits the only genuinely new failure mode here: a model that answers with
almost-JSON.

A half-parsed artifact is worse than no artifact. A summary missing its third
section looks complete, and a reader has no way to tell which parts of the
document went unmentioned because the model skipped them from those that
vanished in a truncated response. So the parse is strict: JSON that does not
decode, is not a list, or carries no usable section raises
``MalformedSummaryError``. That mirrors ``LlmProposer`` in the harness, which
validates every field of a model's reply and treats anything else as a
failure rather than a partial success.

What is *not* strict: an individual section missing its optional fields. A
section with a heading and a body is usable; one with neither is not. The
line is drawn at "can a reader use this", not at "does it match the schema
exactly", because tightening past that point turns a cosmetic difference
between models into an outage.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from monkeygrab.application.context_assembly import build_context_for_model
from monkeygrab.config.app_config import AppConfig
from monkeygrab.domain.document_summary import (
    DocumentOutline,
    DocumentSummary,
    OutlineNode,
    SummarySection,
)
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.chat_model import ChatModel

# Asking for a JSON array of objects rather than prose with markers: the
# structure is the deliverable, and a delimiter-based format ("### heading")
# fails silently when the model varies its punctuation, where malformed JSON
# fails loudly. Written in English regardless of the corpus language -- the
# instruction is to the model about format, and the language of the summary
# itself is set separately below, from the fragments' own language.
_SUMMARY_SYSTEM_PROMPT = (
    "You summarise documents. You reply with a JSON array and nothing else: "
    "no preamble, no explanation, no markdown fences. Each element is an "
    'object with exactly two string keys, "heading" and "body". The heading '
    "is at most eight words. The body is two to four sentences. Write between "
    "three and six elements, ordered so that reading them in sequence gives a "
    "coherent account of the material. Use only what the provided material "
    "states; never add facts from your own knowledge, and if the material "
    "does not support a section, write fewer sections rather than inventing "
    "one."
)

# A fenced block is the single most common deviation from "JSON and nothing
# else" across every model tried, and unwrapping it costs one regex against
# an outage. Anything past that -- prose around the JSON, trailing commas --
# is left to fail: each accommodation makes the parse guess further from what
# the model actually said.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_MAX_SECTIONS = 12

# Asking for a nested JSON structure directly rather than markdown headings
# with "#" levels: the nesting is what the caller wants, and reconstructing it
# from prefix counts means re-deriving structure the model already knew and
# threw away.
_OUTLINE_SYSTEM_PROMPT = (
    "You produce document outlines. You reply with a JSON array and nothing "
    "else: no preamble, no explanation, no markdown fences. Each element is an "
    'object with a string key "title" and an optional key "children" holding '
    "an array of the same shape. Titles are at most ten words and contain no "
    "prose. Nest no deeper than three levels. Describe only the structure the "
    "provided material actually has; never invent a section it does not "
    "contain."
)

# A model that nests without limit turns one bad reply into an outline nobody
# can render and, before that, into unbounded recursion in the parser. Three
# is what the prompt asks for; this is the guard for when it is ignored.
_MAX_OUTLINE_DEPTH = 4
_MAX_OUTLINE_NODES = 60


class MalformedOutlineError(RuntimeError):
    """The generator's reply could not be read as an outline.

    Separate from ``MalformedSummaryError`` so a caller offering both can tell
    which artifact failed without parsing an error message.
    """


class MalformedSummaryError(RuntimeError):
    """The generator's reply could not be read as a summary.

    Raised rather than returning a partial summary: a summary missing a
    section looks complete, and a reader cannot tell a section the model chose
    not to write from one lost to a truncated response.
    """


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text.strip()


def _pages_of(fragments: Sequence[Fragment]) -> Tuple[int, ...]:
    """Ascending, deduplicated page numbers of ``fragments``."""
    return tuple(sorted({f.metadata.page for f in fragments}))


def _single_document(fragments: Sequence[Fragment]) -> str:
    """The one document these fragments came from, or "" if more than one.

    Empty rather than the first: naming one document for a summary written
    from several would be a claim the summary does not support.
    """
    sources = {f.metadata.source for f in fragments}
    return next(iter(sources)) if len(sources) == 1 else ""


def _parse_sections(raw: str) -> List[Dict[str, Any]]:
    """Decode the generator's reply into section dicts, or raise.

    Raises:
        MalformedSummaryError: The reply is not JSON, is not a list, or
            contains no element with a usable heading or body.
    """
    try:
        decoded = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise MalformedSummaryError(
            f"generator did not return JSON: {exc}. First 200 characters: {raw[:200]!r}"
        ) from exc

    if not isinstance(decoded, list):
        raise MalformedSummaryError(
            f"expected a JSON array of sections, got {type(decoded).__name__}"
        )

    usable = [
        element
        for element in decoded
        if isinstance(element, dict)
        and (str(element.get("heading", "")).strip() or str(element.get("body", "")).strip())
    ]
    if not usable:
        raise MalformedSummaryError(
            f"no section carried a heading or a body ({len(decoded)} element(s) returned)"
        )
    return usable[:_MAX_SECTIONS]


class Study:
    """Structured artifacts over evidence the caller has already retrieved.

    The caller retrieves, exactly as it does for ``Answer``: this use case has
    no opinion about which fragments are relevant, only about what to build
    from them.
    """

    def __init__(self, chat_model: ChatModel) -> None:
        """
        Args:
            chat_model: The generator role. Injected rather than resolved, so
                a caller can point summarisation at a different model from
                answering without either use case knowing.
        """
        self._chat_model = chat_model

    def summarize(
        self,
        fragments: Sequence[Fragment],
        config: AppConfig,
        *,
        language: Optional[str] = None,
    ) -> DocumentSummary:
        """Summarise ``fragments`` into ordered, sourced sections.

        Args:
            fragments: Evidence to summarise, already ranked by the caller.
            config: Read fresh on every call -- never captured in a default
                argument (``tests/characterization/test_stale_default_config_bug.py``
                is the bug that rule forecloses).
            language: Language to write the summary in, e.g. "Castellano".
                ``None`` leaves it to the model, which follows the material.

        Returns:
            A ``DocumentSummary``. Empty input gives an empty summary without
            calling the generator -- there is nothing to summarise, which is
            not a failure.

        Raises:
            MalformedSummaryError: The generator's reply could not be read as
                a summary.
            Exception: Whatever the chat model raises (hard-fail policy).
        """
        if not fragments:
            return DocumentSummary(sections=(), source_document="")

        # Same context builder Answer uses, so a summary reads the evidence in
        # the format the pipeline already produces rather than a second one
        # that could drift from it. The metrics half of the return is the
        # caller's concern there and nobody's here.
        context, _metrics = build_context_for_model(
            list(fragments), config.flags.usar_optimizacion_contexto
        )
        instruction = "Summarise the following material."
        if language:
            instruction += f" Write the summary in {language}."

        raw = self._chat_model.generate(
            f"{instruction}\n\n{context}", system=_SUMMARY_SYSTEM_PROMPT
        )

        # Every section cites the whole retrieval's pages. Per-section
        # attribution would need the model to report which fragment it used,
        # and a model that miscounts would produce confident, wrong citations
        # -- worse than coarse ones, because a reader checks a specific page
        # and finds the claim absent. Coarse and true beats precise and
        # guessed; per-section attribution needs its own design.
        pages = _pages_of(fragments)
        sections = tuple(
            SummarySection(
                heading=str(element.get("heading", "")).strip(),
                body=str(element.get("body", "")).strip(),
                source_pages=pages,
            )
            for element in _parse_sections(raw)
        )
        return DocumentSummary(sections=sections, source_document=_single_document(fragments))

    def outline(
        self,
        fragments: Sequence[Fragment],
        config: AppConfig,
        *,
        language: Optional[str] = None,
    ) -> DocumentOutline:
        """Build a heading tree over ``fragments``.

        Args:
            fragments: Evidence to outline, already ranked by the caller.
            config: Read fresh on every call, as in ``summarize``.
            language: Language for the headings. ``None`` follows the material.

        Returns:
            A ``DocumentOutline``. Empty input gives an empty outline without
            calling the generator.

        Raises:
            MalformedOutlineError: The reply is not JSON, is not an array, or
                describes no titled node.
            Exception: Whatever the chat model raises (hard-fail policy).
        """
        if not fragments:
            return DocumentOutline(nodes=(), source_document="")

        context, _metrics = build_context_for_model(
            list(fragments), config.flags.usar_optimizacion_contexto
        )
        instruction = "Outline the structure of the following material."
        if language:
            instruction += f" Write the headings in {language}."

        raw = self._chat_model.generate(
            f"{instruction}\n\n{context}", system=_OUTLINE_SYSTEM_PROMPT
        )

        try:
            decoded = json.loads(_strip_fence(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedOutlineError(
                f"generator did not return JSON: {exc}. First 200 characters: {raw[:200]!r}"
            ) from exc
        if not isinstance(decoded, list):
            raise MalformedOutlineError(
                f"expected a JSON array of nodes, got {type(decoded).__name__}"
            )

        nodes = _build_outline_nodes(decoded, depth=1, budget=[_MAX_OUTLINE_NODES])
        if not nodes:
            raise MalformedOutlineError(
                f"no node carried a title ({len(decoded)} element(s) returned)"
            )
        return DocumentOutline(nodes=nodes, source_document=_single_document(fragments))


def _build_outline_nodes(elements: Any, depth: int, budget: List[int]) -> Tuple[OutlineNode, ...]:
    """Turn decoded JSON into nodes, bounded in both depth and total count.

    Two bounds because they fail differently. Depth stops a model that nests
    without end, which would otherwise recurse until the interpreter gives up.
    The node budget stops a model that returns a thousand siblings, which
    parses fine and produces an outline nobody can read.

    Both truncate rather than raise. An outline is a navigational aid: one cut
    off at three levels is still usable, where a summary missing a section is
    not -- which is why this differs from ``_parse_sections`` deliberately.

    Args:
        elements: Decoded JSON, expected to be a list of dicts.
        depth: Current nesting level, 1 at the top.
        budget: Single-element list used as a mutable counter across the
            recursion, decremented per node kept.
    """
    if not isinstance(elements, list) or depth > _MAX_OUTLINE_DEPTH:
        return ()

    nodes: List[OutlineNode] = []
    for element in elements:
        if budget[0] <= 0:
            break
        if not isinstance(element, dict):
            continue
        title = str(element.get("title", "")).strip()
        if not title:
            # A node with no title is not a heading. Its children, if any, are
            # dropped with it: re-parenting them would invent a structure the
            # model did not describe.
            continue
        budget[0] -= 1
        nodes.append(
            OutlineNode(
                title=title,
                children=_build_outline_nodes(element.get("children"), depth + 1, budget),
            )
        )
    return tuple(nodes)


def summary_to_dict(summary: DocumentSummary) -> Dict[str, Any]:
    """Plain-JSON view for an interface layer.

    Lives here rather than in the interface so the web app and the CLI cannot
    drift into two different shapes for the same artifact.
    """
    # Lists, not the dataclass's tuples: `dataclasses.asdict` keeps tuples,
    # and json.dumps turns them into lists, so a caller comparing a payload
    # against its own round-trip would see a difference that is not one.
    return {
        "source_document": summary.source_document,
        "sections": [
            {
                "heading": section.heading,
                "body": section.body,
                "source_pages": list(section.source_pages),
            }
            for section in summary.sections
        ],
    }
