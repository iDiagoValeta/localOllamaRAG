"""Study -- structured artifacts over retrieved evidence, instead of prose.

Issue #140. ``Answer`` streams prose for one question. This produces artifacts
with parts: a summary, an outline and a quiz. The three share a path -- take
fragments the caller already retrieved, ask the generator for structure, parse
it, attach the evidence each part came from -- and differ only in schema,
prompt, and how strict the parse is (see below).

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

The three artifacts draw that line in three places, and the difference is the
cost of being wrong, not a matter of taste. An outline truncates: one cut off
at three levels still navigates. A summary drops unusable sections but raises
if none survive: a section a reader cannot read is visibly absent. A quiz is
strictest, because it is the only one that can be wrong *without looking
wrong* -- a key pointing at the wrong option grades the reader against a
falsehood, and checking it is exactly what they could not do.
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
    Quiz,
    QuizQuestion,
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

# The same shape the prompt asks for in prose, in the one channel the backend
# can enforce. Measured 2026-09-02 on higgs-boson.pdf, the document whose
# quizzes failed every single time: unconstrained 2/5, and 5/5 with a schema.
# A prompt can only ask; this refuses to decode a reply that would not parse.
#
# Note what a schema does NOT do: it constrains structure, never truth. A key
# pointing at a plausible wrong option satisfies every schema ever written,
# which is why _usable_question still runs afterwards and still drops what it
# cannot verify.
_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"heading": {"type": "string"}, "body": {"type": "string"}},
        "required": ["heading", "body"],
    },
}

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

# Nested to the depth the prompt asks for and no further. A recursive schema
# ($ref) would express "any depth", which is exactly what _MAX_OUTLINE_DEPTH
# exists to refuse -- and a backend that cannot resolve the reference would
# fall back to unconstrained decoding without saying so.
_OUTLINE_NODE_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
_OUTLINE_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "children": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "children": {"type": "array", "items": _OUTLINE_NODE_SCHEMA},
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["title"],
    },
}


# Asking for the key as an index into the options rather than as the correct
# answer's text: a model that restates the text introduces a second copy that
# can disagree with the option it names -- by a comma, by a truncation -- and
# then nothing downstream can tell which of the two is the answer.
_QUIZ_SYSTEM_PROMPT = (
    "You write multiple-choice comprehension questions. You reply with a JSON "
    "array and nothing else: no preamble, no explanation, no markdown fences. "
    'Each element is an object with three keys: "prompt" (the question), '
    '"options" (an array of three or four distinct answer strings) and '
    '"correct_index" (the zero-based position in that array of the correct '
    "option). Exactly one option is correct and the others must be plausible "
    "but wrong. Ask only about what the provided material states; never draw "
    "on your own knowledge, and if the material supports fewer questions than "
    "requested, return fewer rather than inventing any."
)

# A quiz nobody asked the length of is not a quiz. The ceiling is a guard on
# the model, not a product limit: past it the reply is long enough that
# truncation, not question count, is what decides where it ends.
_DEFAULT_QUESTION_COUNT = 5
_MAX_QUESTION_COUNT = 20

# Two is a coin toss and four is the conventional ceiling for a readable
# question. Below the floor there is no choice to make; above the ceiling the
# model starts padding with near-duplicates of the correct option.
_MIN_OPTIONS = 2
_MAX_OPTIONS = 6

# correct_index is bounded by the schema as well as by _usable_question. The
# schema stops the model emitting 7 for a four-option question; the parse still
# checks the key against the options actually returned, because a schema cannot
# know how many there were.
_QUIZ_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 4,
            },
            "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
        },
        "required": ["prompt", "options", "correct_index"],
    },
}


class MalformedQuizError(RuntimeError):
    """The generator's reply held no question safe to grade a reader against.

    Separate from the summary and outline errors so a caller offering several
    artifacts can tell which one failed without parsing an error message.
    """


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


def _calculate_context_char_budget(config: AppConfig) -> int:
    """Derive a safe character budget for document context from config.models.ollama.rag_num_ctx.

    Reserves tokens for instructions, system prompt, JSON schema, and output generation.
    Assumes a conservative 3.0 characters per token to account for non-ASCII multi-lingual
    text and dense mathematical notations.
    """
    num_ctx = config.models.ollama.rag_num_ctx
    available_tokens = max(1000, num_ctx - 2048)
    return int(available_tokens * 3.0)


def _budget_fragments(fragments: Sequence[Fragment], max_chars: int) -> Sequence[Fragment]:
    """Sample fragments evenly if the full document exceeds the character budget (issue #170).

    Study artifacts operate over whole documents rather than a top-k retrieval. For
    large documents (e.g. 60+ pages with > 200k tokens), sending every chunk
    exceeds the generator's context window (num_ctx) and causes Ollama to reject
    the call with HTTP 400 exceed_context_size_error.

    When total characters fit within max_chars, all fragments are preserved verbatim.
    When exceeding max_chars, fragments are sampled uniformly across the document
    so all sections remain represented in the artifact.
    """
    if not fragments or max_chars <= 0:
        return fragments

    total_chars = sum(len(f.doc) for f in fragments)
    if total_chars <= max_chars:
        return fragments

    avg_chunk_len = max(1, total_chars // len(fragments))
    target_count = max(1, min(len(fragments), max_chars // avg_chunk_len))
    if target_count >= len(fragments):
        return fragments

    step = len(fragments) / target_count
    sampled_indices = [int(i * step) for i in range(target_count)]
    seen = set()
    unique_indices = []
    for idx in sampled_indices:
        if idx not in seen and idx < len(fragments):
            seen.add(idx)
            unique_indices.append(idx)

    sampled = [fragments[i] for i in unique_indices]
    current_chars = 0
    kept = []
    for f in sampled:
        if current_chars + len(f.doc) > max_chars and kept:
            break
        kept.append(f)
        current_chars += len(f.doc)
    return kept or fragments[:1]


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


def _usable_question(element: Any) -> Optional[Dict[str, Any]]:
    """The question's fields if it is safe to grade against, else ``None``.

    Stricter than ``_parse_sections`` on purpose. A thin summary section is
    thin and the reader can see that. A question whose key points at the wrong
    option teaches something false with the authority of an answer key, and
    the reader has no way to notice -- checking is precisely what they were
    unable to do, which is why they are taking the quiz.

    So every dropped case below is one where the artifact would still *look*
    answerable. Nothing here is repaired: there is no evidence for what the
    model meant, and a guessed key is the failure this function exists to
    prevent.
    """
    if not isinstance(element, dict):
        return None

    prompt = str(element.get("prompt", "")).strip()
    if not prompt:
        return None

    raw_options = element.get("options")
    if not isinstance(raw_options, list):
        return None
    options = [str(option).strip() for option in raw_options if str(option).strip()]
    if not _MIN_OPTIONS <= len(options) <= _MAX_OPTIONS:
        return None
    # Duplicates make the key ambiguous rather than wrong: a reader who picks
    # the other identical option answered correctly and grades as incorrect.
    if len(set(options)) != len(options):
        return None

    index = element.get("correct_index")
    # `bool` is an `int` in Python, and `True` would index option 1 silently.
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    # Not `-len <= index < len`: a negative index is legal in Python and would
    # quietly select from the end, but from a model it means nothing at all.
    if not 0 <= index < len(options):
        return None

    return {"prompt": prompt, "options": options, "correct_index": index}


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
        budgeted = _budget_fragments(fragments, _calculate_context_char_budget(config))
        context, _metrics = build_context_for_model(
            list(budgeted), config.flags.usar_optimizacion_contexto
        )
        instruction = "Summarise the following material."
        if language:
            instruction += f" Write the summary in {language}."

        raw = self._chat_model.generate(
            f"{instruction}\n\n{context}",
            system=_SUMMARY_SYSTEM_PROMPT,
            response_format=_SUMMARY_SCHEMA,
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

        budgeted = _budget_fragments(fragments, _calculate_context_char_budget(config))
        context, _metrics = build_context_for_model(
            list(budgeted), config.flags.usar_optimizacion_contexto
        )
        instruction = "Outline the structure of the following material."
        if language:
            instruction += f" Write the headings in {language}."

        raw = self._chat_model.generate(
            f"{instruction}\n\n{context}",
            system=_OUTLINE_SYSTEM_PROMPT,
            response_format=_OUTLINE_SCHEMA,
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


    def quiz(
        self,
        fragments: Sequence[Fragment],
        config: AppConfig,
        *,
        language: Optional[str] = None,
        question_count: int = _DEFAULT_QUESTION_COUNT,
    ) -> Quiz:
        """Write multiple-choice questions over ``fragments``.

        Args:
            fragments: Evidence to question, already ranked by the caller.
            config: Read fresh on every call, as in ``summarize``.
            language: Language for the questions. ``None`` follows the
                material.
            question_count: How many to ask for. The model may return fewer
                if the material does not support that many, which is not an
                error -- padding would mean inventing questions.

        Returns:
            A ``Quiz``. Empty input gives an empty quiz without calling the
            generator.

        Raises:
            ValueError: ``question_count`` is outside 1..20. Checked before
                the model is called, because a bad count is the caller's bug
                and spending a generation on it hides that.
            MalformedQuizError: The reply is not JSON, is not an array, or
                held no question safe to grade against.
            Exception: Whatever the chat model raises (hard-fail policy).
        """
        if not 1 <= question_count <= _MAX_QUESTION_COUNT:
            raise ValueError(
                f"question_count must be between 1 and {_MAX_QUESTION_COUNT}, got {question_count}"
            )
        if not fragments:
            return Quiz(questions=(), source_document="")

        budgeted = _budget_fragments(fragments, _calculate_context_char_budget(config))
        context, _metrics = build_context_for_model(
            list(budgeted), config.flags.usar_optimizacion_contexto
        )
        instruction = (
            f"Write {question_count} multiple-choice questions about the following material."
        )
        if language:
            instruction += f" Write the questions and options in {language}."

        raw = self._chat_model.generate(
            f"{instruction}\n\n{context}",
            system=_QUIZ_SYSTEM_PROMPT,
            response_format=_QUIZ_SCHEMA,
        )

        try:
            decoded = json.loads(_strip_fence(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedQuizError(
                f"generator did not return JSON: {exc}. First 200 characters: {raw[:200]!r}"
            ) from exc
        if not isinstance(decoded, list):
            raise MalformedQuizError(
                f"expected a JSON array of questions, got {type(decoded).__name__}"
            )

        usable = [fields for fields in map(_usable_question, decoded) if fields is not None]
        if not usable:
            raise MalformedQuizError(
                "no question carried a prompt, distinct options and an in-range key "
                f"({len(decoded)} element(s) returned)"
            )

        # Same coarse attribution as summarize: the whole retrieval's pages
        # per question. Asking the model which fragment it used would buy
        # precision at the price of confident, wrong page numbers -- and here
        # a reader follows the citation precisely when they doubt the key.
        pages = _pages_of(fragments)
        questions = tuple(
            QuizQuestion(
                prompt=fields["prompt"],
                options=tuple(fields["options"]),
                correct_index=fields["correct_index"],
                source_pages=pages,
            )
            for fields in usable[:question_count]
        )
        return Quiz(questions=questions, source_document=_single_document(fragments))


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


def quiz_to_dict(quiz: Quiz) -> Dict[str, Any]:
    """Plain-JSON view for an interface layer.

    Lives here for the same reason ``summary_to_dict`` does: one shape for the
    artifact, so the CLI and the web app cannot drift into two.
    """
    return {
        "source_document": quiz.source_document,
        "questions": [
            {
                "prompt": question.prompt,
                "options": list(question.options),
                "correct_index": question.correct_index,
                "source_pages": list(question.source_pages),
            }
            for question in quiz.questions
        ],
    }


def _outline_node_to_dict(node: OutlineNode) -> Dict[str, Any]:
    return {"title": node.title, "children": [_outline_node_to_dict(c) for c in node.children]}


def outline_to_dict(outline: DocumentOutline) -> Dict[str, Any]:
    """Plain-JSON view for an interface layer.

    Children stay nested rather than being flattened with a depth number: the
    nesting is the artifact, and flattening it here would make every consumer
    rebuild the tree, each in its own slightly different way.
    """
    return {
        "source_document": outline.source_document,
        "nodes": [_outline_node_to_dict(node) for node in outline.nodes],
    }
