"""Documentation drift check: the labels the contribution standard names
against the labels this repository declares.

``CONTRIBUTING.md``'s issue standard states a "one label minimum". On
2026-09-01 that list named six labels, of which two did not exist (``docs``
-- the real label is ``documentation`` -- and ``loop``, never created), while
two labels carrying live issues (``performance``, ``research``) were absent
from it (issue #109). An agent following the standard literally got an error
from ``gh issue create --label docs`` or invented a label; the distinction
``loop`` was meant to draw, between the measurement gate and the optimiser
built on it, was recorded nowhere, and all three open harness issues sat
under ``eval``.

This is the defect class of issue #63 (``.env.example`` documenting three
variables no code read), one level up: a pointer that names something which
does not exist, reading as authoritative because it is written down. The
answer here is the same as the answer there -- check it mechanically instead
of hand-auditing it -- and the same as this repo's answer everywhere else:
``tests/unit/test_architecture_boundaries.py`` parses imports,
``harness/tests/test_harness_boundaries.py`` parses write calls, neither
trusts the eye.

Both directions are checked, because the drift ran both ways:

1. Every label the documentation names is declared in ``.github/labels.json``.
2. Every ``kind: type`` label declared there is named by the documentation.

## Scope boundary

This file compares two files in the working tree. It cannot see the labels
GitHub actually holds -- that needs the API, which the fast gate has no
network for and no business calling. ``.github/workflows/labels.yml`` closes
that half of the loop with ``gh label list`` on a runner that is already
authenticated. Detection only, in both places: a failure means a human reads
the diff and decides which side is wrong, per ``docs/README.md``'s "the code
is the source of truth" rule inverted for a file whose source of truth is
GitHub itself.

Standard library plus pytest, no YAML parser, so this runs in the fast
gate's ``architecture`` job (``.github/workflows/ci.yml``) with nothing
installed. That is also why ``.github/labels.json`` is JSON and not the
``labels.yml`` such files conventionally are: a dependency to read the
declaration would put this check in a slower job, and a check that runs late
is a check people learn to skip.
"""

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABELS_FILE = _REPO_ROOT / ".github" / "labels.json"
_CONTRIBUTING = _REPO_ROOT / "CONTRIBUTING.md"
_TEMPLATE_DIR = _REPO_ROOT / ".github" / "ISSUE_TEMPLATE"

# The documentation names labels in prose full of other backticked things
# (`feat`, `pytest`, `requirements*.txt`), so the label list is delimited
# explicitly rather than guessed at. Deleting the markers is itself a
# failure -- see test_contributing_still_carries_the_delimited_label_block --
# so the check cannot be silenced by removing what it reads.
_BLOCK_BEGIN = "<!-- labels:begin -->"
_BLOCK_END = "<!-- labels:end -->"

_BACKTICKED = re.compile(r"`([^`]+)`")
# HTML comments inside the block explain the block, and their prose contains
# backticked things that are not labels (`gh issue create`). Stripping them
# is not a hole: a label named only inside a comment is invisible to the
# reader the standard is written for, so it is not a claim this check owes
# anything to.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# GitHub issue-template frontmatter: `labels: eval, loop` on one line.
_FRONTMATTER_LABELS = re.compile(r"^labels:\s*(.+)$", re.MULTILINE)

_VALID_KINDS = ("type", "modifier", "github-default")
_HEX_COLOR = re.compile(r"^[0-9a-f]{6}$")


def _declared():
    """Every label in ``.github/labels.json``, as ``{name: entry}``."""
    data = json.loads(_LABELS_FILE.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in data["labels"]}


def _contributing_text() -> str:
    return _CONTRIBUTING.read_text(encoding="utf-8")


def _labels_named_in_contributing() -> set:
    """Backticked names inside CONTRIBUTING.md's delimited label block."""
    text = _contributing_text()
    start = text.index(_BLOCK_BEGIN) + len(_BLOCK_BEGIN)
    end = text.index(_BLOCK_END)
    block = _HTML_COMMENT.sub("", text[start:end])
    return set(_BACKTICKED.findall(block))


def _labels_named_in_templates() -> dict:
    """``{template filename: {label, ...}}`` from each template's frontmatter."""
    named = {}
    for path in sorted(_TEMPLATE_DIR.glob("*.md")):
        found = set()
        for raw in _FRONTMATTER_LABELS.findall(path.read_text(encoding="utf-8")):
            found.update(part.strip() for part in raw.split(",") if part.strip())
        named[path.name] = found
    return named


def test_contributing_still_carries_the_delimited_label_block():
    """The markers this check reads must exist, or it silently passes on nothing."""
    text = _contributing_text()
    assert _BLOCK_BEGIN in text and _BLOCK_END in text, (
        f"CONTRIBUTING.md must delimit its label list with {_BLOCK_BEGIN} / "
        f"{_BLOCK_END}. Without them this test would read an empty set and pass "
        "regardless of what the standard says."
    )
    assert text.index(_BLOCK_BEGIN) < text.index(_BLOCK_END)
    assert _labels_named_in_contributing(), "the delimited block names no label at all"


def test_every_label_contributing_names_is_declared():
    """Direction 1, the failure that opened #109: `docs` and `loop` were fiction."""
    declared = _declared()
    undeclared = sorted(_labels_named_in_contributing() - set(declared))
    assert not undeclared, (
        f"CONTRIBUTING.md names labels absent from {_LABELS_FILE.name}: {undeclared}. "
        "Either the label was renamed on GitHub (fix the doc) or it is new "
        "(declare it here and create it there)."
    )


def test_every_label_an_issue_template_assigns_is_declared():
    """A template assigning an unknown label silently drops it on issue creation."""
    declared = set(_declared())
    offenders = {
        name: sorted(labels - declared)
        for name, labels in _labels_named_in_templates().items()
        if labels - declared
    }
    assert not offenders, (
        f"issue templates assign undeclared labels: {offenders}. GitHub applies "
        "what it can and drops the rest without warning, so the issue lands "
        "unlabelled instead of failing loudly."
    )


def test_every_type_label_is_named_by_the_standard():
    """Direction 2: `performance` and `research` carried issues while unmentioned.

    Only ``kind: type`` labels are required in the prose. Modifiers stack on a
    type and need no separate mention; ``github-default`` entries are
    inventory, kept so this file is a complete list of what exists, not part
    of the standard.
    """
    declared = _declared()
    expected = {name for name, entry in declared.items() if entry["kind"] == "type"}
    missing = sorted(expected - _labels_named_in_contributing())
    assert not missing, (
        f"labels declared as `type` but not named in CONTRIBUTING.md's label "
        f"block: {missing}. A type label nobody documents is one nobody applies."
    )


@pytest.mark.parametrize("field", ("name", "color", "description", "kind"))
def test_every_declared_label_carries_every_field(field):
    for name, entry in _declared().items():
        assert entry.get(field), f"label {name!r} has no {field!r}"


def test_declared_labels_are_well_formed():
    """Kinds are from the fixed set, colours are bare hex, names are unique.

    Uniqueness is checked against the raw list rather than the dict built from
    it: ``_declared()`` would silently keep the last of two entries sharing a
    name, which is exactly the kind of duplicate worth catching.
    """
    raw = json.loads(_LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    names = [entry["name"] for entry in raw]
    assert len(names) == len(set(names)), f"duplicate label names: {names}"
    for entry in raw:
        assert entry["kind"] in _VALID_KINDS, (
            f"label {entry['name']!r} has kind {entry['kind']!r}, "
            f"expected one of {_VALID_KINDS}"
        )
        assert _HEX_COLOR.match(entry["color"]), (
            f"label {entry['name']!r} has colour {entry['color']!r}; expected six "
            "lowercase hex digits with no leading '#', matching `gh label list`"
        )
