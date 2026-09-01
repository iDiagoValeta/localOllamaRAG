"""environment -- the installed stack an iteration was measured on.

Issue #107. Two ledger entries are treated as comparable when their models,
chunking and index-time flags match (``loop._comparable_config_view``). That
view is complete for *configuration* and blind to everything else: MinerU and
jina-clip get upgraded, an upstream fix changes what a page extracts to, the
corpus is reindexed. The entry from August still declares itself comparable,
still holds passes the current stack cannot reach, and recovery mode pairs
against a high water nobody can climb back to -- so every campaign ends
``rejected_regression`` on ghost passes, with no bug to fix.

This module makes that drift visible instead of silent: which versions of the
stack an iteration ran on, recorded per entry, so a later launch can say
whether it is standing in the same world.

## What "the stack" means here, and what it deliberately excludes

Two environments decide a case's outcome, and they are not the same
environment (README.md, "Install, models and configuration"):

- **isolated** (``.venv-mineru``) -- MinerU and jina-clip-v2 build the index.
  A change here changes stored content, which no amount of retrieval tuning
  can undo.
- **product** (whatever interpreter runs the gate) -- the BGE reranker, FAISS
  and BM25 decide retrieval; the Ollama client decides how generation is
  called.

``transformers``, ``sentence-transformers`` and ``torch`` are installed in
BOTH, at deliberately different versions (transformers 4.x isolated, 5.x
product -- ``rag/requirements.txt`` explains why), and they mean different
things in each. So versions are recorded per environment, never merged: a
single ``transformers`` key would record one of the two and silently drop the
other.

**The git commit is not part of this.** It is already recorded per entry
(``ledger.read_git_commit``) and it moves on every commit, including the ones
that touch a README. Folding it into comparability would make every entry
incomparable with every other one, which is not "strict", it is useless.

## Why this reads directories instead of asking Python

``importlib`` and ``subprocess`` are both banned under ``harness/``
(``harness/tests/test_harness_boundaries.py``, issue #31 spec section 1), and
that ban is what makes the "this harness cannot invoke git" claim checked
rather than asserted. Neither ``importlib.metadata.version`` nor
``pip freeze`` is available, so versions are read the one way left: the name
of the ``*.dist-info`` directory pip writes, which carries the version
already. That restriction turns out to be the right tool anyway -- it is the
only one of the three that can read the *isolated* venv, which is a different
interpreter this process never runs.

Best-effort throughout, like ``ledger.read_git_commit``: an unreadable
environment records ``None`` and the loop carries on. A campaign must not die
over provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

FINGERPRINT_SCHEMA = 1

# Declared by hand, like search_space.SEARCH_SPACE: each entry is a decision
# that this package can change a case's outcome, not a dump of what happens to
# be installed. A `pip freeze` fingerprint would change on every unrelated
# transitive bump and report drift that decides nothing.
#
# Read in both environments -- a package absent from one records None there
# and is skipped when comparing (see `compare`), so one list serves both
# without pretending the product venv holds MinerU.
TRACKED_PACKAGES = (
    "mineru",                 # extraction: what a PDF becomes
    "transformers",           # jina-clip's and the reranker's runtime
    "sentence-transformers",  # BGE reranker; jina-clip's loader in the isolated venv
    "torch",                  # numerics under both of the above
    "faiss-cpu",              # the index and its search
    "rank-bm25",              # the keyword half of hybrid retrieval
    "ollama",                 # how generation is called
)

# Ordered: the isolated environment is what builds the index, so it is named
# first everywhere a human reads this.
ISOLATED_ENV = "isolated"
PRODUCT_ENV = "product"

MATCH = "match"
DIFFERS = "differs"
UNKNOWN = "unknown"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(name: str) -> str:
    """`sentence_transformers` and `sentence-transformers` are one package.

    pip writes the underscore form into the ``.dist-info`` directory name and
    the hyphen form is what humans and requirements files use.
    """
    return name.replace("_", "-").lower()


def read_stack_versions(roots: Iterable[Path]) -> Dict[str, Optional[str]]:
    """Versions of ``TRACKED_PACKAGES`` found under ``roots``, first root wins.

    Args:
        roots: ``site-packages`` directories, most specific first. Missing
            directories are skipped rather than raising -- an environment that
            was never installed is a normal state here, not an error.

    Returns:
        One entry per tracked package, ``None`` where it was not found.
    """
    found: Dict[str, Optional[str]] = {name: None for name in TRACKED_PACKAGES}
    tracked = set(TRACKED_PACKAGES)
    for root in roots:
        try:
            entries = sorted(Path(root).glob("*.dist-info"))
        except OSError:
            continue
        for entry in entries:
            stem = entry.name[: -len(".dist-info")]
            dist_name, _, version = stem.rpartition("-")
            normalized = _normalize(dist_name)
            if normalized in tracked and found[normalized] is None and version:
                found[normalized] = version
    return found


def _isolated_site_packages() -> List[Path]:
    """``.venv-mineru``'s site-packages, both layouts, matching composition.py."""
    venv = _PROJECT_ROOT / ".venv-mineru"
    candidates = list(venv.glob("lib/python*/site-packages"))
    windows = venv / "Lib" / "site-packages"
    if windows.exists():
        candidates.append(windows)
    return candidates


def _product_site_packages() -> List[Path]:
    """The running interpreter's site-packages, read off ``sys.path``.

    Not ``site.getsitepackages()``: under a virtualenv that reports the base
    interpreter's directories too, which is where a *different* copy of these
    packages can sit. ``sys.path`` is what imports actually resolve against.
    """
    return [Path(entry) for entry in sys.path if entry.endswith("site-packages")]


def default_environments() -> Dict[str, List[Path]]:
    """Where to look for each environment's packages."""
    return {
        ISOLATED_ENV: _isolated_site_packages(),
        PRODUCT_ENV: _product_site_packages(),
    }


def environment_fingerprint(
    environments: Optional[Dict[str, Sequence[Path]]] = None,
) -> Optional[Dict[str, Any]]:
    """The installed stack, per environment, or ``None`` if nothing is readable.

    ``None`` rather than a dict full of ``None``s on purpose: two environments
    that know nothing about themselves would otherwise compare equal and claim
    a verified comparability neither of them has. Absence has to stay
    distinguishable from agreement -- the same distinction
    ``index_fingerprint.fingerprint_is_stale`` draws between "unknown" and
    "mismatch".

    Args:
        environments: ``{name: [site-packages, ...]}``. Defaults to the
            isolated venv and this interpreter's own packages.

    Returns:
        ``{"schema": ..., "packages": {"<env>:<package>": version | None}}``,
        or ``None`` when not a single version could be read.
    """
    environments = environments if environments is not None else default_environments()
    packages: Dict[str, Optional[str]] = {}
    for env_name, roots in environments.items():
        for package, version in read_stack_versions(roots).items():
            packages[f"{env_name}:{package}"] = version
    if not any(version is not None for version in packages.values()):
        return None
    return {"schema": FINGERPRINT_SCHEMA, "packages": packages}


def _comparable_pairs(left: Dict[str, Any], right: Dict[str, Any]) -> List[tuple]:
    """``(key, left_version, right_version)`` for keys both sides actually know."""
    left_packages = left.get("packages") or {}
    right_packages = right.get("packages") or {}
    return [
        (key, left_packages[key], right_packages[key])
        for key in sorted(set(left_packages) & set(right_packages))
        if left_packages[key] is not None and right_packages[key] is not None
    ]


def _environments_only_one_side_measured(
    left: Dict[str, Any], right: Dict[str, Any]
) -> List[str]:
    """Declared environments one side read and the other did not read at all.

    Distinct from a package missing on one side, which ``_comparable_pairs``
    already skips for good reason. An environment with NO version on one side
    was not measured there -- the campaign ran with an interpreter that has
    none of its packages -- so it makes no claim the other side's versions can
    be checked against (issue #132).
    """
    left_packages = left.get("packages") or {}
    right_packages = right.get("packages") or {}
    measured_left, measured_right = set(), set()
    for packages, measured in ((left_packages, measured_left), (right_packages, measured_right)):
        for key, version in packages.items():
            if version is not None:
                measured.add(key.split(":", 1)[0])
    declared = {key.split(":", 1)[0] for key in set(left_packages) | set(right_packages)}
    return sorted(
        env for env in declared if (env in measured_left) != (env in measured_right)
    )


def compare(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> str:
    """``MATCH``, ``DIFFERS`` or ``UNKNOWN`` for two fingerprints.

    ``UNKNOWN`` is not a hedge, it is the honest third answer, and keeping it
    separate from ``DIFFERS`` is what stops this feature from disarming
    recovery mode across the whole existing ledger: entries written before
    schema v3 carry no fingerprint, and calling them "different" would discard
    the very evidence issue #92 was built on. Callers decide what to do with
    an unverified comparison; they are told which one they have.

    A schema change also reads as ``UNKNOWN``: a later schema may track a
    different package set, and comparing across two definitions of "the stack"
    is a guess dressed as a measurement.
    """
    if left is None or right is None:
        return UNKNOWN
    if left.get("schema") != right.get("schema"):
        return UNKNOWN
    pairs = _comparable_pairs(left, right)
    if not pairs:
        return UNKNOWN
    if not all(a == b for _, a, b in pairs):
        # A stack that provably moved is not made uncertain by a second
        # environment nobody read: DIFFERS is the stronger answer and wins.
        return DIFFERS
    # Everything both sides read agrees -- but agreement over half the stack
    # is not verification of the stack. An entry whose product environment was
    # never measured (a campaign launched with the wrong interpreter, issue
    # #132) says nothing about retrieval or generation, and must not read as
    # verified against one that does.
    if _environments_only_one_side_measured(left, right):
        return UNKNOWN
    return MATCH


def describe_difference(
    ledger_fingerprint: Optional[Dict[str, Any]],
    launch_fingerprint: Optional[Dict[str, Any]],
) -> List[str]:
    """Field-level differences, phrased like ``loop._diff_comparable_views``.

    Empty when the two match, when either is missing, or when they share no
    known package -- in all three cases there is no difference to name, which
    is exactly why ``compare`` is the function that says whether one exists.
    """
    if ledger_fingerprint is None or launch_fingerprint is None:
        return []
    if ledger_fingerprint.get("schema") != launch_fingerprint.get("schema"):
        return []
    return [
        f"{key}: this launch {launch!r}, ledger {stored!r}"
        for key, stored, launch in _comparable_pairs(ledger_fingerprint, launch_fingerprint)
        if stored != launch
    ]
