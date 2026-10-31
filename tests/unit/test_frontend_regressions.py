"""Regression checks for frontend-only defects (#327, #328 and #538).

Neither defect reaches the Python suite through runtime behavior: the
frontend's checks are ``tsc --noEmit`` and a Vite build, so a store row left
stale after a reindex, or a PDF toolbar tooltip still in Spanish under the
English UI, stays green until someone clicks through the flow by hand.

There is no JavaScript test runner in this repository (``package.json``
exposes ``lint`` and ``build`` only), and the fast CI gate that runs this
module installs pytest and nothing else. The checks are therefore structural
over the TypeScript source, the same approach ``test_env_example_drift.py``
uses for frontend environment reads. Each assertion names the user-visible
symptom it prevents rather than mirroring one particular implementation.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "rag" / "web" / "frontend" / "src"
APP = FRONTEND / "App.tsx"
I18N = FRONTEND / "lib" / "i18n.ts"
PDF_PANE = FRONTEND / "components" / "PdfPane.tsx"


def _read(path: Path) -> str:
    """Return a frontend source file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _locale_keys() -> dict[str, set[str]]:
    """Return the key set of each locale table in ``i18n.ts``.

    The tables are one level deep and keys appear either at the start of a
    4-space-indented line or after a comma on the same line, so the shipped
    regex is enough; a TypeScript parser in Python would drag a toolchain
    into a stdlib-only CI job for no extra safety.
    """
    source = _read(I18N)
    keys: dict[str, set[str]] = {}
    for lang in ("es", "en", "ca"):
        block = re.search(rf"\n  {lang}: \{{\n(.*?)\n  \}},", source, re.S)
        assert block, f"i18n.ts has no {lang} table"
        keys[lang] = set(re.findall(r"(?:^|,)\s*([A-Za-z_]\w*):\s*['\"]", block.group(1), re.M))
    return keys


def test_every_i18n_key_exists_in_all_three_languages() -> None:
    """A key added to one table only would render blank in the other two."""
    keys = _locale_keys()
    assert keys["es"] == keys["en"] == keys["ca"]


def _referenced_keys(keys: set[str], sources: list[str]) -> set[str]:
    """Keys a component reads: ``strings.key``/``T.key`` or a quoted ``'key'``.

    The quoted form covers a key chosen at runtime from a literal; nothing in
    ``src/`` indexes the table with a computed expression, so these two shapes
    are every way a key is reached.
    """
    code = "\n".join(sources)
    return {key for key in keys if re.search(rf"\.{key}\b|['\"`]{key}['\"`]", code)}


def test_every_i18n_key_is_read_by_some_component() -> None:
    """A key nothing reads is copy translated three times for no screen (#538).

    knip cannot see these: they are object keys, not exports.
    """
    sources = [
        _read(path) for path in sorted(FRONTEND.rglob("*.ts*")) if path != I18N
    ]
    keys = _locale_keys()["es"]
    unread = sorted(keys - _referenced_keys(keys, sources))
    assert not unread, f"i18n.ts keys no component reads: {unread}"


def test_the_key_reference_check_tells_a_read_key_from_an_unread_one() -> None:
    """Guards the guard: a mention as part of a longer name is not a read."""
    sources = ["<p>{T.studyHint}</p>", "const fb = { code: 'retryError' };", "x.studyPagesShort"]
    keys = {"studyHint", "retryError", "studyPages", "youLabel"}
    assert _referenced_keys(keys, sources) == {"studyHint", "retryError"}


def test_question_too_short_error_is_translated() -> None:
    """The 400 body must not surface its hard-coded Spanish ``message``.

    ``streamSSE`` maps backend error codes through its ``fb`` table; a code
    missing there falls back to ``data.message``, which the backend writes in
    Spanish regardless of the UI language.
    """
    mapped = re.search(r"\bquestion_too_short:\s*T\.(\w+)", _read(APP))
    assert mapped, "App.tsx does not map the question_too_short error code"

    key = mapped.group(1)
    for lang, lang_keys in _locale_keys().items():
        assert key in lang_keys, f"{lang} table has no {key}"


def test_pdf_pane_tooltips_come_from_the_active_language() -> None:
    """The PDF toolbar's two titles were the last Spanish literals in the UI."""
    pane = _read(PDF_PANE)
    assert re.search(r"strings:\s*Strings", pane), "PdfPane does not accept the locale table"
    assert "strings.openInBrowser" in pane
    assert "strings.closeEsc" in pane
    assert 'title="Abrir en el navegador"' not in pane
    assert 'title="Cerrar (Esc)"' not in pane

    keys = _locale_keys()
    for key in ("openInBrowser", "closeEsc"):
        for lang, lang_keys in keys.items():
            assert key in lang_keys, f"{lang} table has no {key}"

    call = re.search(r"<PdfPane\b.*?/>", _read(APP), re.S)
    assert call and "strings={T}" in call.group(0), "App.tsx does not pass strings to PdfPane"


def test_init_poller_reloads_stores_when_indexing_finishes() -> None:
    """After a reindex the sidebar store row must stop saying "not indexed".

    ``isInitialized`` is already true on that path, so the control-panel effect
    never re-runs; the poller's success branch is the only place that can
    refresh the rows without the user switching store.
    """
    source = _read(APP)
    ok_branch = re.search(r"if \(initData\.ok\) \{(.*?)\n        \} else \{", source, re.S)
    assert ok_branch, "could not locate the successful init branch in App.tsx"
    assert "loadStores()" in ok_branch.group(1), (
        "the init poller does not reload the store list on success"
    )
