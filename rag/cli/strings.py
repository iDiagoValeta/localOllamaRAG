"""
MonkeyGrab CLI string tables — bilingual support (ES / EN).

All user-visible text is defined here as a key → string mapping.  The active
language is read once from the ``MONKEYGRAB_LANG`` environment variable
(``"es"`` default; ``"en"`` for English).

Usage (internal)::

    from rag.cli.strings import s
    s("farewell.msg")                     # -> "Sesión finalizada." / "Session ended."
    s("indexing.done", total=42)          # -> "42 fragmentos indexados"

Keys that are not found fall back to Spanish, then to the key itself, so raw
values (paths, model names, numbers) pass through unchanged.

Selecting the language at launch (PowerShell / cmd / bash)::

    # PowerShell (Windows)
    $env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py

    # cmd.exe (Windows)
    set MONKEYGRAB_LANG=en && python rag/chat_pdfs.py

    # bash / zsh (Linux / macOS)
    MONKEYGRAB_LANG=en python rag/chat_pdfs.py

The default language is Spanish (``"es"``).  Supported values: ``"es"``, ``"en"``.
"""

from __future__ import annotations

import os
from typing import Optional

# ─────────────────────────────────────────────
# SPANISH (default)
# ─────────────────────────────────────────────

_ES: dict = {
    # Branding
    "brand.subtitle":           "local PDF RAG",
    "brand.academic":           "TFG · ETSINF · UPV · 2025-2026",

    # Modes
    "mode.chat.label":          "CHAT",
    "mode.rag.label":           "RAG",
    "mode.chat.desc":           "Conversación libre con historial.",
    "mode.rag.desc":            "Consulta los PDFs indexados y muestra fuentes.",
    "mode.chat.purpose":        "conversación libre",
    "mode.rag.purpose":         "consulta documental",

    # Init panel
    "init.mode_prefix":         "Modo inicial",
    "init.use_help":            "usa /ayuda para comandos",
    "init.corpus_title":        "Corpus",
    "init.pipeline_title":      "Pipeline",

    # Corpus section
    "corpus.pdfs":              "PDFs",
    "corpus.fragments":         "Fragmentos",
    "corpus.docs":              "Documentos",
    "corpus.docs_indexed":      "Documentos indexados",
    "corpus.folder":            "Carpeta",
    "corpus.vector_db":         "Base vectorial",
    "corpus.collection":        "Colección",

    # Pipeline section
    "pipeline.extractor":       "Extractor",
    "pipeline.search":          "Búsqueda",
    "pipeline.reranker":        "Reranker",
    "pipeline.chunks":          "Chunks",
    "pipeline.search.hybrid":   "híbrida (semántica + keywords)",
    "pipeline.search.semantic": "solo semántica",

    # Extractor names
    "extractor.pymupdf":        "pymupdf4llm",
    "extractor.pypdf":          "pypdf (fallback)",

    # Models table
    "models.role":              "Rol",
    "models.model":             "Modelo",
    "models.tag":               "Tag",
    "models.use":               "Uso",
    "model.use.rag":            "respuestas",
    "model.use.chat":           "general",
    "model.use.embed":          "índice",
    "model.use.contextual":     "indexación",
    "model.use.recomp":         "síntesis",
    "model.use.ocr":            "imágenes",

    # Model roles
    "role.rag":                 "RAG",
    "role.chat":                "Chat",
    "role.embed":               "Embeddings",
    "role.contextual":          "Contextual",
    "role.recomp":              "RECOMP",
    "role.ocr":                 "OCR",

    # Help / welcome
    "help.modes_title":         "Modos",
    "help.commands_title":      "Comandos",
    "help.command_col":         "Comando",
    "help.desc_col":            "Descripción",
    "help.shortcuts_title":     "Atajos",
    "help.shortcut.arrows":     "Recuperar consultas anteriores del historial.",
    "help.shortcut.tab":        "Autocompletar con fuzzy matching sobre comandos /.",
    "help.shortcut.ctrlc":      "Salir guardando el historial.",

    # Command descriptions (COMMANDS list)
    "cmd.rag.desc":             "Activar modo RAG",
    "cmd.chat.desc":            "Activar modo CHAT",
    "cmd.limpiar.desc":         "Limpiar historial",
    "cmd.stats.desc":           "Estado, modelos y base vectorial",
    "cmd.docs.desc":            "Documentos indexados",
    "cmd.temas.desc":           "Resumen de contenidos",
    "cmd.reindex.desc":         "Reconstruir el índice",
    "cmd.ayuda.desc":           "Mostrar esta ayuda",
    "cmd.salir.desc":           "Terminar la sesión",

    # Alias descriptions
    "alias.clear.desc":         "alias de /limpiar",
    "alias.help.desc":          "alias de /ayuda",
    "alias.exit.desc":          "alias de /salir",

    # Ollama / exceptions
    "ollama.title":             "Ollama",
    "ollama.advice":            "Arranca el servidor con `ollama serve` antes de consultar.",
    "exception.advice":         "Comprueba que Ollama esté activo y que el modelo exista localmente.",

    # Pipeline feedback
    "pipeline.start":           "Iniciando búsqueda...",
    "phase.search":             "búsqueda",
    "phase.search.detail":      "semántica + léxica (RRF)",
    "phase.rerank":             "rerank",
    "phase.rerank.detail":      "cross-encoder · {n} candidatos",
    "phase.expand":             "expansión",
    "phase.expand.detail":      "chunks adyacentes a los top-{n}",
    "phase.synthesis":          "síntesis",
    "phase.synthesis.detail":   "RECOMP · {model}",
    "phase.generation":         "generación",
    "phase.generation.detail":  "streaming · {model}",

    # Response
    "response.header.rag":      "RAG",
    "response.header.chat":     "Chat",
    "response.empty":           "(sin respuesta)",

    # Sources
    "sources.title":            "Fuentes ({n})",
    "sources.col.doc":          "Documento",
    "sources.col.pages":        "Páginas",
    "sources.col.score":        "Score",
    "sources.footer.cited":     "{n} fuentes citadas",
    "sources.footer.none":      "sin fuentes",

    # Stats
    "stats.title":              "Estado",
    "stats.dashboard.title":    "Estado del sistema",
    "stats.metric_col":         "Métrica",
    "stats.value_col":          "Valor",

    # Docs table
    "docs.title":               "Documentos",
    "docs.col.num":             "#",
    "docs.col.doc":             "Documento",
    "docs.col.pages":           "Págs.",
    "docs.col.frags":           "Frag.",
    "docs.col.types":           "Tipos",
    "docs.none":                "No hay documentos indexados.",

    # Topics table
    "topics.title":             "Contenidos",
    "topics.col.doc":           "Documento",
    "topics.col.pages":         "Págs.",
    "topics.col.frags":         "Frag.",
    "topics.col.terms":         "Términos frecuentes",
    "topics.tip":               "Escribe una pregunta concreta o usa /docs para revisar el corpus.",

    # Edge cases
    "no_results.msg":           "No se encontró información relevante en los documentos.",
    "no_results.reason1":       "- La información puede no estar indexada",
    "no_results.reason2":       "- La pregunta puede necesitar más detalle",
    "no_results.reason3":       "- El tema puede estar fuera del corpus",
    "no_results.tip":           "Prueba con /temas o reformula la consulta.",
    "out_of_scope.title":       "Fuera de ámbito",
    "out_of_scope.msg":         "Pregunta fuera de ámbito (score {score:.4f} < {threshold})",
    "out_of_scope.tip":         "Usa /temas para explorar el corpus.",
    "question_too_short":       "Pregunta demasiado corta. Formula una pregunta concreta o usa /chat.",
    "no_pdfs":                  "No existe la carpeta de PDFs o está vacía: {folder}",
    "unknown_cmd.msg":          "Comando no reconocido: {cmd}",
    "unknown_cmd.suggestion":   "¿Quisiste decir: {hint}?  (usa /ayuda)",
    "unknown_cmd.hint":         "usa /ayuda para ver los comandos disponibles",

    # Reindex
    "reindex.title":            "Reindex",
    "reindex.start":            "Reindexando documentos: se reconstruirá la base vectorial.",
    "reindex.db_deleted":       "base de datos anterior eliminada",
    "reindex.complete":         "Reindexación completada: {total} fragmentos",
    "reindex.restart":          "Reinicia el programa para usar la nueva base de datos",

    # History
    "history.loaded":           "historial restaurado: {n} mensajes",
    "history.cleared":          "historial limpiado",

    # Farewell / session summary
    "farewell.msg":             "Sesión finalizada.",
    "farewell.title":           "Resumen de sesión",
    "farewell.duration":        "Duración",
    "farewell.rag_queries":     "Consultas RAG",
    "farewell.chat_queries":    "Consultas CHAT",
    "farewell.models":          "Modelos usados",

    # Toolbar
    "toolbar.shortcuts":        "·  /ayuda  /stats  Ctrl-C = salir",

    # Misc
    "no_config":                "no configurado",
    "indexing.done":            "{total} fragmentos indexados",
    "indexing.none":            "No se indexaron documentos.",
}


# ─────────────────────────────────────────────
# ENGLISH
# ─────────────────────────────────────────────

_EN: dict = {
    # Branding
    "brand.subtitle":           "local PDF RAG",
    "brand.academic":           "FDP · ETSINF · UPV · 2025-2026",

    # Modes
    "mode.chat.label":          "CHAT",
    "mode.rag.label":           "RAG",
    "mode.chat.desc":           "Free conversation with message history.",
    "mode.rag.desc":            "Query indexed PDFs and display sources.",
    "mode.chat.purpose":        "free conversation",
    "mode.rag.purpose":         "document query",

    # Init panel
    "init.mode_prefix":         "Initial mode",
    "init.use_help":            "use /help for commands",
    "init.corpus_title":        "Corpus",
    "init.pipeline_title":      "Pipeline",

    # Corpus section
    "corpus.pdfs":              "PDFs",
    "corpus.fragments":         "Fragments",
    "corpus.docs":              "Documents",
    "corpus.docs_indexed":      "Indexed documents",
    "corpus.folder":            "Folder",
    "corpus.vector_db":         "Vector DB",
    "corpus.collection":        "Collection",

    # Pipeline section
    "pipeline.extractor":       "Extractor",
    "pipeline.search":          "Search",
    "pipeline.reranker":        "Reranker",
    "pipeline.chunks":          "Chunks",
    "pipeline.search.hybrid":   "hybrid (semantic + keywords)",
    "pipeline.search.semantic": "semantic only",

    # Extractor names
    "extractor.pymupdf":        "pymupdf4llm",
    "extractor.pypdf":          "pypdf (fallback)",

    # Models table
    "models.role":              "Role",
    "models.model":             "Model",
    "models.tag":               "Tag",
    "models.use":               "Use",
    "model.use.rag":            "responses",
    "model.use.chat":           "general",
    "model.use.embed":          "index",
    "model.use.contextual":     "indexing",
    "model.use.recomp":         "synthesis",
    "model.use.ocr":            "images",

    # Model roles
    "role.rag":                 "RAG",
    "role.chat":                "Chat",
    "role.embed":               "Embeddings",
    "role.contextual":          "Contextual",
    "role.recomp":              "RECOMP",
    "role.ocr":                 "OCR",

    # Help / welcome
    "help.modes_title":         "Modes",
    "help.commands_title":      "Commands",
    "help.command_col":         "Command",
    "help.desc_col":            "Description",
    "help.shortcuts_title":     "Shortcuts",
    "help.shortcut.arrows":     "Browse previous queries from history.",
    "help.shortcut.tab":        "Fuzzy autocomplete slash commands.",
    "help.shortcut.ctrlc":      "Exit saving history.",

    # Command descriptions
    "cmd.rag.desc":             "Switch to RAG mode",
    "cmd.chat.desc":            "Switch to CHAT mode",
    "cmd.limpiar.desc":         "Clear history",
    "cmd.stats.desc":           "System status, models, and vector DB",
    "cmd.docs.desc":            "Indexed documents",
    "cmd.temas.desc":           "Content overview",
    "cmd.reindex.desc":         "Rebuild the index",
    "cmd.ayuda.desc":           "Show this help",
    "cmd.salir.desc":           "End session",

    # Alias descriptions
    "alias.clear.desc":         "alias for /limpiar",
    "alias.help.desc":          "alias for /ayuda",
    "alias.exit.desc":          "alias for /salir",

    # Ollama / exceptions
    "ollama.title":             "Ollama",
    "ollama.advice":            "Start the server with `ollama serve` before querying.",
    "exception.advice":         "Make sure Ollama is running and the model is available locally.",

    # Pipeline feedback
    "pipeline.start":           "Starting search...",
    "phase.search":             "search",
    "phase.search.detail":      "semantic + lexical (RRF)",
    "phase.rerank":             "rerank",
    "phase.rerank.detail":      "cross-encoder · {n} candidates",
    "phase.expand":             "expand",
    "phase.expand.detail":      "adjacent chunks around top-{n}",
    "phase.synthesis":          "synthesis",
    "phase.synthesis.detail":   "RECOMP · {model}",
    "phase.generation":         "generation",
    "phase.generation.detail":  "streaming · {model}",

    # Response
    "response.header.rag":      "RAG",
    "response.header.chat":     "Chat",
    "response.empty":           "(no response)",

    # Sources
    "sources.title":            "Sources ({n})",
    "sources.col.doc":          "Document",
    "sources.col.pages":        "Pages",
    "sources.col.score":        "Score",
    "sources.footer.cited":     "{n} sources cited",
    "sources.footer.none":      "no sources",

    # Stats
    "stats.title":              "Status",
    "stats.dashboard.title":    "System status",
    "stats.metric_col":         "Metric",
    "stats.value_col":          "Value",

    # Docs table
    "docs.title":               "Documents",
    "docs.col.num":             "#",
    "docs.col.doc":             "Document",
    "docs.col.pages":           "Pages",
    "docs.col.frags":           "Frags.",
    "docs.col.types":           "Types",
    "docs.none":                "No documents indexed.",

    # Topics table
    "topics.title":             "Contents",
    "topics.col.doc":           "Document",
    "topics.col.pages":         "Pages",
    "topics.col.frags":         "Frags.",
    "topics.col.terms":         "Frequent terms",
    "topics.tip":               "Ask a specific question or use /docs to browse the corpus.",

    # Edge cases
    "no_results.msg":           "No relevant information found in the documents.",
    "no_results.reason1":       "- The information may not be indexed",
    "no_results.reason2":       "- The question may need more detail",
    "no_results.reason3":       "- The topic may be outside the corpus",
    "no_results.tip":           "Try /temas or rephrase your query.",
    "out_of_scope.title":       "Out of scope",
    "out_of_scope.msg":         "Query out of scope (score {score:.4f} < {threshold})",
    "out_of_scope.tip":         "Use /temas to explore the corpus.",
    "question_too_short":       "Query too short. Ask a specific question or switch to /chat.",
    "no_pdfs":                  "PDF folder not found or empty: {folder}",
    "unknown_cmd.msg":          "Unknown command: {cmd}",
    "unknown_cmd.suggestion":   "Did you mean: {hint}?  (use /ayuda)",
    "unknown_cmd.hint":         "use /ayuda to see available commands",

    # Reindex
    "reindex.title":            "Reindex",
    "reindex.start":            "Re-indexing: the vector database will be rebuilt.",
    "reindex.db_deleted":       "previous database deleted",
    "reindex.complete":         "Re-indexing complete: {total} fragments",
    "reindex.restart":          "Restart the program to use the new database",

    # History
    "history.loaded":           "history restored: {n} messages",
    "history.cleared":          "history cleared",

    # Farewell / session summary
    "farewell.msg":             "Session ended.",
    "farewell.title":           "Session summary",
    "farewell.duration":        "Duration",
    "farewell.rag_queries":     "RAG queries",
    "farewell.chat_queries":    "CHAT queries",
    "farewell.models":          "Models used",

    # Toolbar
    "toolbar.shortcuts":        "·  /ayuda  /stats  Ctrl-C = exit",

    # Misc
    "no_config":                "not configured",
    "indexing.done":            "{total} fragments indexed",
    "indexing.none":            "No documents were indexed.",
}

# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

LANGUAGES: dict = {"es": _ES, "en": _EN}


def s(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """Return the localized string for *key* in *lang*.

    Falls back to Spanish if the key is not in *lang*, then to the key itself
    so that raw values (paths, model names, numbers) pass through unchanged.

    Args:
        key:    String key defined in this module, or any literal string.
        lang:   Language code (``"es"`` / ``"en"``).  If *None*, the value of
                the ``MONKEYGRAB_LANG`` environment variable is used (default
                ``"es"``).
        **kwargs: Format arguments substituted into the resolved string via
                  ``str.format``.

    Returns:
        Localized and formatted string.
    """
    if lang is None:
        lang = os.getenv("MONKEYGRAB_LANG", "es").strip().lower()
    if lang not in LANGUAGES:
        lang = "es"
    table = LANGUAGES[lang]
    text = table.get(key, _ES.get(key, key))
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):
        return text
