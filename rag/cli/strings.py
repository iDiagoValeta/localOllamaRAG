"""MonkeyGrab CLI string tables — trilingual support (ES / EN / CA).

All user-visible text is defined here as a key → string mapping.  The active
language is read once from the ``MONKEYGRAB_LANG`` environment variable
(``"es"`` default; ``"en"`` for English; ``"ca"`` for Valencian).

Usage (internal)::

    from rag.cli.strings import s
    s("farewell.msg")                     # -> "Sesión finalizada." / "Session ended." / "Sessió finalitzada."
    s("indexing.done", total=42)          # -> "42 fragmentos indexados"

Keys that are not found fall back to Spanish, then to the key itself, so raw
values (paths, model names, numbers) pass through unchanged.

Selecting the language at launch (PowerShell / cmd / bash)::

    $env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py
    $env:MONKEYGRAB_LANG = "ca"; python rag/chat_pdfs.py

    set MONKEYGRAB_LANG=en && python rag/chat_pdfs.py

    MONKEYGRAB_LANG=en python rag/chat_pdfs.py
    MONKEYGRAB_LANG=ca python rag/chat_pdfs.py

The default language is Spanish (``"es"``).  Supported values: ``"es"``, ``"en"``, ``"ca"``.
"""

from __future__ import annotations

import os
from typing import Optional


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
    "mode.switch.prefix":       "modo",

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
    "pipeline.chunks":          "Fragmentos",
    "pipeline.overlap":         "solape",
    "pipeline.flags":           "Opciones",
    "pipeline.search.hybrid":   "híbrida (semántica + BM25)",
    "pipeline.search.semantic": "solo semántica",
    "state.on":                 "activo",
    "state.off":                "inactivo",

    # Extractor names
    "extractor.mineru":         "MinerU",

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

    # Model roles
    "role.rag":                 "RAG",
    "role.chat":                "Chat",
    "role.embed":               "Embeddings",
    "role.contextual":          "Contextual",
    "role.recomp":              "RECOMP",

    # Help / welcome
    "help.modes_title":         "Modos",
    "help.commands_title":      "Comandos",
    "help.command_col":         "Comando",
    "help.desc_col":            "Descripción",
    "help.shortcuts_title":     "Atajos",
    "help.shortcut.arrows":     "Recuperar consultas anteriores del historial.",
    "help.shortcut.tab":        "Autocompletar con fuzzy matching sobre comandos /.",
    "help.shortcut.ctrlc":      "Salir guardando el historial.",
    "help.shortcuts.inline":    "historial  Tab  Ctrl-C",

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
    "ollama.active":            "Ollama activo en {base} · {n} modelos locales",
    "ollama.unavailable":       "Ollama no responde en {base}: {error}",
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
    "docs.col.pct":             "% Corpus",
    "docs.col.types":           "Tipos",
    "docs.inline.meta":         "(págs: {pages}, frag: {frags}, %: {pct}, tipos: {types})",
    "docs.none":                "No hay documentos indexados.",

    # Topics table
    "topics.title":             "Contenidos",
    "topics.col.doc":           "Documento",
    "topics.col.pages":         "Págs.",
    "topics.col.frags":         "Frag.",
    "topics.col.chunks":        "Chunks",
    "topics.col.terms":         "Términos frecuentes",
    "topics.inline.meta":       "(págs: {pages}, frag: {frags}, chunks: {chunks})",
    "topics.tip":               "Escribe una pregunta concreta o usa /docs para revisar el corpus.",

    # Edge cases
    "no_results.msg":           "No se encontró información relevante en los documentos.",
    "no_results.reason1":       "- La información puede no estar indexada",
    "no_results.reason2":       "- La pregunta puede necesitar más detalle",
    "no_results.reason3":       "- El tema puede estar fuera del corpus",
    "no_results.tip":           "Prueba con /temas o reformula la consulta.",
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


_EN: dict = {
    # Branding
    "brand.subtitle":           "local PDF RAG",
    "brand.academic":           "TFG · ETSINF · UPV · 2025-2026",

    # Modes
    "mode.chat.label":          "CHAT",
    "mode.rag.label":           "RAG",
    "mode.chat.desc":           "Free conversation with message history.",
    "mode.rag.desc":            "Query indexed PDFs and display sources.",
    "mode.chat.purpose":        "free conversation",
    "mode.rag.purpose":         "document query",
    "mode.switch.prefix":       "mode",

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
    "pipeline.overlap":         "overlap",
    "pipeline.flags":           "Flags",
    "pipeline.search.hybrid":   "hybrid (semantic + BM25)",
    "pipeline.search.semantic": "semantic only",
    "state.on":                 "on",
    "state.off":                "off",

    # Extractor names
    "extractor.mineru":         "MinerU",

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

    # Model roles
    "role.rag":                 "RAG",
    "role.chat":                "Chat",
    "role.embed":               "Embeddings",
    "role.contextual":          "Contextual",
    "role.recomp":              "RECOMP",

    # Help / welcome
    "help.modes_title":         "Modes",
    "help.commands_title":      "Commands",
    "help.command_col":         "Command",
    "help.desc_col":            "Description",
    "help.shortcuts_title":     "Shortcuts",
    "help.shortcut.arrows":     "Browse previous queries from history.",
    "help.shortcut.tab":        "Fuzzy autocomplete slash commands.",
    "help.shortcut.ctrlc":      "Exit saving history.",
    "help.shortcuts.inline":    "history  Tab  Ctrl-C",

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
    "ollama.active":            "Ollama active at {base} · {n} local models",
    "ollama.unavailable":       "Ollama is not responding at {base}: {error}",
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
    "docs.col.pct":             "% Corpus",
    "docs.col.types":           "Types",
    "docs.inline.meta":         "(pages: {pages}, frags: {frags}, %: {pct}, types: {types})",
    "docs.none":                "No documents indexed.",

    # Topics table
    "topics.title":             "Contents",
    "topics.col.doc":           "Document",
    "topics.col.pages":         "Pages",
    "topics.col.frags":         "Frags.",
    "topics.col.chunks":        "Chunks",
    "topics.col.terms":         "Frequent terms",
    "topics.inline.meta":       "(pages: {pages}, frags: {frags}, chunks: {chunks})",
    "topics.tip":               "Ask a specific question or use /docs to browse the corpus.",

    # Edge cases
    "no_results.msg":           "No relevant information found in the documents.",
    "no_results.reason1":       "- The information may not be indexed",
    "no_results.reason2":       "- The question may need more detail",
    "no_results.reason3":       "- The topic may be outside the corpus",
    "no_results.tip":           "Try /topics or rephrase your query.",
    "question_too_short":       "Query too short. Ask a specific question or switch to /chat.",
    "no_pdfs":                  "PDF folder not found or empty: {folder}",
    "unknown_cmd.msg":          "Unknown command: {cmd}",
    "unknown_cmd.suggestion":   "Did you mean: {hint}?  (use /help)",
    "unknown_cmd.hint":         "use /help to see available commands",

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
    "toolbar.shortcuts":        "·  /help  /stats  Ctrl-C = exit",

    # Misc
    "no_config":                "not configured",
    "indexing.done":            "{total} fragments indexed",
    "indexing.none":            "No documents were indexed.",
}


_CA: dict = {
    # Branding
    "brand.subtitle":           "local PDF RAG",
    "brand.academic":           "TFG · ETSINF · UPV · 2025-2026",

    # Modes
    "mode.chat.label":          "CHAT",
    "mode.rag.label":           "RAG",
    "mode.chat.desc":           "Conversa lliure amb historial.",
    "mode.rag.desc":            "Consulta els PDFs indexats i mostra les fonts.",
    "mode.chat.purpose":        "conversa lliure",
    "mode.rag.purpose":         "consulta documental",
    "mode.switch.prefix":       "mode",

    # Init panel
    "init.mode_prefix":         "Mode inicial",
    "init.use_help":            "usa /ajuda per als comandos",
    "init.corpus_title":        "Corpus",
    "init.pipeline_title":      "Pipeline",

    # Corpus section
    "corpus.pdfs":              "PDFs",
    "corpus.fragments":         "Fragments",
    "corpus.docs":              "Documents",
    "corpus.docs_indexed":      "Documents indexats",
    "corpus.folder":            "Carpeta",
    "corpus.vector_db":         "Base vectorial",
    "corpus.collection":        "Col·lecció",

    # Pipeline section
    "pipeline.extractor":       "Extractor",
    "pipeline.search":          "Cerca",
    "pipeline.reranker":        "Reranker",
    "pipeline.chunks":          "Fragments",
    "pipeline.overlap":         "solapament",
    "pipeline.flags":           "Opcions",
    "pipeline.search.hybrid":   "híbrida (semàntica + BM25)",
    "pipeline.search.semantic": "només semàntica",
    "state.on":                 "actiu",
    "state.off":                "inactiu",

    # Extractor names
    "extractor.mineru":         "MinerU",

    # Models table
    "models.role":              "Rol",
    "models.model":             "Model",
    "models.tag":               "Tag",
    "models.use":               "Ús",
    "model.use.rag":            "respostes",
    "model.use.chat":           "general",
    "model.use.embed":          "índex",
    "model.use.contextual":     "indexació",
    "model.use.recomp":         "síntesi",

    # Model roles
    "role.rag":                 "RAG",
    "role.chat":                "Xat",
    "role.embed":               "Embeddings",
    "role.contextual":          "Contextual",
    "role.recomp":              "RECOMP",

    # Help / welcome
    "help.modes_title":         "Modes",
    "help.commands_title":      "Comandos",
    "help.command_col":         "Comando",
    "help.desc_col":            "Descripció",
    "help.shortcuts_title":     "Dreceres",
    "help.shortcut.arrows":     "Recuperar consultes anteriors de l'historial.",
    "help.shortcut.tab":        "Autocompletar amb fuzzy matching sobre comandos /.",
    "help.shortcut.ctrlc":      "Eixir guardant l'historial.",
    "help.shortcuts.inline":    "historial  Tab  Ctrl-C",

    # Command descriptions (shared keys — CA command names set in commands.py)
    "cmd.rag.desc":             "Activar mode RAG",
    "cmd.chat.desc":            "Activar mode CHAT",
    "cmd.limpiar.desc":         "Netejar historial",
    "cmd.stats.desc":           "Estat, models i base vectorial",
    "cmd.docs.desc":            "Documents indexats",
    "cmd.temas.desc":           "Resum de continguts",
    "cmd.reindex.desc":         "Reconstruir l'índex",
    "cmd.ayuda.desc":           "Mostrar esta ajuda",
    "cmd.salir.desc":           "Finalitzar la sessió",

    # Alias descriptions
    "alias.clear.desc":         "àlies de /netejar",
    "alias.help.desc":          "àlies de /ajuda",
    "alias.exit.desc":          "àlies de /eixir",

    # Ollama / exceptions
    "ollama.title":             "Ollama",
    "ollama.active":            "Ollama actiu en {base} · {n} models locals",
    "ollama.unavailable":       "Ollama no respon en {base}: {error}",
    "ollama.advice":            "Inicia el servidor amb `ollama serve` abans de consultar.",
    "exception.advice":         "Comprova que Ollama estiga actiu i que el model existisca localment.",

    # Pipeline feedback
    "pipeline.start":           "Iniciant la cerca...",
    "phase.search":             "cerca",
    "phase.search.detail":      "semàntica + lèxica (RRF)",
    "phase.rerank":             "rerank",
    "phase.rerank.detail":      "cross-encoder · {n} candidats",
    "phase.expand":             "expansió",
    "phase.expand.detail":      "chunks adjacents als top-{n}",
    "phase.synthesis":          "síntesi",
    "phase.synthesis.detail":   "RECOMP · {model}",
    "phase.generation":         "generació",
    "phase.generation.detail":  "streaming · {model}",

    # Response
    "response.header.rag":      "RAG",
    "response.header.chat":     "Xat",
    "response.empty":           "(sense resposta)",

    # Sources
    "sources.title":            "Fonts ({n})",
    "sources.col.doc":          "Document",
    "sources.col.pages":        "Pàgines",
    "sources.col.score":        "Score",
    "sources.footer.cited":     "{n} fonts citades",
    "sources.footer.none":      "sense fonts",

    # Stats
    "stats.title":              "Estat",
    "stats.dashboard.title":    "Estat del sistema",
    "stats.metric_col":         "Mètrica",
    "stats.value_col":          "Valor",

    # Docs table
    "docs.title":               "Documents",
    "docs.col.num":             "#",
    "docs.col.doc":             "Document",
    "docs.col.pages":           "Pàgs.",
    "docs.col.frags":           "Frag.",
    "docs.col.pct":             "% Corpus",
    "docs.col.types":           "Tipus",
    "docs.inline.meta":         "(pàgs: {pages}, frag: {frags}, %: {pct}, tipus: {types})",
    "docs.none":                "No hi ha documents indexats.",

    # Topics table
    "topics.title":             "Continguts",
    "topics.col.doc":           "Document",
    "topics.col.pages":         "Pàgs.",
    "topics.col.frags":         "Frag.",
    "topics.col.chunks":        "Chunks",
    "topics.col.terms":         "Termes freqüents",
    "topics.inline.meta":       "(pàgs: {pages}, frag: {frags}, chunks: {chunks})",
    "topics.tip":               "Escriu una pregunta concreta o usa /docs per revisar el corpus.",

    # Edge cases
    "no_results.msg":           "No s'ha trobat informació rellevant en els documents.",
    "no_results.reason1":       "- La informació pot no estar indexada",
    "no_results.reason2":       "- La pregunta pot necessitar més detall",
    "no_results.reason3":       "- El tema pot estar fora del corpus",
    "no_results.tip":           "Prova amb /temes o reformula la consulta.",
    "question_too_short":       "Pregunta massa curta. Formula una pregunta concreta o usa /chat.",
    "no_pdfs":                  "No existix la carpeta de PDFs o està buida: {folder}",
    "unknown_cmd.msg":          "Comando no reconegut: {cmd}",
    "unknown_cmd.suggestion":   "Volies dir: {hint}?  (usa /ajuda)",
    "unknown_cmd.hint":         "usa /ajuda per veure els comandos disponibles",

    # Reindex
    "reindex.title":            "Reindex",
    "reindex.start":            "Reindexant documents: es reconstruirà la base vectorial.",
    "reindex.db_deleted":       "base de dades anterior eliminada",
    "reindex.complete":         "Reindexació completada: {total} fragments",
    "reindex.restart":          "Reinicia el programa per a usar la nova base de dades",

    # History
    "history.loaded":           "historial restaurat: {n} missatges",
    "history.cleared":          "historial netejat",

    # Farewell / session summary
    "farewell.msg":             "Sessió finalitzada.",
    "farewell.title":           "Resum de sessió",
    "farewell.duration":        "Durada",
    "farewell.rag_queries":     "Consultes RAG",
    "farewell.chat_queries":    "Consultes CHAT",
    "farewell.models":          "Models usats",

    # Toolbar
    "toolbar.shortcuts":        "·  /ajuda  /stats  Ctrl-C = eixir",

    # Misc
    "no_config":                "no configurat",
    "indexing.done":            "{total} fragments indexats",
    "indexing.none":            "No s'han indexat documents.",
}


LANGUAGES: dict = {"es": _ES, "en": _EN, "ca": _CA}


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
