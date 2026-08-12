"""MonkeyGrab CLI Application.

Main interactive loop for the MonkeyGrab command-line interface. Orchestrates
the user prompt, slash command dispatch, and integration with the RAG engine.
Uses the Display class (rich) for all visual output.

Usage:
    from rag.cli.app import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()

Dependencies:
    - FAISS through the MonkeyGrab vector-store port
    - rag.cli.display (ui singleton, QueryTimer, SessionStats)
    - rag.cli.commands (single source of truth for slash-commands)
    - A RAG engine module providing search, indexing, and generation functions
"""

import difflib
import os
import signal
from collections import Counter
from typing import Any, Dict, List, Tuple

from rag.cli.commands import ALIASES, COMMANDS, primary_commands
from rag.cli.display import QueryTimer, SessionStats, ui
from rag.cli.strings import s


# MAIN CLI CLASS

class MonkeyGrabCLI:
    """Main CLI application for MonkeyGrab.

    Encapsulates the interaction loop, state management (mode, history),
    session statistics, and slash command dispatch. Delegates RAG logic to the
    provided rag_engine module.
    """


    def __init__(self, rag_engine):
        """Initialize the CLI application.

        Args:
            rag_engine: Module or namespace providing RAG functions
                        (realizar_busqueda_hibrida, generar_respuesta, etc.)
                        and configuration constants.
        """
        self.rag = rag_engine
        self.mode = "chat"
        self.collection = None
        self.historial_chat: List[Dict[str, str]] = []
        self.session = SessionStats()

        handlers = {
            "/rag":     self._cmd_rag,
            "/chat":    self._cmd_chat,
            "/limpiar": self._cmd_clear,
            "/stats":   self._cmd_stats,
            "/ayuda":   self._cmd_help,
            "/docs":    self._cmd_docs,
            "/temas":   self._cmd_topics,
            "/reindex": self._cmd_reindex,
            "/salir":   self._cmd_exit,
        }
        self._commands: Dict[str, Any] = dict(handlers)
        for alias, target, *_ in ALIASES:
            if target in handlers:
                self._commands[alias] = handlers[target]
        self._validate_commands_registry()

    def _validate_commands_registry(self) -> None:
        """Fail loudly if commands.COMMANDS and _commands drift apart.

        Keeps the /ayuda screen, autocompleter and dispatcher in sync with a
        single source of truth; cheaper to fix at startup than to debug later.
        """
        listed = {cmd for cmd, _ in COMMANDS}
        registered = set(self._commands.keys()) - {alias for alias, *_ in ALIASES}
        missing = listed - registered
        orphaned = registered - listed
        if missing or orphaned:
            details = []
            if missing:
                details.append(f"listed but not handled: {sorted(missing)}")
            if orphaned:
                details.append(f"handled but not listed: {sorted(orphaned)}")
            raise RuntimeError("CLI command registry out of sync — " + "; ".join(details))

    # STARTUP

    def run(self) -> None:
        """Entry point. Initialize the system and start the main loop."""
        # On Windows, MKL/Fortran runtime libraries register their own SIGINT
        # handler that prints "forrtl: error 200" when Ctrl-C is pressed during
        # Python cleanup.  Installing Python's default_int_handler first ensures
        # that KeyboardInterrupt is raised cleanly and the Fortran handler never
        # runs.
        if os.name == "nt":
            signal.signal(signal.SIGINT, signal.default_int_handler)

        ui.logo()

        ok, detail = self._ollama_health()
        ui.ollama_status(ok, detail)

        self.collection = self.rag.obtener_vector_store()

        archivos_pdf = self._list_pdf_files()
        pdfs_count = len(archivos_pdf)

        if not archivos_pdf:
            ui.no_pdfs(self.rag.CARPETA_DOCS)

        if self.collection.count() == 0:
            total_chunks = self.rag.indexar_documentos(
                self.rag.CARPETA_DOCS, self.collection
            )
            if total_chunks == 0:
                ui.warning(ui._s("indexing.none"))
                return
            ui.success(ui._s("indexing.done", total=total_chunks))
        else:
            # A store just built by indexar_documentos above always matches
            # (indexar_documentos writes the fingerprint after a full run);
            # only a reused store can be stale, so the check only runs here.
            self._check_index_fingerprint()

        self._show_init_info(pdfs_count, self.collection.count())

        self.historial_chat = self.rag.cargar_historial()
        if self.historial_chat:
            ui.history_loaded(len(self.historial_chat))

        self._loop()

    # MAIN LOOP

    def _loop(self) -> None:
        """Read-dispatch-respond loop. Runs until the user exits."""
        while True:
            model = (self.rag.MODELO_CHAT if self.mode == "chat"
                     else self.rag.MODELO_RAG)

            try:
                pregunta = ui.read_input(self.mode, model).strip()
            except (EOFError, KeyboardInterrupt):
                self.rag.guardar_historial(self.historial_chat)
                ui.farewell(self.session)
                # Suppress any pending SIGINT so MKL/Fortran cleanup does not
                # re-fire it and print "forrtl: error 200" during Python exit.
                if os.name == "nt":
                    try:
                        signal.signal(signal.SIGINT, signal.SIG_IGN)
                    except Exception:
                        pass
                break

            if not pregunta:
                continue

            cmd_lower = pregunta.lower()
            if cmd_lower in self._commands:
                should_exit = self._commands[cmd_lower]()
                if should_exit:
                    break
                continue

            if pregunta.startswith('/'):
                suggestions = difflib.get_close_matches(
                    cmd_lower, [cmd for cmd, _ in primary_commands()], n=2, cutoff=0.6
                )
                ui.unknown_command(pregunta, suggestions=suggestions)
                continue

            if self.mode == "rag":
                self._process_rag(pregunta)
            else:
                self._process_chat(pregunta)

    # CHAT / RAG PROCESSING

    def _process_chat(self, pregunta: str) -> None:
        """Process a question in chat mode.

        Args:
            pregunta: The user's input question.
        """
        timer = QueryTimer()
        ui.response_header("chat", self.rag.MODELO_CHAT)

        try:
            respuesta = self._chat_stream(pregunta)
        except Exception as e:
            ui.exception("Error de chat", e)
            return

        if not ui.can_stream_responses():
            ui.render_response(respuesta)

        timer.mark("respuesta")
        ui.response_footer()
        self.session.tick_chat(timer.total, self.rag.MODELO_CHAT)

        self.historial_chat.append({"role": "user", "content": pregunta})
        self.historial_chat.append({"role": "assistant", "content": respuesta})
        self.rag.guardar_historial(self.historial_chat)

    def _chat_stream(self, pregunta: str) -> str:
        """Execute streaming chat using the chat model.

        Builds the message list from the system prompt and recent history,
        then streams the response token by token.

        Args:
            pregunta: The user's input question.

        Returns:
            The full assembled response text.
        """
        from monkeygrab.adapters.chat.ollama_chat import ollama_client_for

        messages = [{"role": "system", "content": self.rag.SYSTEM_PROMPT_CHAT}]
        mensajes_recientes = self.historial_chat[-(self.rag.MAX_HISTORIAL_MENSAJES):]
        messages.extend(mensajes_recientes)
        messages.append({"role": "user", "content": pregunta})

        stream = ollama_client_for(self.rag.OLLAMA_BASE_URL).chat(
            model=self.rag.MODELO_CHAT,
            messages=messages,
            stream=True,
            think=False,
            keep_alive=self.rag.OLLAMA_KEEP_ALIVE,
            options={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
        )

        respuesta = ""
        if ui.can_stream_responses():
            ui.begin_stream()
        for chunk in stream:
            content = (chunk.get("message", {}).get("content", "")
                       or chunk.get("content", ""))
            if content:
                respuesta += content
                if ui.can_stream_responses():
                    ui.stream_token(content)
        if ui.can_stream_responses():
            ui.end_stream()

        return respuesta

    def _process_rag(self, pregunta: str) -> None:
        """Process a question in RAG mode with per-phase visual feedback.

        Emits labelled pipeline phases (search / rerank / expand / generate)
        and a compact metrics summary at the end. Validates the question
        length, runs hybrid search, applies reranker filtering if enabled,
        expands context with adjacent chunks, and generates a response with
        source citations.

        Args:
            pregunta: The user's input question.
        """
        if len(pregunta.strip()) < self.rag.MIN_LONGITUD_PREGUNTA_RAG:
            ui.question_too_short()
            self.rag.guardar_debug_rag(
                pregunta,
                motivo_interrupcion="Pregunta demasiado corta.",
                metricas={
                    "longitud": len(pregunta.strip()),
                    "min_requerido": self.rag.MIN_LONGITUD_PREGUNTA_RAG,
                },
            )
            return

        timer = QueryTimer()
        fragmentos_finales: List[Dict[str, Any]] = []

        try:
            ui.pipeline_start()
            ui.pipeline_phase(ui._s("phase.search"), ui._s("phase.search.detail"))

            fragmentos_ranked, _, metricas = (
                self.rag.realizar_busqueda_hibrida(pregunta, self.collection)
            )
            timer.mark("búsqueda")

            if not fragmentos_ranked:
                ui.pipeline_stop()
                ui.no_results()
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=[],
                    motivo_interrupcion="No se encontraron fragmentos.",
                    metricas=metricas,
                )
                return

            if self.rag.USAR_RERANKER:
                ui.pipeline_phase(
                    ui._s("phase.rerank"),
                    ui._s("phase.rerank.detail", n=len(fragmentos_ranked)),
                )
                timer.mark("rerank")
            fragmentos_finales, metricas_contexto = (
                self.rag.preparar_fragmentos_para_generacion(
                    fragmentos_ranked,
                    self.collection,
                )
            )
            metricas = {**metricas, "fase_contexto": metricas_contexto}
            if metricas_contexto.get("fragmentos_expandidos", 0) > 0:
                ui.pipeline_phase(
                    ui._s("phase.expand"),
                    ui._s("phase.expand.detail", n=self.rag.N_TOP_PARA_EXPANSION),
                )
                timer.mark("expansión")

            if not fragmentos_finales:
                ui.pipeline_stop()
                ui.no_results()
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=fragmentos_ranked,
                    motivo_interrupcion="Ningún candidato superó el filtro de relevancia.",
                    metricas={**metricas, "umbral_reranker": self.rag.UMBRAL_SCORE_RERANKER},
                )
                return

            if self.rag.USAR_RECOMP_SYNTHESIS:
                ui.pipeline_phase(
                    ui._s("phase.synthesis"),
                    ui._s("phase.synthesis.detail", model=self.rag.MODELO_RECOMP),
                )

            ui.pipeline_phase(
                ui._s("phase.generation"),
                ui._s("phase.generation.detail", model=self.rag.MODELO_RAG),
            )
            ui.pipeline_stop()

            ui.response_header("rag", self.rag.MODELO_RAG)
            if ui.can_stream_responses():
                ui.begin_stream()
            respuesta = self.rag.generar_respuesta(
                pregunta,
                fragmentos_finales,
                metricas=metricas,
                on_token=ui.stream_token if ui.can_stream_responses() else None,
            )
            timer.mark("generación")
        except Exception as e:
            ui.pipeline_stop()
            ui.exception("Error RAG", e)
            try:
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=[],
                    motivo_interrupcion=f"Excepción en CLI RAG: {e}",
                    metricas={"error": e.__class__.__name__},
                )
            except Exception:
                pass
            return
        finally:
            ui.pipeline_stop()

        if not ui.can_stream_responses():
            ui.render_response(respuesta)
        ui.response_footer_rag(fragmentos_finales, timer)
        self.session.tick_rag(timer.total, self.rag.MODELO_RAG)

    # COMMAND HANDLERS

    def _cmd_rag(self) -> bool:
        self.mode = "rag"
        ui.mode_change("rag", self.rag.MODELO_RAG)
        return False

    def _cmd_chat(self) -> bool:
        self.mode = "chat"
        ui.mode_change("chat", self.rag.MODELO_CHAT)
        return False

    def _cmd_clear(self) -> bool:
        self.rag.limpiar_historial(self.historial_chat)
        ui.history_cleared()
        return False

    def _cmd_stats(self) -> bool:
        docs = self._get_document_summaries()
        info = self._runtime_info(len(self._list_pdf_files()), self.collection.count())
        ui.stats_dashboard(self.collection.count(), docs, info, self.session)
        return False

    def _cmd_help(self) -> bool:
        ui.welcome()
        return False

    def _cmd_docs(self) -> bool:
        ui.docs_table(self._get_document_summaries())
        return False

    def _cmd_topics(self) -> bool:
        self._show_topics()
        return False

    def _cmd_reindex(self) -> bool:
        """Delete the current database and re-index all documents.

        Returns:
            True to signal the main loop to exit (restart required).
        """
        ui.reindex_start()
        try:
            self.collection.clear()
            ui.success(ui._s("reindex.db_deleted"))
            collection_new = self.collection
            total = self.rag.indexar_documentos(
                self.rag.CARPETA_DOCS, collection_new
            )
            ui.reindex_complete(total)
            self.rag.guardar_historial(self.historial_chat)
            return True
        except Exception as e:
            ui.error(f"error durante reindexación: {e}")
            return False

    def _cmd_exit(self) -> bool:
        """Save history and exit the application."""
        self.rag.guardar_historial(self.historial_chat)
        ui.farewell(self.session)
        if os.name == "nt":
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception:
                pass
        return True

    # HELPERS

    def _check_index_fingerprint(self) -> None:
        """Warn if the reused store no longer matches the active configuration.

        Detection only: reindexing stays an explicit user action (/reindex)
        because a settings change (e.g. chunk size) must never silently
        trigger a MinerU + jina-clip pass over the corpus, which can take an
        hour. A store with no recorded fingerprint (every index built before
        this feature existed) is an unknown recipe, not a mismatch, and stays
        silent -- see index_fingerprint_mismatch's docstring.
        """
        if self.rag.index_fingerprint_mismatch(self.collection):
            ui.warning(ui._s("index.fingerprint_mismatch"))

    def _show_init_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> None:
        ui.init_panel(self._runtime_info(total_documentos, total_fragmentos))

    def _runtime_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> Dict[str, Any]:
        """Build display-only runtime metadata for CLI panels."""
        rag = self.rag

        reranker_info = 'on' if rag.USAR_RERANKER else 'off'
        reranker_model = None
        reranker_device = None
        if rag.USAR_RERANKER:
            reranker_device_val, _forced = rag.resolve_reranker_device()
            reranker_model = 'BAAI/bge-reranker-v2-m3'
            reranker_device = (reranker_device_val.upper()
                               + (' (FP16)' if reranker_device_val == 'cuda' else ''))

        return {
            'mode': self.mode,
            'modelo_rag': rag.MODELO_RAG,
            'modelo_chat': rag.MODELO_CHAT,
            'modelo_embedding': 'jinaai/jina-clip-v2',
            'modelo_contextual': rag.MODELO_CONTEXTUAL,
            'modelo_recomp': rag.MODELO_RECOMP,
            'extractor': 'extractor.mineru',
            'busqueda': ('pipeline.search.hybrid' if rag.USAR_BUSQUEDA_HIBRIDA
                         else 'pipeline.search.semantic'),
            'hybrid': rag.USAR_BUSQUEDA_HIBRIDA,
            'contextual': rag.USAR_CONTEXTUAL_RETRIEVAL,
            'recomp': rag.USAR_RECOMP_SYNTHESIS,
            'images': rag.USAR_EMBEDDINGS_IMAGEN,
            'expand': rag.EXPANDIR_CONTEXTO,
            'reranker': reranker_info,
            'reranker_model': reranker_model,
            'reranker_device': reranker_device,
            'chunk_size': rag.CHUNK_SIZE,
            'chunk_overlap': rag.CHUNK_OVERLAP,
            'total_documentos': total_documentos,
            'total_fragmentos': total_fragmentos,
            'docs_folder': rag.CARPETA_DOCS,
            'path_db': rag.PATH_DB,
            'collection_name': rag.COLLECTION_NAME,
        }

    def _list_pdf_files(self) -> List[str]:
        """Return PDF filenames in the configured docs folder."""
        try:
            return [
                f for f in os.listdir(self.rag.CARPETA_DOCS)
                if f.lower().endswith('.pdf')
            ]
        except FileNotFoundError:
            return []

    def _get_document_summaries(self) -> List[Dict[str, Any]]:
        """Aggregate document metadata from the FAISS sidecar."""
        summaries: Dict[str, Dict[str, Any]] = {}
        try:
            fragments = self.collection.get_page(None, 0)
        except Exception as e:
            ui.error(f"error leyendo metadatos de documentos: {e}")
            return []

        for fragment in fragments:
            meta = fragment.metadata
            source = meta.source
            if not source:
                continue
            entry = summaries.setdefault(source, {
                'name': source,
                'pages_set': set(),
                'fragments': 0,
                'formats_set': set(),
            })
            entry['fragments'] += 1
            if isinstance(meta.page, int):
                entry['pages_set'].add(meta.page)
            if meta.format:
                entry['formats_set'].add(meta.format)

        total_fragments = sum(e['fragments'] for e in summaries.values())
        result = []
        for source in sorted(summaries):
            entry = summaries[source]
            pct = (entry['fragments'] / total_fragments * 100) if total_fragments > 0 else 0
            result.append({
                'name': entry['name'],
                'pages': len(entry['pages_set']) if entry['pages_set'] else '-',
                'fragments': entry['fragments'],
                'pct_corpus': f"{pct:.1f}%",
                'formats': ', '.join(sorted(entry['formats_set'])) or '-',
            })
        return result

    def _show_topics(self) -> None:
        """Gather topic information from indexed documents and display it."""
        docs = self._get_document_summaries()

        if not docs:
            ui.topics_display([])
            return

        docs_data = []
        for doc_summary in docs:
            doc_name = doc_summary['name']
            n_frags = doc_summary.get('fragments', 0)
            doc_info = {
                'name': doc_name,
                'pages': doc_summary.get('pages'),
                'fragments': n_frags,
            }
            try:
                documents = [
                    fragment.doc
                    for fragment in self.collection.get_page(None, 0)
                    if fragment.metadata.source == doc_name
                ]
                analizados = len(documents)
                doc_info['analizados'] = analizados
                if documents:
                    texto = " ".join(documents)
                    palabras = texto.split()
                    significativas = [
                        p.strip('.,;:()[]{}"\'-').lower()
                        for p in palabras
                        if (len(p) > 3
                            and p.strip('.,;:()[]{}"\'-').lower()
                            not in self.rag.STOPWORDS)
                    ]
                    frecuencias = Counter(significativas)
                    top = [w for w, _ in frecuencias.most_common(10)]
                    doc_info['terms'] = ', '.join(top) if top else None
                else:
                    doc_info['analizados'] = 0
            except Exception as e:
                doc_info['terms'] = f"error: {e}"
                doc_info['analizados'] = 0

            docs_data.append(doc_info)

        ui.topics_display(docs_data)

    # OLLAMA HEALTH CHECK

    def _ollama_health(self, timeout: float = 2.0) -> Tuple[bool, str]:
        """Ping Ollama's ``/api/tags`` endpoint and summarize the result.

        The check is intentionally cheap and short-lived: a failed server is
        reported once at startup without blocking the rest of initialization,
        so the user understands why later calls will fail.
        """
        # Read off the engine rather than the environment: the engine already
        # resolved OLLAMA_BASE_URL/OLLAMA_HOST into the endpoint its adapters
        # generate against, and re-reading the raw variable here is how this
        # check used to report a server the pipeline never talked to.
        base = getattr(self.rag, "OLLAMA_BASE_URL", None) or "http://localhost:11434"
        try:
            import requests
            r = requests.get(f"{base}/api/tags", timeout=timeout)
            r.raise_for_status()
            models = r.json().get("models", []) or []
            return True, s("ollama.active", base=base, n=len(models))
        except Exception as e:
            return False, s("ollama.unavailable", base=base, error=e.__class__.__name__)
