"""
MonkeyGrab CLI Application.

Main interactive loop for the MonkeyGrab command-line interface. Orchestrates
the user prompt, slash command dispatch, and integration with the RAG engine.
Uses the Display class (rich) for all visual output.

Usage:
    from rag.cli.app import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()

Dependencies:
    - chromadb
    - rag.cli.display (ui singleton, QueryTimer, SessionStats)
    - rag.cli.commands (single source of truth for slash-commands)
    - A RAG engine module providing search, indexing, and generation functions
"""


# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports
#
#  MonkeyGrabCLI CLASS
#  +-- 2. INITIALIZATION        __init__, ChromaDB + engine wiring
#  +-- 3. STARTUP               run() — logo, indexing, ollama health check
#  +-- 4. MAIN LOOP             _loop() — prompt dispatch
#  +-- 5. CHAT / RAG PROCESSING _process_chat, _chat_stream, _process_rag
#  +-- 6. COMMAND HANDLERS      /docs, /stats, /reindex, /temas, /help, /salir
#  +-- 7. HELPERS               runtime info, documents summary, topics
#  +-- 8. OLLAMA HEALTH CHECK   _ollama_health()
#
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────

import difflib
import os
import shutil
import signal
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from rag.cli.commands import ALIASES, COMMANDS
from rag.cli.display import QueryTimer, SessionStats, ui


# ─────────────────────────────────────────────
# SECTION 2: MAIN CLI CLASS
# ─────────────────────────────────────────────

class MonkeyGrabCLI:
    """Main CLI application for MonkeyGrab.

    Encapsulates the interaction loop, state management (mode, history),
    session statistics, and slash command dispatch. Delegates RAG logic to the
    provided rag_engine module.
    """

    # ─────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────

    def __init__(self, rag_engine):
        """Initialize the CLI application.

        Args:
            rag_engine: Module or namespace providing RAG functions
                        (realizar_busqueda_hibrida, generar_respuesta, etc.)
                        and configuration constants.
        """
        self.rag = rag_engine
        self.mode = "chat"
        self.collection: Optional[chromadb.Collection] = None
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

    # ─────────────────────────────────────────────
    # SECTION 3: STARTUP
    # ─────────────────────────────────────────────

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

        client = chromadb.PersistentClient(path=self.rag.PATH_DB)
        self.collection = client.get_or_create_collection(
            name=self.rag.COLLECTION_NAME
        )

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

        self._show_init_info(pdfs_count, self.collection.count())

        self.historial_chat = self.rag.cargar_historial()
        if self.historial_chat:
            ui.history_loaded(len(self.historial_chat))

        self._loop()

    # ─────────────────────────────────────────────
    # SECTION 4: MAIN LOOP
    # ─────────────────────────────────────────────

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
                    cmd_lower, self._commands.keys(), n=2, cutoff=0.6
                )
                ui.unknown_command(pregunta, suggestions=suggestions)
                continue

            if self.mode == "rag":
                self._process_rag(pregunta)
            else:
                self._process_chat(pregunta)

    # ─────────────────────────────────────────────
    # SECTION 5: CHAT / RAG PROCESSING
    # ─────────────────────────────────────────────

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
        import ollama

        messages = [{"role": "system", "content": self.rag.SYSTEM_PROMPT_CHAT}]
        mensajes_recientes = self.historial_chat[-(self.rag.MAX_HISTORIAL_MENSAJES):]
        messages.extend(mensajes_recientes)
        messages.append({"role": "user", "content": pregunta})

        stream = ollama.chat(
            model=self.rag.MODELO_CHAT,
            messages=messages,
            stream=True,
            think=False,
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
        best_score: Optional[float] = None

        try:
            ui.pipeline_start()
            ui.pipeline_phase(ui._s("phase.search"), ui._s("phase.search.detail"))

            fragmentos_ranked, mejor_score, metricas = (
                self.rag.realizar_busqueda_hibrida(pregunta, self.collection)
            )
            timer.mark("búsqueda")
            best_score = mejor_score

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

            # UMBRAL_RELEVANCIA applies to reranker-scale scores, not RRF-only fusion.
            if self.rag.USAR_RERANKER and mejor_score < self.rag.UMBRAL_RELEVANCIA:
                ui.pipeline_stop()
                ui.out_of_scope(mejor_score, self.rag.UMBRAL_RELEVANCIA)
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=fragmentos_ranked,
                    motivo_interrupcion=f"Score insuficiente: {mejor_score:.4f}",
                    metricas={**metricas, "mejor_score": mejor_score},
                )
                return

            if self.rag.USAR_RERANKER:
                ui.pipeline_phase(
                    ui._s("phase.rerank"),
                    ui._s("phase.rerank.detail", n=len(fragmentos_ranked)),
                )
                fragmentos_filtrados = [
                    f for f in fragmentos_ranked
                    if f.get('score_reranker', f.get('score_final', 0))
                       >= self.rag.UMBRAL_SCORE_RERANKER
                ]
                timer.mark("rerank")
                if not fragmentos_filtrados:
                    ui.pipeline_stop()
                    ui.no_results()
                    self.rag.guardar_debug_rag(
                        pregunta,
                        fragmentos=fragmentos_ranked,
                        motivo_interrupcion="Reranker filtró todos los candidatos.",
                        metricas={**metricas, "umbral_reranker": self.rag.UMBRAL_SCORE_RERANKER},
                    )
                    return
                fragmentos_ranked = fragmentos_filtrados

            fragmentos_finales = fragmentos_ranked[:self.rag.TOP_K_FINAL]
            ids_usados = {f['id'] for f in fragmentos_finales}

            if (self.rag.EXPANDIR_CONTEXTO and fragmentos_finales
                    and 'chunk' in fragmentos_finales[0]['metadata']):
                ui.pipeline_phase(
                    ui._s("phase.expand"),
                    ui._s("phase.expand.detail", n=self.rag.N_TOP_PARA_EXPANSION),
                )
                self._expand_context(fragmentos_finales, ids_usados)
                timer.mark("expansión")

            contexto_total = sum(len(f['doc']) for f in fragmentos_finales)
            if contexto_total > self.rag.MAX_CONTEXTO_CHARS:
                fragmentos_truncados = []
                chars_acum = 0
                for f in fragmentos_finales:
                    if chars_acum + len(f['doc']) > self.rag.MAX_CONTEXTO_CHARS:
                        break
                    fragmentos_truncados.append(f)
                    chars_acum += len(f['doc'])
                fragmentos_finales = fragmentos_truncados

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

    def _expand_context(self, fragmentos: list, ids_usados: set) -> None:
        """Expand context by retrieving adjacent chunks from the collection.

        For each top fragment, fetches neighboring chunks and appends them to
        the fragment list if they haven't been included yet.

        Args:
            fragmentos: List of selected fragments (modified in place).
            ids_usados: Set of already-used fragment IDs (modified in place).
        """
        chunks_adicionales = []

        for frag in fragmentos[:self.rag.N_TOP_PARA_EXPANSION]:
            ids_vecinos = self.rag.expandir_con_chunks_adyacentes(
                frag['id'], frag['metadata'], n_vecinos=1
            )
            if ids_vecinos:
                try:
                    vecinos = self.collection.get(
                        ids=ids_vecinos,
                        include=['documents', 'metadatas']
                    )
                    for v_doc, v_meta in zip(
                        vecinos['documents'], vecinos['metadatas']
                    ):
                        v_id = (f"{v_meta['source']}_pag{v_meta['page']}"
                                f"_chunk{v_meta.get('chunk', 0)}")
                        if v_id not in ids_usados:
                            chunks_adicionales.append({
                                'doc': v_doc,
                                'metadata': v_meta,
                                'distancia': float('inf'),
                                'score_final': 0.0,
                                'id': v_id,
                            })
                            ids_usados.add(v_id)
                except Exception:
                    pass

        if chunks_adicionales:
            fragmentos.extend(chunks_adicionales)

    # ─────────────────────────────────────────────
    # SECTION 6: COMMAND HANDLERS
    # ─────────────────────────────────────────────

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
        docs = self.rag.obtener_documentos_indexados(self.collection)
        info = self._runtime_info(len(self._list_pdf_files()), self.collection.count())
        ui.stats_dashboard(self.collection.count(), docs, info)
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
            if os.path.exists(self.rag.PATH_DB):
                shutil.rmtree(self.rag.PATH_DB)
                ui.success(ui._s("reindex.db_deleted"))

            client_new = chromadb.PersistentClient(path=self.rag.PATH_DB)
            collection_new = client_new.get_or_create_collection(
                name=self.rag.COLLECTION_NAME
            )
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

    # ─────────────────────────────────────────────
    # SECTION 7: HELPERS
    # ─────────────────────────────────────────────

    def _show_init_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> None:
        ui.init_panel(self._runtime_info(total_documentos, total_fragmentos))

    def _runtime_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> Dict[str, Any]:
        """Build display-only runtime metadata for CLI panels."""
        rag = self.rag

        reranker_info = 'on' if rag.USAR_RERANKER else 'off'
        reranker_model = None
        reranker_device = None
        if rag.USAR_RERANKER:
            reranker_device_val = rag._detectar_dispositivo_reranker()
            reranker_model = ('BAAI/bge-reranker-v2-m3'
                              if rag.RERANKER_MODEL_QUALITY == 'quality'
                              else 'ms-marco-MiniLM-L-6-v2')
            reranker_device = (reranker_device_val.upper()
                               + (' (FP16)' if reranker_device_val == 'cuda' else ''))

        return {
            'mode': self.mode,
            'modelo_rag': rag.MODELO_RAG,
            'modelo_chat': rag.MODELO_CHAT,
            'modelo_embedding': rag.MODELO_EMBEDDING,
            'modelo_contextual': rag.MODELO_CONTEXTUAL,
            'modelo_recomp': rag.MODELO_RECOMP,
            'modelo_ocr': rag.MODELO_OCR,
            'extractor': ('extractor.pymupdf' if rag.PYMUPDF_AVAILABLE
                          else 'extractor.pypdf'),
            'busqueda': ('pipeline.search.hybrid' if rag.USAR_BUSQUEDA_HIBRIDA
                         else 'pipeline.search.semantic'),
            'hybrid': rag.USAR_BUSQUEDA_HIBRIDA,
            'exhaustive': rag.USAR_BUSQUEDA_EXHAUSTIVA,
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
        """Aggregate document metadata from Chroma without truncating counts."""
        summaries: Dict[str, Dict[str, Any]] = {}
        try:
            all_metadata = self.collection.get(include=['metadatas'])
        except Exception as e:
            ui.error(f"error leyendo metadatos de documentos: {e}")
            return []

        for meta in all_metadata.get('metadatas', []) or []:
            if not meta:
                continue
            source = meta.get('source')
            if not source:
                continue
            entry = summaries.setdefault(source, {
                'name': source,
                'pages_set': set(),
                'fragments': 0,
                'formats_set': set(),
            })
            entry['fragments'] += 1
            if isinstance(meta.get('page'), int):
                entry['pages_set'].add(meta['page'])
            if meta.get('format'):
                entry['formats_set'].add(meta['format'])

        result = []
        for source in sorted(summaries):
            entry = summaries[source]
            result.append({
                'name': entry['name'],
                'pages': len(entry['pages_set']) if entry['pages_set'] else '-',
                'fragments': entry['fragments'],
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
            doc_info = {
                'name': doc_name,
                'pages': doc_summary.get('pages'),
                'fragments': doc_summary.get('fragments'),
            }
            try:
                all_data = self.collection.get(
                    where={"source": doc_name},
                    include=['documents', 'metadatas'],
                    limit=100,
                )
                documents = all_data.get('documents') or []
                if documents:
                    texto = " ".join(documents[:20])
                    palabras = texto.split()
                    significativas = [
                        p.strip('.,;:()[]{}"\'-').lower()
                        for p in palabras
                        if (len(p) > 5
                            and p.strip('.,;:()[]{}"\'-').lower()
                            not in self.rag.STOPWORDS)
                    ]
                    frecuencias = Counter(significativas)
                    top = [w for w, _ in frecuencias.most_common(10)]
                    doc_info['terms'] = ', '.join(top) if top else None
            except Exception as e:
                doc_info['terms'] = f"error: {e}"

            docs_data.append(doc_info)

        ui.topics_display(docs_data)

    # ─────────────────────────────────────────────
    # SECTION 8: OLLAMA HEALTH CHECK
    # ─────────────────────────────────────────────

    def _ollama_health(self, timeout: float = 2.0) -> Tuple[bool, str]:
        """Ping Ollama's ``/api/tags`` endpoint and summarize the result.

        The check is intentionally cheap and short-lived: a failed server is
        reported once at startup without blocking the rest of initialization,
        so the user understands why later calls will fail.
        """
        base = (
            os.getenv("OLLAMA_BASE_URL")
            or getattr(self.rag, "OLLAMA_BASE_URL", None)
            or "http://localhost:11434"
        )
        try:
            import requests
            r = requests.get(f"{base}/api/tags", timeout=timeout)
            r.raise_for_status()
            models = r.json().get("models", []) or []
            return True, f"Ollama activo en {base} · {len(models)} modelos locales"
        except Exception as e:
            return False, f"Ollama no responde en {base}: {e.__class__.__name__}"
