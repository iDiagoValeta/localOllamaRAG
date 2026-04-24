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
    - rag.cli.display (ui singleton)
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
#  +-- 2. INITIALIZATION       __init__, ChromaDB + engine wiring
#  +-- 3. STARTUP              print_startup_info, banner
#  +-- 4. MAIN LOOP            run() — prompt dispatch
#  +-- 5. CHAT / RAG PROCESSING  handle_chat, handle_rag
#  +-- 6. COMMAND HANDLERS     /docs, /stats, /reindex, /temas, /help, /salir
#  +-- 7. HELPERS              context truncation, term extraction
#
# ─────────────────────────────────────────────

import os
import shutil
from typing import List, Dict, Any
from collections import Counter

import chromadb

from rag.cli.display import ui


# ─────────────────────────────────────────────
# SECTION 2: MAIN CLI CLASS
# ─────────────────────────────────────────────

class MonkeyGrabCLI:
    """
    Main CLI application for MonkeyGrab.

    Encapsulates the interaction loop, state management (mode, history),
    and slash command dispatch. Delegates RAG logic to the provided
    rag_engine module.
    """

    # ─────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────

    def __init__(self, rag_engine):
        """
        Initialize the CLI application.

        Args:
            rag_engine: Module or namespace providing RAG functions
                        (realizar_busqueda_hibrida, generar_respuesta, etc.)
                        and configuration constants.
        """
        self.rag = rag_engine
        self.mode = "chat"
        self.collection = None
        self.historial_chat: List[Dict[str, str]] = []

        self._commands = {
            "/rag":     self._cmd_rag,
            "/chat":    self._cmd_chat,
            "/limpiar": self._cmd_clear,
            "/clear":   self._cmd_clear,
            "/stats":   self._cmd_stats,
            "/ayuda":   self._cmd_help,
            "/help":    self._cmd_help,
            "/docs":    self._cmd_docs,
            "/temas":   self._cmd_topics,
            "/reindex": self._cmd_reindex,
            "/salir":   self._cmd_exit,
            "/exit":    self._cmd_exit,
        }

    # ─────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────

    def run(self) -> None:
        """Entry point. Initialize the system and start the main loop."""
        ui.logo()

        client = chromadb.PersistentClient(path=self.rag.PATH_DB)
        self.collection = client.get_or_create_collection(
            name=self.rag.COLLECTION_NAME
        )

        archivos_pdf = []
        try:
            archivos_pdf = [
                f for f in os.listdir(self.rag.CARPETA_DOCS)
                if f.lower().endswith('.pdf')
            ]
        except FileNotFoundError:
            pass

        pdfs_count = len(archivos_pdf)

        if not archivos_pdf:
            ui.no_pdfs(self.rag.CARPETA_DOCS)

        if self.collection.count() == 0:
            total_chunks = self.rag.indexar_documentos(
                self.rag.CARPETA_DOCS, self.collection
            )
            if total_chunks == 0:
                ui.warning("No se indexaron documentos.")
                return
            else:
                ui.success(f"{total_chunks} fragmentos indexados")

        self._show_init_info(pdfs_count, self.collection.count())
        ui.info("usa /ayuda para ver todos los comandos")

        self.historial_chat = self.rag.cargar_historial()
        if self.historial_chat:
            ui.history_loaded(len(self.historial_chat))

        self._loop()

    # ─────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────

    def _loop(self) -> None:
        """Read-dispatch-respond loop. Runs until the user exits."""
        while True:
            model = (self.rag.MODELO_CHAT if self.mode == "chat"
                     else self.rag.MODELO_RAG)

            try:
                pregunta = ui.read_input(self.mode, model).strip()
            except (EOFError, KeyboardInterrupt):
                ui.farewell()
                self.rag.guardar_historial(self.historial_chat)
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
                ui.unknown_command(pregunta)
                continue

            if self.mode == "rag":
                self._process_rag(pregunta)
            else:
                self._process_chat(pregunta)

    # ─────────────────────────────────────────────
    # CHAT / RAG PROCESSING
    # ─────────────────────────────────────────────

    def _process_chat(self, pregunta: str) -> None:
        """Process a question in chat mode.

        Args:
            pregunta: The user's input question.
        """
        ui.response_header("chat", self.rag.MODELO_CHAT)

        try:
            respuesta = self._chat_stream(pregunta)
        except Exception as e:
            ui.exception("Error de chat", e)
            return

        if ui.safe_tty:
            ui.render_response(respuesta)

        ui.response_footer()

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
        if not ui.safe_tty:
            ui.console.print()
        for chunk in stream:
            content = (chunk.get("message", {}).get("content", "")
                       or chunk.get("content", ""))
            if content:
                respuesta += content
                if not ui.safe_tty:
                    ui.stream_token(content)
        if not ui.safe_tty:
            ui.console.print()

        return respuesta

    def _process_rag(self, pregunta: str) -> None:
        """Process a question in RAG mode.

        Validates the question length, runs hybrid search, applies
        reranker filtering if enabled, expands context with adjacent
        chunks, and generates a response with source citations.

        Args:
            pregunta: The user's input question.
        """
        if len(pregunta.strip()) < self.rag.MIN_LONGITUD_PREGUNTA_RAG:
            ui.question_too_short()
            self.rag.guardar_debug_rag(
                pregunta,
                motivo_interrupcion="Pregunta demasiado corta.",
                metricas={"longitud": len(pregunta.strip()), "min_requerido": self.rag.MIN_LONGITUD_PREGUNTA_RAG}
            )
            return

        try:
            ui.pipeline_start("Buscando conceptos en los documentos...")
            fragmentos_ranked, mejor_score, metricas = (
                self.rag.realizar_busqueda_hibrida(pregunta, self.collection)
            )

            if not fragmentos_ranked:
                ui.pipeline_stop()
                ui.no_results()
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=[],
                    motivo_interrupcion="No se encontraron fragmentos.",
                    metricas=metricas
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
                    metricas={**metricas, "mejor_score": mejor_score}
                )
                return

            ui.pipeline_update("Re-ordenando resultados...")

            if self.rag.USAR_RERANKER:
                fragmentos_filtrados = [
                    f for f in fragmentos_ranked
                    if f.get('score_reranker', f.get('score_final', 0))
                       >= self.rag.UMBRAL_SCORE_RERANKER
                ]
                if not fragmentos_filtrados:
                    ui.pipeline_stop()
                    ui.no_results()
                    self.rag.guardar_debug_rag(
                        pregunta,
                        fragmentos=fragmentos_ranked,
                        motivo_interrupcion="Reranker filtró todos los candidatos.",
                        metricas={**metricas, "umbral_reranker": self.rag.UMBRAL_SCORE_RERANKER}
                    )
                    return
                fragmentos_ranked = fragmentos_filtrados

            fragmentos_finales = fragmentos_ranked[:self.rag.TOP_K_FINAL]
            ids_usados = {f['id'] for f in fragmentos_finales}

            if (self.rag.EXPANDIR_CONTEXTO and fragmentos_finales
                    and 'chunk' in fragmentos_finales[0]['metadata']):
                self._expand_context(fragmentos_finales, ids_usados)

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

            ui.pipeline_update("Generando respuesta...")
            ui.pipeline_stop()

            ui.response_header("rag", self.rag.MODELO_RAG)
            respuesta = self.rag.generar_respuesta(
                pregunta,
                fragmentos_finales,
                metricas=metricas,
                on_token=None if ui.safe_tty else ui.stream_token,
            )
        except Exception as e:
            ui.pipeline_stop()
            ui.exception("Error RAG", e)
            try:
                self.rag.guardar_debug_rag(
                    pregunta,
                    fragmentos=[],
                    motivo_interrupcion=f"Excepción en CLI RAG: {e}",
                    metricas={"error": e.__class__.__name__}
                )
            except Exception:
                pass
            return
        finally:
            ui.pipeline_stop()

        if ui.safe_tty:
            ui.render_response(respuesta)
        ui.sources_panel(fragmentos_finales)
        ui.response_footer()

    def _expand_context(self, fragmentos: list, ids_usados: set) -> None:
        """Expand context by retrieving adjacent chunks from the collection.

        For each top fragment, fetches neighboring chunks and appends
        them to the fragment list if they haven't been included yet.

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
                                'id': v_id
                            })
                            ids_usados.add(v_id)
                except Exception:
                    pass

        if chunks_adicionales:
            fragmentos.extend(chunks_adicionales)

    # ─────────────────────────────────────────────
    # COMMAND HANDLERS
    # ─────────────────────────────────────────────

    def _cmd_rag(self) -> bool:
        """Switch to RAG mode."""
        self.mode = "rag"
        ui.mode_change("rag", self.rag.MODELO_RAG)
        return False

    def _cmd_chat(self) -> bool:
        """Switch to chat mode."""
        self.mode = "chat"
        ui.mode_change("chat", self.rag.MODELO_CHAT)
        return False

    def _cmd_clear(self) -> bool:
        """Clear the chat history."""
        self.rag.limpiar_historial(self.historial_chat)
        ui.history_cleared()
        return False

    def _cmd_stats(self) -> bool:
        """Display database statistics."""
        docs = self.rag.obtener_documentos_indexados(self.collection)
        ui.stats_table(self.collection.count(), docs, self._runtime_info(
            len(self._list_pdf_files()), self.collection.count()
        ))
        return False

    def _cmd_help(self) -> bool:
        """Display the help/welcome screen."""
        ui.welcome()
        return False

    def _cmd_docs(self) -> bool:
        """Display the list of indexed documents."""
        ui.docs_table(self._get_document_summaries())
        return False

    def _cmd_topics(self) -> bool:
        """Display the topics summary."""
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
                ui.success("base de datos anterior eliminada")

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
        """Save history and exit the application.

        Returns:
            True to signal the main loop to exit.
        """
        self.rag.guardar_historial(self.historial_chat)
        ui.farewell()
        return True

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _show_init_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> None:
        """Gather and display initialization information.

        Args:
            total_documentos: Number of PDF files detected.
            total_fragmentos: Number of fragments in the database.
        """
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
            'extractor': ('pymupdf4llm' if rag.PYMUPDF_AVAILABLE
                          else 'pypdf (fallback)'),
            'busqueda': ('híbrida (semántica + keywords)'
                         if rag.USAR_BUSQUEDA_HIBRIDA
                         else 'solo semántica'),
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
                    limit=100
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
