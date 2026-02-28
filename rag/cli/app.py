"""
MonkeyGrab CLI
==============================

Bucle principal de la interfaz interactiva. Orquesta el prompt,
el dispatch de comandos y la integración con el motor RAG.

Usa la clase Display (rich) para toda la salida visual.
"""

import os
import shutil
import time
from typing import List, Dict, Any, Optional
from collections import Counter

import chromadb

from rag.cli.display import ui


class MonkeyGrabCLI:
    """
    Aplicación CLI principal de MonkeyGrab.

    Encapsula el bucle de interacción, gestión de estado (modo, historial)
    y dispatch de comandos slash. Delega la lógica RAG a chat_pdfs.
    """

    def __init__(self, rag_engine):
        """
        Args:
            rag_engine: Módulo o namespace con las funciones RAG
                        (realizar_busqueda_hibrida, generar_respuesta, etc.)
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

    # ─────────────────────────────────────────────────────────────
    # ARRANQUE
    # ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Punto de entrada principal. Inicializa y arranca el bucle."""
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
        db_count = self.collection.count()

        self._show_init_info(pdfs_count, db_count)

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

        ui.welcome()

        self.historial_chat = self.rag.cargar_historial()
        if self.historial_chat:
            ui.history_loaded(len(self.historial_chat))

        self._loop()

    # ─────────────────────────────────────────────────────────────
    # BUCLE PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Bucle de lectura → dispatch → respuesta."""
        while True:
            model = (self.rag.MODELO_CHAT if self.mode == "chat"
                     else self.rag.MODELO_RAG)
            prompt_str = ui.prompt(self.mode, model)

            try:
                pregunta = input(prompt_str).strip()
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

    # ─────────────────────────────────────────────────────────────
    # PROCESAMIENTO DE PREGUNTAS
    # ─────────────────────────────────────────────────────────────

    def _process_chat(self, pregunta: str) -> None:
        """Procesa una pregunta en modo chat."""
        ui.response_header("chat", self.rag.MODELO_CHAT)

        respuesta = self._chat_stream(pregunta)

        ui.response_footer()

        self.historial_chat.append({"role": "user", "content": pregunta})
        self.historial_chat.append({"role": "assistant", "content": respuesta})
        self.rag.guardar_historial(self.historial_chat)

    def _chat_stream(self, pregunta: str) -> str:
        """Ejecuta streaming del modo chat usando el modelo chat."""
        import ollama

        messages = [{"role": "system", "content": self.rag.SYSTEM_PROMPT_CHAT}]
        mensajes_recientes = self.historial_chat[-(self.rag.MAX_HISTORIAL_MENSAJES):]
        messages.extend(mensajes_recientes)
        messages.append({"role": "user", "content": pregunta})

        stream = ollama.chat(
            model=self.rag.MODELO_CHAT,
            messages=messages,
            stream=True,
            options={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192}
        )

        respuesta = ""
        print()
        for chunk in stream:
            content = (chunk.get("message", {}).get("content", "")
                       or chunk.get("content", ""))
            if content:
                ui.stream_token(content)
                respuesta += content
        print()

        return respuesta

    def _process_rag(self, pregunta: str) -> None:
        """Procesa una pregunta en modo RAG."""
        if len(pregunta.strip()) < self.rag.MIN_LONGITUD_PREGUNTA_RAG:
            ui.question_too_short()
            self.rag.guardar_debug_rag(
                pregunta,
                motivo_interrupcion="Pregunta demasiado corta.",
                metricas={"longitud": len(pregunta.strip()), "min_requerido": self.rag.MIN_LONGITUD_PREGUNTA_RAG}
            )
            return

        status = ui.pipeline_start("Buscando conceptos en los documentos...")

        fragmentos_ranked, mejor_score, metricas = \
            self.rag.realizar_busqueda_hibrida(pregunta, self.collection)

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

        if mejor_score < self.rag.UMBRAL_RELEVANCIA:
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
        self.rag.generar_respuesta(pregunta, fragmentos_finales)

        ui.sources_panel(fragmentos_finales)
        ui.response_footer()

    def _expand_context(self, fragmentos: list, ids_usados: set) -> None:
        """Expande contexto con chunks adyacentes."""
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

    # ─────────────────────────────────────────────────────────────
    # COMANDOS
    # ─────────────────────────────────────────────────────────────

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
        ui.stats_table(self.collection.count(), docs)
        return False

    def _cmd_help(self) -> bool:
        ui.welcome()
        return False

    def _cmd_docs(self) -> bool:
        docs = self.rag.obtener_documentos_indexados(self.collection)
        ui.docs_table(docs)
        return False

    def _cmd_topics(self) -> bool:
        self._show_topics()
        return False

    def _cmd_reindex(self) -> bool:
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
        self.rag.guardar_historial(self.historial_chat)
        ui.farewell()
        return True

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _show_init_info(self, total_documentos: int = 0, total_fragmentos: int = 0) -> None:
        """Recopila y muestra info de inicialización."""
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

        ui.init_panel({
            'modelo_rag': rag.MODELO_RAG,
            'modelo_chat': rag.MODELO_CHAT,
            'modelo_embedding': rag.MODELO_EMBEDDING,
            'extractor': ('pymupdf4llm' if rag.PYMUPDF_AVAILABLE
                          else 'pypdf (fallback)'),
            'busqueda': ('híbrida (semántica + keywords)'
                         if rag.USAR_BUSQUEDA_HIBRIDA
                         else 'solo semántica'),
            'reranker': reranker_info,
            'reranker_model': reranker_model,
            'reranker_device': reranker_device,
            'chunk_size': rag.CHUNK_SIZE,
            'chunk_overlap': rag.CHUNK_OVERLAP,
            'total_documentos': total_documentos,
            'total_fragmentos': total_fragmentos,
        })

    def _show_topics(self) -> None:
        """Recopila información de temas y los muestra."""
        docs = self.rag.obtener_documentos_indexados(self.collection)

        if not docs:
            ui.topics_display([])
            return

        docs_data = []
        for doc_name in docs:
            doc_info = {'name': doc_name}
            try:
                all_data = self.collection.get(
                    where={"source": doc_name},
                    include=['documents', 'metadatas'],
                    limit=100
                )
                if all_data['documents']:
                    pages = {meta['page'] for meta in all_data['metadatas']}
                    doc_info['pages'] = len(pages)
                    doc_info['fragments'] = len(all_data['documents'])

                    texto = " ".join(all_data['documents'][:20])
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
            except Exception:
                pass

            docs_data.append(doc_info)

        ui.topics_display(docs_data)
