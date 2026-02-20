"""
MonkeyGrab CLI — App
=====================

Bucle principal de la interfaz interactiva. Orquesta el prompt,
el dispatch de comandos y la integración con el motor RAG.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from collections import Counter

import chromadb

from rag.cli.theme import Theme, get_logo, MESSAGES
from rag.cli import renderer


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

        # Dispatch table de comandos
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
        # Logo
        print(get_logo(self.rag.MODELO_DESC))

        # Base de datos
        client = chromadb.PersistentClient(path=self.rag.PATH_DB)
        self.collection = client.get_or_create_collection(
            name=self.rag.COLLECTION_NAME
        )

        # PDFs disponibles
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

        # Info de inicialización
        self._show_init_info(pdfs_count, db_count)

        if not archivos_pdf:
            print(f"\n  {Theme.YELLOW}{Theme.ICON_WARN}{Theme.RESET} "
                  f"{Theme.TEXT_DIM}No existe la carpeta de PDFs o está vacía: "
                  f"{self.rag.CARPETA_DOCS}{Theme.RESET}")

        # Indexar si está vacía
        if self.collection.count() == 0:
            total_chunks = self.rag.indexar_documentos(
                self.rag.CARPETA_DOCS, self.collection
            )
            if total_chunks > 0:
                renderer.render_banner("INDEXACIÓN COMPLETADA", "doble", color=Theme.GREEN_DIM)
                print(f"\n  {Theme.GREEN}{Theme.ICON_OK}{Theme.RESET} "
                      f"{Theme.TEXT}fragmentos indexados: {total_chunks}{Theme.RESET}")
                print(f"  {Theme.TEXT_DIM}documentos en colección: "
                      f"{self.collection.count()}{Theme.RESET}")
            else:
                print(f"\n  {Theme.YELLOW}{Theme.ICON_WARN}{Theme.RESET} "
                      f"{Theme.TEXT_DIM}No se indexaron documentos.{Theme.RESET}")
                return

        # Bienvenida
        renderer.render_welcome()

        # Historial
        self.historial_chat = self.rag.cargar_historial()
        if self.historial_chat:
            renderer.render_history_loaded(len(self.historial_chat))

        print(f"\n  {Theme.TEXT_DIM}modo activo: {Theme.PURPLE}chat{Theme.RESET}  "
              f"{Theme.TEXT_DIM}(usa {Theme.CYAN}/rag{Theme.TEXT_DIM} para consultar documentos){Theme.RESET}")

        # Bucle principal
        self._loop()

    # ─────────────────────────────────────────────────────────────
    # BUCLE PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Bucle de lectura → dispatch → respuesta."""
        while True:
            # Mostrar prompt
            model = (self.rag.MODELO_AUXILIAR if self.mode == "chat"
                     else self.rag.MODELO_CHAT)
            prompt_str = renderer.build_prompt(self.mode, model)

            try:
                pregunta = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{MESSAGES['farewell']}")
                self.rag.guardar_historial(self.historial_chat)
                break

            if not pregunta:
                continue

            # Verificar si es un comando
            cmd_lower = pregunta.lower()
            if cmd_lower in self._commands:
                should_exit = self._commands[cmd_lower]()
                if should_exit:
                    break
                continue

            # Comando desconocido
            if pregunta.startswith('/'):
                renderer.render_unknown_command(pregunta)
                continue

            # Procesar pregunta según modo
            if self.mode == "rag":
                self._process_rag(pregunta)
            else:
                self._process_chat(pregunta)

    # ─────────────────────────────────────────────────────────────
    # PROCESAMIENTO DE PREGUNTAS
    # ─────────────────────────────────────────────────────────────

    def _process_chat(self, pregunta: str) -> None:
        """Procesa una pregunta en modo chat."""
        renderer.render_response_header("chat", self.rag.MODELO_AUXILIAR)

        respuesta = self._chat_stream(pregunta)

        renderer.render_response_footer()

        self.historial_chat.append({"role": "user", "content": pregunta})
        self.historial_chat.append({"role": "assistant", "content": respuesta})
        self.rag.guardar_historial(self.historial_chat)

    def _chat_stream(self, pregunta: str) -> str:
        """Ejecuta streaming del modo chat usando el modelo auxiliar."""
        import ollama

        messages = [{"role": "system", "content": self.rag.SYSTEM_PROMPT_CHAT}]
        mensajes_recientes = self.historial_chat[-(self.rag.MAX_HISTORIAL_MENSAJES):]
        messages.extend(mensajes_recientes)
        messages.append({"role": "user", "content": pregunta})

        stream = ollama.chat(
            model=self.rag.MODELO_AUXILIAR,
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
                renderer.stream_token(content)
                respuesta += content
        print()

        return respuesta

    def _process_rag(self, pregunta: str) -> None:
        """Procesa una pregunta en modo RAG."""
        renderer.render_response_header("rag", self.rag.MODELO_CHAT)

        if len(pregunta.strip()) < self.rag.MIN_LONGITUD_PREGUNTA_RAG:
            print(MESSAGES['question_too_short'])
            return

        # Búsqueda híbrida
        fragmentos_ranked, mejor_score, metricas = \
            self.rag.realizar_busqueda_hibrida(pregunta, self.collection)

        if not fragmentos_ranked:
            print(f"\n{MESSAGES['no_results']}")
            return

        if mejor_score < self.rag.UMBRAL_RELEVANCIA:
            print(f"\n{MESSAGES['out_of_scope']}")
            print(f"    {Theme.TEXT_DIM}score: {mejor_score:.4f}  "
                  f"umbral: {self.rag.UMBRAL_RELEVANCIA}{Theme.RESET}")
            return

        # Filtro reranker
        if self.rag.USAR_RERANKER:
            fragmentos_filtrados = [
                f for f in fragmentos_ranked
                if f.get('score_reranker', f.get('score_final', 0))
                   >= self.rag.UMBRAL_SCORE_RERANKER
            ]
            if not fragmentos_filtrados:
                print(f"\n{MESSAGES['no_results']}")
                return
            fragmentos_ranked = fragmentos_filtrados

        fragmentos_finales = fragmentos_ranked[:self.rag.TOP_K_FINAL]
        ids_usados = {f['id'] for f in fragmentos_finales}

        # Expansión de contexto
        if (self.rag.EXPANDIR_CONTEXTO and fragmentos_finales
                and 'chunk' in fragmentos_finales[0]['metadata']):
            self._expand_context(fragmentos_finales, ids_usados)

        # Truncar si excede límite
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

        renderer.render_success(
            f"contexto listo: {len(fragmentos_finales)} fragmento(s)"
        )

        # Generar respuesta
        self.rag.generar_respuesta(pregunta, fragmentos_finales)

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
        renderer.render_mode_change("rag", self.rag.MODELO_CHAT)
        return False

    def _cmd_chat(self) -> bool:
        self.mode = "chat"
        renderer.render_mode_change("chat", self.rag.MODELO_AUXILIAR)
        return False

    def _cmd_clear(self) -> bool:
        self.rag.limpiar_historial(self.historial_chat)
        renderer.render_history_cleared()
        return False

    def _cmd_stats(self) -> bool:
        docs = self.rag.obtener_documentos_indexados(self.collection)
        renderer.render_stats(self.collection.count(), docs)
        return False

    def _cmd_help(self) -> bool:
        renderer.render_banner("AYUDA", "simple", color=Theme.BRAND_DIM)
        renderer.render_welcome()
        return False

    def _cmd_docs(self) -> bool:
        docs = self.rag.obtener_documentos_indexados(self.collection)
        renderer.render_docs(docs)
        return False

    def _cmd_topics(self) -> bool:
        self._show_topics()
        return False

    def _cmd_reindex(self) -> bool:
        renderer.render_reindex_start()
        try:
            if os.path.exists(self.rag.PATH_DB):
                shutil.rmtree(self.rag.PATH_DB)
                renderer.render_success("base de datos anterior eliminada")

            client_new = chromadb.PersistentClient(path=self.rag.PATH_DB)
            collection_new = client_new.get_or_create_collection(
                name=self.rag.COLLECTION_NAME
            )
            total = self.rag.indexar_documentos(
                self.rag.CARPETA_DOCS, collection_new
            )
            renderer.render_reindex_complete(total)
            self.rag.guardar_historial(self.historial_chat)
            return True  # Exit after reindex
        except Exception as e:
            renderer.render_error(f"error durante reindexación: {e}")
            return False

    def _cmd_exit(self) -> bool:
        self.rag.guardar_historial(self.historial_chat)
        print(f"\n{MESSAGES['farewell']}")
        return True

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _get_config(self) -> Dict[str, Any]:
        """Construye dict de configuración para render_welcome."""
        return {
            'chunk_size': self.rag.CHUNK_SIZE,
            'chunk_overlap': self.rag.CHUNK_OVERLAP,
            'extractor': ('pymupdf4llm'
                          if self.rag.PYMUPDF_AVAILABLE else 'pypdf'),
            'reranker': self.rag.USAR_RERANKER,
            'hybrid': self.rag.USAR_BUSQUEDA_HIBRIDA,
            'llm_decomp': self.rag.USAR_LLM_QUERY_DECOMPOSITION,
        }

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

        renderer.render_init_info({
            'cwd': os.getcwd(),
            'pdfs_path': rag.CARPETA_DOCS,
            'db_path': rag.PATH_DB,
            'historial_path': rag.HISTORIAL_PATH,
            'modelo_chat': rag.MODELO_CHAT,
            'modelo_auxiliar': rag.MODELO_AUXILIAR,
            'modelo_embedding': rag.MODELO_EMBEDDING,
            'extractor': ('pymupdf4llm' if rag.PYMUPDF_AVAILABLE
                          else 'pypdf (fallback)'),
            'busqueda': ('híbrida (semántica + keywords)'
                         if rag.USAR_BUSQUEDA_HIBRIDA
                         else 'solo semántica'),
            'llm_decomp': (f"on  modelo: {rag.MODELO_AUXILIAR}"
                           if rag.USAR_LLM_QUERY_DECOMPOSITION
                           else 'off'),
            'reranker': reranker_info,
            'reranker_model': reranker_model,
            'reranker_device': reranker_device,
            'chunk_size': rag.CHUNK_SIZE,
            'chunk_overlap': rag.CHUNK_OVERLAP,
            'embed_max': rag.MAX_CHARS_EMBED,
            'embed_prefix_desc': rag._EMBED_PREFIX_DESC,
            'db_version': rag._DB_VERSION,
            'total_documentos': total_documentos,
            'total_fragmentos': total_fragmentos,
        })

    def _show_topics(self) -> None:
        """Recopila información de temas y los muestra."""
        docs = self.rag.obtener_documentos_indexados(self.collection)

        if not docs:
            renderer.render_topics([])
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
                    doc_info['sample'] = all_data['documents'][0][:300]
            except Exception:
                pass

            docs_data.append(doc_info)

        renderer.render_topics(docs_data)
