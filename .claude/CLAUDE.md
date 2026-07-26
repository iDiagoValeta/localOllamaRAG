# CLAUDE.md — MonkeyGrab (localOllamaRAG)

Local RAG over PDFs, with hybrid retrieval and a local Ollama LLM. Production
product only — the thesis research layer was defended and removed from HEAD
(history preserved at tag `v1.0.0-tfg`).

User-facing doc lives in `README.md`. Pipeline technical reference in
`rag/README.md`.

---

## 1. Behavior rules

1. **Never commit or push without asking first.** No exceptions.
2. **The current design lives in `docs/design/2026-07-26-monkeygrab-v2.md`.** It takes precedence over any other documentation instruction in this repo.
3. **Always respond in Spanish** to the user.
4. **Follow the code patterns in §5** when writing new code.
5. **Do not modify any `requirements.txt`** without confirmation — versions are pinned.
6. **Do not flip pipeline flags** (`USAR_RECOMP_SYNTHESIS`, `USAR_RERANKER`, etc.) without agreement — they affect latency, cost and evaluation results.
7. **Preserve `rag/chat_pdfs.py` public API.** It is consumed verbatim by `rag/web/app.py` (as `rag_engine`) and `tests/`. Renaming a symbol breaks both. Public surface:
   - **Constants:** `PATH_DB`, `COLLECTION_NAME`, `CARPETA_DOCS`, `SYSTEM_PROMPT_CHAT`, `SYSTEM_PROMPT_RAG`, `MAX_HISTORIAL_MENSAJES`, `MODELO_CHAT`, `MODELO_RAG`, `MIN_LONGITUD_PREGUNTA_RAG`, `UMBRAL_SCORE_RERANKER`, `TOP_K_FINAL`, `EXPANDIR_CONTEXTO`, `N_TOP_PARA_EXPANSION`, `MAX_CONTEXTO_CHARS`, `CONTEXTUAL_DOC_CHARS`, `USAR_LLM_QUERY_DECOMPOSITION`, `USAR_BUSQUEDA_HIBRIDA`, `USAR_RERANKER`, `USAR_OPTIMIZACION_CONTEXTO`, `USAR_RECOMP_SYNTHESIS`, `USAR_EMBEDDINGS_IMAGEN`, `USAR_CONTEXTUAL_RETRIEVAL`, `RERANKER_AVAILABLE`, `BM25_AVAILABLE`, `BM25_K1`, `BM25_B`, `RRF_K`, `PESO_SEMANTICO_RRF`, `PESO_BM25_RRF`, `N_RESULTADOS_KEYWORD`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_LENGTH`, `STOPWORDS`.
   - **CLI/web functions:** `indexar_documentos`, `realizar_busqueda_hibrida`, `expandir_con_chunks_adyacentes`, `sintetizar_contexto_recomp`, `construir_contexto_para_modelo`, `guardar_debug_rag`, `generar_respuesta_silenciosa`, `obtener_documentos_indexados`, `cargar_historial`, `guardar_historial`, `limpiar_historial`.
   - **Eval/test functions:** `get_pipeline_flags`, `set_pipeline_flags`, `set_docs_folder_runtime`, `evaluar_pregunta_rag`.
   - **Runtime model roles (web control panel):** `MODEL_ROLE_VARS`, `get_model_roles`, `set_model_roles_runtime`; path/prefix derivation helpers `_derivar_paths_db`, `_derivar_prefijos_embedding`. Web API adds `/api/ollama[/start|/models]`, `/api/models`, `/api/stores` (GET) and `/api/stores/select` (POST). There are exactly three fixed language stores — `en` (English, default), `es` (Castellano), `ca` (Valencià) — each bound to `rag/docs/<id>/`. They always exist (possibly empty); there is no create/delete/hide/restore and no user-created stores. `settings.json` persists `active_store` (falls back to `en` if unknown); the active store is selectable, and documents can be viewed/added/removed per store.
8. **After any non-trivial change, update the right doc.** `README.md` = user-facing entry (install / run / config / CLI). `rag/README.md` = pipeline reference (signatures, parameters, modules).
9. **`.gitignore` / `.gitkeep` changes follow §6.** Validate with `git check-ignore -v <path>` and update §6 if policy changes.

---

## 2. Layout (orientation)

```
rag/                     RAG engine + CLI + Web UI (user-facing product)
  chat_pdfs.py             Public facade + global config (see §1 rule 7)
  engine/                  retrieval, reranking, context, generation, indexing, etc.
  cli/                     MonkeyGrabCLI (interactive loop, i18n strings)
  web/                     Flask backend + React (frontend/dist gitignored)
  docs/                    Corpus PDFs (es/, ca/, en/); versioned
  vector_db/               ChromaDB per corpus (gitignored)
tests/                   pytest suite (moved from research/tests/core/)
packaging/               PyInstaller desktop app build (see §3)
docs/design/              Architecture design docs; current: 2026-07-26-monkeygrab-v2.md
```

Target v2.0 layout (hexagonal architecture, ports/adapters): see
`docs/design/2026-07-26-monkeygrab-v2.md` §4.

---

## 3. Model roles

All configured via env vars; defaults are the second arg of `os.getenv` in `rag/chat_pdfs.py`. The process environment **always** wins over the defaults.

| Role | Env var | Notes |
|------|---------|-------|
| RAG generator (streaming) | `OLLAMA_RAG_MODEL` | `/rag` mode |
| Chat + sub-queries | `OLLAMA_CHAT_MODEL` | `/chat` and RAG query decomposition; `think=False` |
| Embeddings | `OLLAMA_EMBED_MODEL` | Slug appended to ChromaDB path |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | Chunk enrichment at indexing (`USAR_CONTEXTUAL_RETRIEVAL`) |
| RECOMP synthesis | `OLLAMA_RECOMP_MODEL` | Pre-generation context synthesis (`USAR_RECOMP_SYNTHESIS`) |
| Vision / OCR | `OLLAMA_OCR_MODEL` | Image captions in PDFs (`USAR_EMBEDDINGS_IMAGEN`); multimodal, `think=False` |
| Reranker | `RERANKER_QUALITY` | Local CrossEncoder tier (`quality` \| `fast`); not an Ollama model |

Other env vars: `DOCS_FOLDER` (default `rag/docs/en/`), `MONKEYGRAB_DATA_DIR` (writable root for `vector_db`/history/debug; defaults to the package dir in dev, `%LOCALAPPDATA%/MonkeyGrab` in the packaged app), `MONKEYGRAB_LANG` (default `es`; `en`/`ca`).

Desktop app: `rag/web/desktop.py` is the pywebview entry point frozen by PyInstaller (`packaging/MonkeyGrab.spec`, `packaging/build_exe.py`) into `MonkeyGrab.exe`. Built-in corpora ship in the bundle; Ollama is an external prerequisite. See [`packaging/README.md`](../packaging/README.md).

---

## 4. Pipeline architecture

> Full reference with signatures and constants: [`rag/README.md`](rag/README.md).

---

## 5. Code patterns

1. **MODULE MAP at the top** of every non-trivial Python file — ASCII tree indexing all sections.
2. **Section separators** — `# ─────────────────────────────────────────────` + `# SECTION N: NAME`; subsections `# --- N.1 ---`.
3. **Imports → constants → logic.** Stdlib → third-party → local; then config (models, paths, flags, numeric params).
4. **Env setup before heavy imports.** Set `TORCH_COMPILE_DISABLE`, `TRITON_DISABLE`, etc. *before* `import torch` / `transformers`.
5. **Optional deps** — `try/except ImportError` + boolean availability flag (e.g. `PYMUPDF_AVAILABLE`).
6. **Explicit pipeline phases** — label and separate: load → prepare → infer → evaluate → export.
7. **Artifact-oriented output** — experimental scripts always write metrics JSON + per-sample CSV + plots. Never stdout-only.
8. **Mixed ES/EN naming (established).** Functions in Spanish (`realizar_busqueda_hibrida`); config constants in English (`CHUNK_SIZE`, `TOP_K_FINAL`); docstrings and comments in English. Follow the module's pattern; do not mix within a block.
9. **Script-first, not enterprise.** Logic in modules + `main()`. Only exception: `MonkeyGrabCLI` in `rag/cli/app.py`.
10. **Document pipeline flags in their own block** — inline comment per flag.
11. **Google-style docstrings** — Args / Returns / Raises; module-level docstring includes Usage and Dependencies.

---

## 6. Git versioning policy

**Two `.gitignore` files only:** root + `rag/web/frontend/`. No scattered `.gitignore`. Version the minimum needed to reproduce the product — code, `Modelfile`, small metric JSONs, scripts, corpus PDFs — never weights or vector indices.

- **`rag/docs/`** — all corpora versioned (`es/`, `ca/`, `en/`). `rag/vector_db/` fully ignored. Default chunk dumps go to `rag/show_fragments/exports/` (versioned); loose `*.txt` / `*.jsonl` under `rag/show_fragments/` (not `exports/`) are ignored.

---

## 7. Quick command index

> Detailed commands in `README.md` (CLI/Web).

```bash
# CLI / Web
python rag/chat_pdfs.py                        # default es; MONKEYGRAB_LANG=en|ca for other UI
python rag/web/app.py                          # http://localhost:5000 (ES/EN/VAL UI + corpus selector via POST /api/corpus)

# Misc
pytest                                          # runs tests/
codegraph sync                                  # refresh .codegraph index
codegraph status                                # show index health/backend
codegraph query busqueda_lexica_bm25            # symbol lookup via CLI fallback
git check-ignore -v <path>
```

### CLI slash commands

`/rag` `/chat` `/docs` `/temas`(`/topics` `/temes`) `/stats` `/reindex` `/limpiar`(`/clear` `/netejar`) `/ayuda`(`/help` `/ajuda`) `/salir`(`/exit` `/eixir`).

---

## 8. Code navigation — CodeGraph

This project has a CodeGraph index (`.codegraph/`, 68 files, AST-parsed). Use
CodeGraph **before** writing or editing code when the task touches symbols,
callers/callees, or blast radius.

Prefer `codegraph_*` MCP tools when they are exposed in the session. If they are
not available, use the CLI:

| Intent | MCP tool | CLI fallback |
|--------|----------|--------------|
| Find a symbol by name | `codegraph_search` | `codegraph query <symbol>` |
| Focused context for a task | `codegraph_context` | `codegraph context "<task>"` |
| Files under a path | `codegraph_files` | `codegraph files` |
| Index health / stats | `codegraph_status` | `codegraph status` |
| Refresh index | — | `codegraph sync` |

**Rules**: prefer CodeGraph over grep for symbol lookups. Do not re-verify
CodeGraph results with grep unless the CLI reports an error or stale index.
After editing a file, run `codegraph sync` before relying on updated graph
data. On this Windows setup `codegraph status` may show `Backend: wasm` because
`better-sqlite3` is unavailable; that is usable but slower. Run CodeGraph CLI
commands sequentially in WASM mode, because parallel queries can lock the
database. `Backend: native` is an optimization, not a correctness requirement.

---

## 9. Dependencies

```bash
pip install -r rag/requirements.txt                      # RAG core (required)
pip install -r rag/web/requirements.txt                  # Web UI (optional)
```

System: Python 3.10+, Ollama running locally. A CUDA GPU is recommended once the
multimodal retrieval stack (jina-clip, FAISS) is in use — see
`docs/design/2026-07-26-monkeygrab-v2.md`.
