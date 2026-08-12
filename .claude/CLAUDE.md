# CLAUDE.md — MonkeyGrab (localOllamaRAG)

Local RAG over PDFs, with hybrid retrieval and a local Ollama LLM. Production
product only — the thesis research layer was defended and removed from HEAD
(history preserved at tag `tfg-final`).

Hexagonal architecture: `src/monkeygrab/` holds domain, ports, application use
cases and adapters; `rag/` holds the interfaces (CLI, web) and the wiring that
builds the adapters and runs the use cases. All three pipeline stages —
indexing, retrieval, generation — go through the core. See §2 for the
dependency rule and the layout.

User-facing doc lives in `README.md`. Architecture entry points:
`src/monkeygrab/README.md` and `rag/README.md`. Documentation standard (what
gets documented where, what doesn't get documented at all): `docs/README.md`.

This file is the operating contract for **any** agent working here, not just
Claude. `.codex/AGENTS.md` exists only to point at it and must stay a pointer:
it once held a second copy and spent months describing a research layer,
fine-tuned models and evaluation hooks that had already been deleted.

---

## 1. Behavior rules

1. **Autonomous delivery.** Branch, PR, CI, squash-merge and post-merge
   verification run without asking. Stop for explicit approval only where the
   global protocol says to — production data or migrations, destructive
   operations, auth/PII/secrets, irreversible external integrations, anything
   without trivial rollback — or where a rule below demands agreement
   (5, 6, 9). Repo conventions win over any plan that contradicts them.
2. **Two design docs are current and they do not overlap.** `docs/design/2026-07-26-monkeygrab-v2.md` governs the product architecture and takes precedence over any other architecture instruction in this repo. `docs/design/2026-07-28-loop-automejorable.md` governs the evaluation gate and the self-improving loop built on it. Neither outranks the other; they answer different questions. `docs/README.md` governs *how* things get documented.
3. **Always respond in Spanish** to the user.
4. **Follow the code patterns in §6** when writing new code in `rag/`; §7 lists what `src/monkeygrab/` does differently.
5. **Do not modify any `requirements.txt`** without confirmation — versions are pinned.
6. **Do not flip pipeline flags** (`USAR_RECOMP_SYNTHESIS`, `USAR_RERANKER`, etc.) without agreement — they affect latency, cost and evaluation results.
7. **Preserve `rag/chat_pdfs.py` public API.** It is consumed verbatim by `rag/web/app.py` (as `rag_engine`), the CLI and `tests/`. Renaming a symbol breaks all three at once. The authoritative list is the re-export block at the end of that file; the groups are:
   - **Constants:** paths and collection names, model roles, every pipeline flag and numeric parameter, both system prompts, plus the `*_AVAILABLE` compatibility constants the UIs display.
   - **Pipeline entry points:** `indexar_documentos`, `realizar_busqueda_hibrida`, `preparar_fragmentos_para_generacion`, `generar_respuesta`, `generar_respuesta_silenciosa`, `generar_tokens_respuesta`, `evaluar_pregunta_rag`.
   - **Support:** context assembly, debug dumps, chat history, `obtener_documentos_indexados`, and the text helpers re-exported from `monkeygrab.application.keywords` (`STOPWORDS`, `extract_keywords`, ...).
   - **Runtime switches (web control panel):** `get_pipeline_flags`, `set_pipeline_flags`, `set_docs_folder_runtime`, `MODEL_ROLE_VARS`, `get_model_roles`, `set_model_roles_runtime`, plus the path derivation helper `_derivar_paths_db`.

   The web API adds `/api/ollama[/start|/models]`, `/api/models`, `/api/stores` (GET) and `/api/stores/select` (POST). There are exactly three fixed language stores — `en` (English, default), `es` (Castellano), `ca` (Valencià) — each bound to `rag/docs/<id>/`. They always exist, possibly empty; there is no create/delete/hide/restore and no user-created stores. `settings.json` persists `active_store`, falling back to `en` if unknown. The active store is selectable, and documents can be viewed, added and removed per store.
8. **Hard-fail policy, project-wide.** Every adapter under `src/monkeygrab/adapters/` raises on failure instead of degrading — no silent CUDA→CPU fallback, no silent extractor swap, no silent RECOMP-to-raw-context fallback. Do not add a fallback chain inside an adapter; if a caller needs one, it composes two ports explicitly. See `docs/design/2026-07-26-monkeygrab-v2.md` §3 ("Política de fallos").
9. **Test boundaries are load-bearing, not incidental:**
   - `tests/characterization/` pins the *current* pipeline's observed behavior, including its known bugs. Do not edit these tests to make a change pass — if a change legitimately alters behavior, that's a signal to stop and confirm, not to update the test. The one documented exception is `test_stale_default_config_bug.py`, whose own docstring says exactly when and how it is allowed to change.
   - `tests/eval/` holds the gold-case evaluation gate (`gold_cases.jsonl`, `run_eval.py`, `grade.py`) with a hand-verified expected answer per case. It is the acceptance gate for anything touching retrieval or generation — do not weaken a grading rule to raise the pass rate.
   - `tests/unit/test_architecture_boundaries.py` enforces the dependency rule in §2 by parsing imports; don't special-case it.
10. **After any non-trivial change, update the right doc.** `README.md` = user-facing entry (install / run / config / CLI). `src/monkeygrab/README.md` / `rag/README.md` = architecture entry points. See `docs/README.md` for what does *not* need documenting.
11. **`.gitignore` / `.gitkeep` changes follow §8.** Validate with `git check-ignore -v <path>` and update §8 if policy changes.

---

## 2. Architecture

**Dependency rule** (enforced by `tests/unit/test_architecture_boundaries.py`,
which parses imports — not a convention to trust by eye): `application` may
import `domain`/`ports`/`config`; `ports` may import `domain`; `domain` and
`config` import nothing internal. `adapters` implement `ports` but are never
imported *by* `domain`/`ports`/`config`/`application`. Full explanation and
how to add a new adapter: [`src/monkeygrab/README.md`](../src/monkeygrab/README.md).

```
src/monkeygrab/          Hexagonal core (see src/monkeygrab/README.md)
  domain/                  Entities, zero infrastructure imports
  ports/                   Protocols the application layer depends on
  application/             Use cases (IndexCorpus, Retrieve, Answer) + pure helpers
  config/                  AppConfig — immutable, from_env() / with_overrides()
  adapters/                Port implementations: MinerU, jina-clip, FAISS, Ollama, BM25, BGE
rag/                      Interfaces (CLI + web) and pipeline entry points
  chat_pdfs.py              Public facade + global config (see §1 rule 7)
  engine/                    wiring, retrieval, indexing, context, generation, chunking, debug, history
  cli/                       MonkeyGrabCLI (interactive loop, i18n strings)
  web/                       Flask backend + React frontend (pnpm; frontend/dist gitignored)
  docs/                      Corpus PDFs (es/, ca/, en/); versioned
  vector_db/                 FAISS per corpus (gitignored)
tests/
  unit/                      domain/ports/config/application + adapters, doubled infrastructure
  characterization/          pins current pipeline behavior — do not edit (§1 rule 9)
  eval/                      gold-case evaluation gate (§5) — do not touch without agreement
harness/                  Configuration search harness (issue #31); not product — see harness/README.md
packaging/                PyInstaller desktop app build (see §3)
docs/design/               Architecture design docs; current: 2026-07-26-monkeygrab-v2.md
docs/README.md             Documentation standard
```

**Wiring today, not aspirational:**

- **All three stages run through the core.** `rag/engine/indexing.py` runs
  `IndexCorpus`, `rag/engine/retrieval.py` runs `Retrieve`, and
  `rag/engine/generation.py` runs `Answer` — the same use cases
  `tests/eval/run_eval.py` constructs, which is what stops the evaluation gate
  and the shipped product from measuring different behavior. Every one builds
  its ports through `rag/engine/wiring.py`, the single bridge between this
  package's mutable globals and the immutable `AppConfig` the core takes.
- **`rag/engine/` is wiring, not logic.** Each module builds adapters, calls a
  use case, and converts between domain entities and the dicts the interfaces
  consume. `rag/engine/context.py` remains the home of the pure context-
  assembly helpers the core imports.
- `Answer` exposes `select_evidence` / `build_user_message` / `stream`
  separately as well as a composing `run`. The split is load-bearing: the web
  layer sends cited sources before the first token, so it cannot have prompt
  preparation and generation in one call.
- `rag/engine/wiring.py` caches the Cross-Encoder and the tokenized BM25
  corpus. Anything else it builds is cheap and rebuilt per call, which is what
  makes a runtime model or flag change take effect on the next query.

---

## 3. Models

Ollama roles are configured via env vars; defaults are the second arg of
`os.getenv` in `rag/chat_pdfs.py`. The process environment **always** wins over
the defaults. Jina CLIP v2 and BGE Reranker v2 M3 are fixed.

| Role | Env var | Notes |
|------|---------|-------|
| RAG generator (streaming) | `OLLAMA_RAG_MODEL` | `/rag` mode |
| Chat + sub-queries | `OLLAMA_CHAT_MODEL` | `/chat` and RAG query decomposition; `think=False` |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | Chunk enrichment at indexing (`USAR_CONTEXTUAL_RETRIEVAL`) |
| RECOMP synthesis | `OLLAMA_RECOMP_MODEL` | Pre-generation context synthesis (`USAR_RECOMP_SYNTHESIS`) |
| Reranker | fixed | `BAAI/bge-reranker-v2-m3`; not an Ollama model |

Other env vars: `DOCS_FOLDER` (default `rag/docs/en/`), `MONKEYGRAB_DATA_DIR` (writable root for `vector_db`/history/debug; defaults to the package dir in dev, `%LOCALAPPDATA%/MonkeyGrab` in the packaged app), `MONKEYGRAB_LANG` (default `es`; `en`/`ca`), `OLLAMA_BASE_URL` (default `http://localhost:11434`, falling back to Ollama's own `OLLAMA_HOST`).

The Ollama endpoint has exactly one reader, `monkeygrab.config.env.read_env_ollama_base_url`, feeding `AppConfig.models.ollama.base_url` and `rag.chat_pdfs.OLLAMA_BASE_URL`. Every generation call, both `/chat` modes and both reachability checks (CLI startup, web control panel) resolve through it. Do not re-read the variable at a call site: a second resolution is how the CLI health check ended up reporting a server the pipeline never talked to.

Desktop app: `rag/web/desktop.py` is the pywebview entry point frozen by PyInstaller (`packaging/MonkeyGrab.spec`, `packaging/build_exe.py`) into `MonkeyGrab.exe`. Built-in corpora ship in the bundle; Ollama is an external prerequisite. See [`packaging/README.md`](../packaging/README.md).

---

## 4. Pipeline architecture

> Full reference with signatures and constants: [`rag/README.md`](../rag/README.md) → [`src/monkeygrab/README.md`](../src/monkeygrab/README.md). Pipeline behavior as currently observed: `tests/characterization/`.

---

## 5. Testing and CI gates

Two gates, deliberately not one — a job that doubles infrastructure is never
presented as pipeline coverage, and vice versa:

- **Fast gate** (`.github/workflows/ci.yml`, every PR, hosted runner, no GPU/Ollama/model
  downloads): lint (`ruff`), `tests/unit` + `tests/eval`'s grader + `harness/tests`
  against test doubles (`architecture` job — must import nothing but the
  standard library, since it exercises `domain`/`ports`/`config`/`application`
  and the harness's own fake-evaluator tests), frontend build
  (`pnpm install --frozen-lockfile` + `pnpm run lint` + `pnpm run build`), and
  the full `pytest` suite with real dependencies but a doubled/absent Ollama
  server (`engine` job — Ollama-dependent tests skip themselves when no
  server answers).
- **Full gate** (`.github/workflows/full-eval.yml`, `workflow_dispatch` only,
  self-hosted GPU runner with Ollama installed): runs the real pipeline —
  Ollama generation, Jina CLIP embeddings, hybrid BM25+semantic retrieval,
  BGE reranking — against every case in `tests/eval/gold_cases.jsonl`
  via `tests/eval/run_eval.py`, and fails if the pass rate drops below
  `tests/eval/baseline_min_pass_rate.txt`. This is the gate that must be
  green, together with the fast gate, before merging any change to retrieval
  or generation (design doc §7.2/7.3) — the fast gate alone never exercises
  the real model or retrieval path.

See §1 rule 9 for what must not be edited (`tests/characterization/`,
`tests/eval/`) and why.

---

## 6. Code patterns — `rag/`

1. **No decorative banners.** No `MODULE MAP` trees, no box-drawing rules, no numbered section headers. A plain `# Title` comment is fine where a file genuinely has parts; the module docstring says what the module is for.
2. **Google-style docstrings** — Args / Returns / Raises. Every module opens with a docstring stating its purpose; PEP 257 for everything else (one-line docstrings on one line).
3. **Comments explain why, not what.** If a comment restates the line below it, delete it. The comments worth writing are the ones that record a constraint, a trade-off or a defect the code works around.
4. **Imports → constants → logic.** Stdlib → third-party → local; then config (models, paths, flags, numeric params).
5. **Env setup before heavy imports.** Set `TORCH_COMPILE_DISABLE`, `TRITON_DISABLE`, etc. *before* `import torch` / `transformers`.
6. **Mixed ES/EN naming (established).** Functions in Spanish (`realizar_busqueda_hibrida`); config constants in English (`CHUNK_SIZE`, `TOP_K_FINAL`); docstrings and comments in English. Follow the module's pattern; do not mix within a block.
7. **Script-first, not enterprise.** Logic in modules + `main()`. Only exception: `MonkeyGrabCLI` in `rag/cli/app.py`.
8. **Document pipeline flags in their own block** — inline comment per flag.

---

## 7. Code patterns — `src/monkeygrab/`

Items 1–5 of §6 apply here too. What differs:

1. **English naming throughout** (`Retrieve`, `Answer`, `IndexCorpus`, `ChunkMetadata`) — no Spanish function names, no ES/EN split.
2. **No service locator.** Every use case takes its ports and its `AppConfig` through the constructor/call and reads config fresh — never captured in a default argument (see `tests/characterization/test_stale_default_config_bug.py` for the bug this structurally forecloses).
3. **Ports are `Protocol`s.** Adapters satisfy them structurally; they do not inherit from the port.
4. **Hard-fail** (§1 rule 8).

---

## 8. Git versioning policy

**Two `.gitignore` files only:** root + `rag/web/frontend/`. No scattered `.gitignore`. Version the minimum needed to reproduce the product — code, `Modelfile`, small metric JSONs, scripts, corpus PDFs — never weights or vector indices.

- **`rag/docs/`** — all corpora versioned (`es/`, `ca/`, `en/`). `rag/vector_db/` fully ignored. Default chunk dumps go to `rag/show_fragments/exports/` (versioned); loose `*.txt` / `*.jsonl` under `rag/show_fragments/` (not `exports/`) and local scratch results under `pipeline/output/` are ignored.

---

## 9. Quick command index

> Detailed commands in `README.md` (CLI/Web).

```bash
# CLI / Web
python rag/chat_pdfs.py                        # default es; MONKEYGRAB_LANG=en|ca for other UI
python rag/web/app.py                          # http://localhost:5000 (ES/EN/VAL UI + corpus selector via POST /api/corpus)

# Frontend (pnpm only — package.json pins packageManager, CI enforces --frozen-lockfile)
cd rag/web/frontend && pnpm install && pnpm run build

# Tests / CI gates
pytest                                          # full suite (unit + characterization + eval grader + harness + loose tests)
pytest tests/unit tests/eval harness/tests --ignore=tests/unit/adapters   # what the fast "architecture" CI job runs
python tests/eval/run_eval.py --models <model...>           # full gate locally; needs Ollama + GPU

# Configuration search harness (issue #31) — not product, see harness/README.md
python -m harness.cli --dry-run --max-iterations 3          # smoke-test the loop, no GPU/Ollama needed

# Misc
git check-ignore -v <path>
```

### CLI slash commands

`/rag` `/chat` `/docs` `/temas`(`/topics` `/temes`) `/stats` `/reindex` `/limpiar`(`/clear` `/netejar`) `/ayuda`(`/help` `/ajuda`) `/salir`(`/exit` `/eixir`).

---

## 11. Dependencies

```bash
pip install -r rag/requirements.txt                      # RAG core (required)
pip install -r rag/web/requirements.txt                  # Web UI (optional)
```

`src/monkeygrab/` has no separate install step: it is not packaged, just
added to `sys.path` (`rag/chat_pdfs.py`'s bootstrap; `pytest.ini`'s
`pythonpath = . src`). `domain`/`ports`/`config`/`application` need nothing
beyond the standard library; `adapters/` reuse what `rag/requirements.txt`
installs (FAISS, Ollama, Pillow, sentence-transformers). MinerU and Jina CLIP
run through the isolated `.venv-mineru` environment.

System: Python 3.10+, Ollama running locally. A CUDA GPU is recommended once the
multimodal retrieval stack (jina-clip, FAISS) is in use — see
`docs/design/2026-07-26-monkeygrab-v2.md`.
