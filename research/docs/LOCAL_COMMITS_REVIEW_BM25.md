# Revision local de commits BM25

Base revisada: `origin/main` (`7321525`).

Commits revisados:

1. `83ecdaf` - `chore: update default RAG model`
2. `3fd0036` - `feat: replace lexical retrieval with BM25`
3. `3b063aa` - `refactor: simplify web pipeline controls`
4. `7772440` - `docs: document BM25 migration`

Restricciones:

- Sin push.
- Sin commit sin confirmacion explicita.
- No tocar `llama.cpp/`.
- No modificar `requirements.txt` salvo que la auditoria lo exija y quede justificado.

## Criterios de revision

- API publica de `rag/chat_pdfs.py` conservada para CLI, web y evaluacion.
- Imports, rutas y entrypoints siguen resolviendo.
- La busqueda BM25 sustituye la busqueda lexica anterior sin dejar referencias funcionales rotas.
- Los controles web siguen alineados con las banderas reales del pipeline.
- La documentacion no contradice el codigo vigente.
- Las pruebas relevantes pasan o quedan bloqueadas por una causa reproducible de entorno.

## Resultado ejecutivo

La migracion a BM25 no rompe la API publica de `rag/chat_pdfs.py` ni el flujo
de evaluacion probado por `test_run_eval_checkpoint.py`.

Si habia dos problemas corregibles:

1. `3b063aa` eliminaba la ruta backend `POST /api/upload`. La UI ya no la usa,
   pero era una ruta preexistente. Se ha restaurado para compatibilidad sin
   reintroducir el boton simplificado fuera de `/api/reindex`.
2. `busqueda_lexica_bm25()` podia lanzar `ZeroDivisionError` si la coleccion
   tenia documentos pero todos tokenizaban a listas vacias. Se ha anadido un
   guard que devuelve `[]` en ese caso.

Tambien habia restos de texto vivo que seguian hablando de keywords/exhaustiva:
CLI, log de retrieval, generador de diagrama y mapa de modulos de `rag/README.md`.
Se han actualizado a BM25. Las referencias historicas dentro de
`BM25_MIGRATION.md` se mantienen porque documentan el antes/despues.

No se ha hecho commit ni push.

## Revision por commit

### `83ecdaf` - `chore: update default RAG model`

Cambio real: `MODELO_RAG` pasa de `phi4-finetuned:latest` a `gemma4:e4b`.

Decision: aceptable. El patron `os.getenv("OLLAMA_RAG_MODEL", ...)` conserva la
sobrescritura por entorno, por lo que no rompe configuraciones externas. El
riesgo restante es operativo: si `gemma4:e4b` no existe en Ollama local, fallara
en tiempo de uso, no en import/ruta/API.

Verificacion en worktree detached:

- `py_compile`: OK.
- API publica de `rag/chat_pdfs.py`: sin simbolos faltantes.
- `POST /api/upload`: presente.
- `POST /api/reindex`: presente.
- `pytest research/tests/evaluation/test_run_eval_checkpoint.py -q -p no:cacheprovider`: 8 passed.

### `3fd0036` - `feat: replace lexical retrieval with BM25`

Cambio real: sustituye busqueda por `$contains` + escaneo exhaustivo por
`rank-bm25` (`BM25Okapi`), elimina `USAR_BUSQUEDA_EXHAUSTIVA` de los flags de
runtime/evaluacion y mantiene `USAR_BUSQUEDA_HIBRIDA` como interruptor de la
rama lexica.

Decision: aceptable con una correccion. La API publica exigida sigue intacta:
`realizar_busqueda_hibrida`, `get_pipeline_flags`, `set_pipeline_flags`,
`evaluar_pregunta_rag`, etc. siguen resolviendo. La eliminacion de
`USAR_BUSQUEDA_EXHAUSTIVA` no rompe la lista publica local y queda alineada con
la tesis: la variante `no_exhaustive_search` ya no tiene sentido.

Correccion aplicada durante esta revision:

- `rag/engine/lexical.py`: si la coleccion contiene documentos pero ningun token
  BM25 util, se devuelve lista vacia antes de construir `BM25Okapi`. Evita
  `ZeroDivisionError`.
- `research/tests/core/test_bm25_lexical.py`: cubre ranking BM25 positivo,
  query sin tokens y corpus tokenizado vacio.

Verificacion en worktree detached:

- `py_compile`: OK.
- API publica de `rag/chat_pdfs.py`: sin simbolos faltantes.
- Runtime flags: ya no incluyen `USAR_BUSQUEDA_EXHAUSTIVA`.
- `POST /api/upload`: presente en este commit.
- `pytest research/tests/evaluation/test_run_eval_checkpoint.py -q -p no:cacheprovider`: 8 passed.

### `3b063aa` - `refactor: simplify web pipeline controls`

Cambio real: simplifica controles de la UI, elimina el toggle de busqueda
exhaustiva y mueve la opcion de imagenes a indexacion. Tambien elimina el
endpoint backend `POST /api/upload`.

Decision: el cambio de UI es coherente con BM25, pero la eliminacion de
`/api/upload` rompe una ruta preexistente. Aunque el frontend actual use
`/api/reindex`, quitar el endpoint reduce compatibilidad sin necesidad.

Correccion aplicada durante esta revision:

- `rag/web/app.py`: restaurado `POST /api/upload` con el comportamiento anterior
  (`add_only=1` para indexado incremental, sin `add_only` para reindexado).
- `research/tests/core/test_web_routes.py`: fija que `/api/upload` y
  `/api/reindex` sigan registrados.

Verificacion en worktree detached antes de la correccion:

- `py_compile`: OK.
- API publica de `rag/chat_pdfs.py`: sin simbolos faltantes.
- `POST /api/upload`: faltante.
- `POST /api/reindex`: presente.
- `pytest research/tests/evaluation/test_run_eval_checkpoint.py -q -p no:cacheprovider`: 8 passed.

### `7772440` - `docs: document BM25 migration`

Cambio real: actualiza README/docs de evaluacion y anade `BM25_MIGRATION.md` y
`REINFERENCIA_BM25.md`.

Decision: documentacion principal correcta en lo esencial. Quedaban referencias
vivas desalineadas fuera del documento historico de migracion.

Correcciones aplicadas durante esta revision:

- `rag/README.md`: mapa de modulos actualizado a "Busqueda lexica BM25".
- `research/utils/generate_diagram.py`: diagrama actualizado de
  "Semantic + Keyword + Exhaustive" a "Semantic + BM25 Lexical Search".
- `rag/cli/strings.py`: estado CLI de busqueda hibrida actualizado a BM25 en
  ES/EN/CA.
- `rag/engine/retrieval.py`: log de pipeline actualizado de `Keywords(...)` a
  `BM25(...)`.
- `rag/engine/lexical.py`: comentario de seccion actualizado.

Verificacion en worktree detached antes de la correccion:

- `py_compile`: OK.
- API publica de `rag/chat_pdfs.py`: sin simbolos faltantes.
- `POST /api/upload`: seguia faltante por `3b063aa`.
- `POST /api/reindex`: presente.
- `pytest research/tests/evaluation/test_run_eval_checkpoint.py -q -p no:cacheprovider`: 8 passed.

## Verificacion final del estado actual

Pasado:

- `python -m py_compile` sobre los modulos modificados y pruebas nuevas: OK.
- Import check: `rank_bm25`, `rag.chat_pdfs`, `rag.engine.lexical`,
  `rag.engine.retrieval`, `rag.web.app`,
  `research.evaluation._lib.pipeline_flags`: OK.
- `PUBLIC_API_MISSING=[]`.
- `BM25_AVAILABLE=True` en este entorno.
- Rutas Flask: `/api/upload=True`, `/api/reindex=True`.
- Smoke BM25:
  - corpus normal: 1 resultado, `doc0.pdf_pag0_chunk0`, score `0.6684`.
  - corpus tokenizado vacio: 0 resultados, sin excepcion.
- `pytest research/tests/core/test_bm25_lexical.py research/tests/core/test_web_routes.py -q -p no:cacheprovider`: 4 passed.
- `pytest research/tests/evaluation/test_run_eval_checkpoint.py -q -p no:cacheprovider`: 8 passed.
- `git diff --check`: sin errores de whitespace; solo avisos de normalizacion LF/CRLF.

Bloqueos no atribuibles a estos commits:

- `pytest research/tests/core -q -p no:cacheprovider` falla en
  `test_backend_defaults_to_prompt_toolkit_on_windows`: el test espera
  `prompt_toolkit`, pero `origin/main` ya devuelve `rich` por diseno actual de
  `_select_backend()`. No esta causado por BM25.
- `pytest research/tests/evaluation -q -p no:cacheprovider` falla en coleccion
  por `ModuleNotFoundError: repeticion_run_eval`, coincidiendo con el problema
  preexistente indicado.

## Cambios locales pendientes

- `rag/web/app.py`: restaurada ruta `/api/upload`.
- `rag/engine/lexical.py`: guard para corpus BM25 sin tokens + comentario.
- `rag/engine/retrieval.py`: log `BM25(...)`.
- `rag/cli/strings.py`: etiquetas de busqueda hibrida BM25.
- `research/utils/generate_diagram.py`: diagrama BM25.
- `rag/README.md`: mapa de modulos BM25.
- `research/tests/core/test_bm25_lexical.py`: pruebas BM25.
- `research/tests/core/test_web_routes.py`: prueba de rutas web.
- `research/docs/LOCAL_COMMITS_REVIEW_BM25.md`: este informe.
