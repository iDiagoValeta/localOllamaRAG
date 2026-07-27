# MonkeyGrab v2.0 — Diseño

**Fecha:** 2026-07-26
**Estado:** aprobado · en ejecución
**Alcance:** reestructuración completa del repositorio hacia clean architecture, eliminación de la
capa de investigación, CI verificable por fases, y adopción del stack MinerU + embeddings
multimodales + FAISS detrás de puertos sustituibles.

> [!NOTE]
> **Estado a 2026-07-27.** F0, F1 y F2 están en `main`. Los adaptadores de F3
> existen y están verificados, pero el default del stack sigue siendo
> pymupdf + Ollama + Chroma.
>
> Del núcleo hexagonal, **indexación y recuperación** se ejecutan a través de
> sus casos de uso (`IndexCorpus`, `Retrieve`); **generación no**, y `Answer`
> sigue sin cablear. `get_runtime()` no ha desaparecido: `rag/engine/wiring.py`
> lo encapsula como único puente hacia `AppConfig`, en lugar de que cada módulo
> lo consulte por su cuenta. La aceptación de F1 describe el objetivo, no lo
> que hay hoy.

---

## 1. Punto de partida

El repositorio nació como TFG y contiene dos capas mezcladas: el producto (`rag/`, ~11k líneas)
y la investigación ya defendida (`research/`, ~18,5k líneas). El producto funciona, pero su
estructura impide evolucionarlo:

- **`chat_pdfs.py` es un service locator, no un módulo de configuración.** Los módulos de
  `rag/engine/` hacen `cfg = get_runtime()`, que devuelve el propio módulo `chat_pdfs`, y de ahí
  leen tanto constantes como las librerías importadas (`cfg.fitz`, `cfg.BM25Okapi`,
  `cfg.CrossEncoder`, `cfg.pymupdf4llm`). El orden de imports de un fichero determina si el motor
  arranca.
- **La configuración en caliente funciona a medias.** Varias funciones capturan config en sus
  argumentos por defecto (`def dividir_en_chunks(texto, chunk_size=cfg.CHUNK_SIZE)`), evaluados al
  importar el módulo. Las lecturas dentro del cuerpo sí son dinámicas. Consecuencia: los cambios
  del panel web aplican a algunos parámetros y a otros no, sin aviso.
- **La configuración tiene tres fuentes de verdad simultáneas:** variables globales mutables de
  Python, `settings.json`, y el `useState` de un `App.tsx` de 2022 líneas.
- **La web depende de internals.** `rag/web/app.py` importa `_derivar_paths_db`,
  `_generar_respuesta_stream` y `_preparar_mensaje_usuario_rag`.
- **Los fallbacks silenciosos ocultan fallos.** El reranker degrada CUDA→CPU→sin-reranking;
  la extracción degrada pymupdf4llm→pypdf; la síntesis RECOMP cae al contexto crudo en cinco
  condiciones distintas. Todas ellas producen resultados válidos pero incomparables entre corridas.
- **El CI produce un falso verde.** `.github/workflows/rag-eval.yml` tiene un job
  `self-improve-gate-docs` que solo imprime texto y hace `test -f`; el único código real que corre
  es el test unitario del grader. Nada del pipeline se ejercita.

## 2. Objetivos

1. Mejor tratamiento de tablas e imágenes (MinerU + embeddings multimodales).
2. Mayor eficiencia de recuperación (FAISS).
3. Sustituibilidad: cada tecnología y cada modelo intercambiables por configuración.
4. Clean architecture real, no nominal.
5. CI que verifique de verdad cada fase del pipeline.
6. Documentación mínima y suficiente.
7. Repositorio centrado en el producto: la investigación desaparece del árbol.

## 3. Decisiones tomadas

| Decisión | Elección | Motivo |
|---|---|---|
| Default del stack en v2.0 | Puertos primero; el default cambia a MinerU/jina/FAISS solo cuando el gate completo pase en verde | La app queda funcional en todo momento y los dos stacks son comparables con el mismo juez |
| Licencia de embeddings | `jinaai/jina-clip-v2` (CC BY-NC) | Uso personal/portfolio; es la mejor torre multilingüe para en/es/ca. Restricción documentada en el README |
| Rust | Fuera de v2.0, issue con criterios de medición | El tiempo está en la inferencia de Ollama y en MinerU, ambos ya nativos. Reescribir el pegamento Python movería lo que no cuesta |
| `research/` | Eliminado de HEAD | TFG defendido. La historia y el tag `v1.0.0-tfg` lo preservan |
| Submódulo `llama.cpp` | Eliminado | Servía para cuantizar los GGUF de los fine-tunes, ya abandonados |
| Modelos fine-tuneados | Retirados de la configuración y borrados de Ollama | Un modelo stock con instrucciones claras rinde igual o mejor. Libera 25 GB |
| Extractor de PDF | MinerU como CLI externo, no como dependencia Python | Ya se invoca por `subprocess`. Sus pins chocan con los del producto |
| Índice vectorial | `faiss-cpu`, `IndexFlatIP` sobre vectores normalizados L2 | Búsqueda exacta suficiente a esta escala; la GPU se reserva al embedder y al LLM |
| Política de fallos | Hard-fail en todos los puertos | Un fallback silencioso hace incomparables dos corridas y ya ocultó que UNIPipe nunca funcionó |
| Corpus | Uno activo a la vez sobre carpeta configurable | Con embeddings multilingües, separar por idioma pierde sentido. Se conservan los PDFs existentes |
| Interfaces | Web + escritorio + CLI interactiva, más entrypoints no interactivos | Todas son producto; el CI necesita los no interactivos de todos modos |
| Gestor de paquetes frontend | `pnpm` | Menor superficie de suplantación de paquetes que npm |
| `embeddinggemma` en Ollama | Se conserva | Es el default de F1. Son 621 MB: irrelevante para el espacio, crítico para que la app arranque |

**Conflicto resuelto:** el spec previo (`2026-06-26-mineru-indexing-design.md`) diseñaba un fallback
en tres niveles (MinerU→pymupdf4llm→pypdf). El spike implementó hard-fail puro. Son incompatibles;
se adopta **hard-fail** y ese spec queda superado.

## 4. Arquitectura objetivo

Cuatro capas con dependencias en un solo sentido: interfaces → aplicación → dominio, y adaptadores
implementando puertos que la aplicación declara. El dominio no importa nada de infraestructura.

```
src/monkeygrab/
  domain/           entidades y reglas puras: Chunk, Document, Hit, GoldCase, grading
  application/      casos de uso: IndexCorpus, Retrieve, Answer, RunEval
  ports/            PdfExtractor, Embedder, VectorStore, LexicalIndex, Reranker, ChatModel
  adapters/
    extraction/     pymupdf.py, mineru.py
    embedding/      ollama.py, jina_clip.py
    vectorstore/    chroma.py, faiss.py
    lexical/        bm25.py
    reranking/      cross_encoder.py
    chat/           ollama.py
  config/           AppConfig inmutable + carga desde entorno y settings
  interfaces/
    cli/            CLI interactiva + comandos no interactivos
    web/            Flask + React
corpus/             PDFs del usuario (antes rag/docs/)
tests/              unit (por capa) + integration + eval (gold cases)
```

**Puertos.** Cada uno es un `Protocol` con una responsabilidad y sin tipos de infraestructura en su
firma. `VectorStore` expone exactamente las cinco operaciones que el código usa hoy (`add`, `query`,
`get` paginado, `get` por ids, `count`); no se usan filtros `where` en ninguna parte, así que un
índice FAISS con metadatos en sidecar cubre el 100% de la superficie.

**Configuración.** `AppConfig` es un objeto inmutable construido una vez desde entorno y
`settings.json`, e **inyectado** en los casos de uso. Cambiar configuración en caliente crea una
instancia nueva y reconstruye el grafo de dependencias afectado, en lugar de mutar globales. Esto
elimina por construcción el bug de los argumentos por defecto y colapsa la triple fuente de verdad:
`settings.json` persiste, `AppConfig` es la verdad en memoria, y el frontend deja de mantener copia
autoritativa.

**Fachada de compatibilidad (temporal).** Durante F1–F3, `rag/chat_pdfs.py` se conserva como shim
que reexporta desde la capa nueva, incluidos los tres símbolos privados que consume la web. Web,
CLI y tests siguen funcionando sin modificarse. El shim se elimina en F4, cuando las interfaces
pasan a consumir los casos de uso directamente.

## 5. Fases

Cada fase es una rama, un PR y un merge independiente. Ninguna deja la app inutilizable.

### F0 — Limpieza y saneamiento · `chore/repo-cleanup-v2`

Sin cambios en código productivo.

- Tag anotado `v1.0.0-tfg` en el último commit que contiene `research/`, antes de borrar.
- **Rescata los tests del producto antes de borrar.** `research/tests/core/` contiene nueve tests
  que ejercitan `rag.*` (BM25 y su caché, RAG sobre imágenes, streaming sin *thinking*, rutas web,
  TTY de la CLI): se mueven a `tests/` y `pytest.ini` se reapunta. Los cuatro de
  `research/tests/evaluation/` dependen de `research.evaluation._lib` y se van con la investigación.
- Elimina `research/`, el submódulo `llama.cpp`, `llama-bin/`, `dist/`, `repeticion*/`,
  `pipeline/output/`, `rag/tmp_mineru_out*/`, `rag/web/zip/`, cachés.
- **No** pinea aún el stack multimodal. El plan original era declararlo aquí copiando las versiones
  de `.venv-mineru`, pero ese venv tiene `torch` y `transformers` **más antiguos** que el entorno
  donde el producto se ejecuta de verdad (2.6.0+cu124 y 4.57.6 frente a 2.11.0+cu128 y 5.1.0): esos
  pines degradarían silenciosamente un entorno que funciona. Las versiones se resuelven en F3 contra
  el runtime real. MinerU queda fuera en cualquier caso, documentado como CLI externo.
- Reduce documentación: un `README.md` raíz corto y un README por directorio cuando aporte.
  `rag/README.md` (895 líneas) y `ENGINE_MAP.md` (396) se sustituyen por documentación derivada de
  los puertos en F1.
- `.gitignore` unificado.

**Aceptación:** `pytest` pasa igual que antes del cambio; la web y la CLI arrancan e indexan;
`git log v1.0.0-tfg -- research/` sigue mostrando la historia.

### F1 — Núcleo hexagonal con adaptadores actuales · `refactor/clean-architecture-core`

Migración estructural **sin cambio de comportamiento observable**.

- Tests de caracterización primero: capturan el comportamiento actual de recuperación, contexto y
  generación con modelos stock antes de mover nada.
- Dominio, casos de uso y puertos. Adaptadores para lo que ya existe: pymupdf, Chroma, Ollama
  (embeddings y chat), BM25, CrossEncoder.
- `AppConfig` inmutable inyectada; se elimina `get_runtime()` y el acceso `cfg.<librería>`.
- Hard-fail en todos los puertos: desaparecen los fallbacks de reranker, de pypdf y las cinco
  degradaciones de RECOMP. Un `PDF_EXTRACTOR` desconocido es un error de arranque, no un default.
- `chat_pdfs.py` queda como shim de compatibilidad.

**Aceptación:** los tests de caracterización pasan sin modificarse; ningún módulo de
`domain/` o `application/` importa `chromadb`, `ollama`, `fitz` ni `torch` (verificado por test de
dependencias); la web y la CLI funcionan sin cambios en su código.

Cierra #6 y #8.

### F2 — CI verificable · `ci/pipeline-gates`

- **Gate rápido** (GitHub, cada PR): dominio y aplicación contra dobles de test, grader
  determinista, lint y comprobación de tipos. Sin GPU, sin red, sin modelos.
- **Gate completo** (local o self-hosted, obligatorio antes de merge): pipeline real con Ollama,
  MinerU y FAISS sobre los gold cases, con ratchet de `baseline_min_pass_rate`.
- El job `self-improve-gate-docs`, que hoy solo imprime texto, se elimina.
- Los gold cases se promueven de `rag/experiments/eval/` a `tests/eval/` con el grader como código
  de producción.
- Modelos de evaluación: `gemma4:e2b` y `qwen3.5:0.8b`, stock, sin fine-tunes.

**Aceptación:** el gate rápido falla si se rompe una regla de dominio; el gate completo falla si
`pass_rate` cae por debajo del baseline; ningún job afirma cubrir lo que no ejecuta.

Cierra #7 y #9.

### F3 — Adaptadores del stack nuevo · `feat/mineru-faiss-adapters`

- `MinerUExtractor` sobre el CLI, hard-fail.
- `JinaClipEmbedder`: texto e imagen en un espacio compartido, `truncate_dim=512`, imagen combinada
  con su caption por suma vectorial renormalizada. Requiere CUDA y lo declara: en CPU aborta, no
  degrada a 100 s/documento.
- `FaissVectorStore`: `IndexFlatIP`, persistencia `index.faiss` + `meta.jsonl` + `version.txt`.
- Las tablas se indexan como texto HTML, no como imágenes.
- **Se generalizan las heurísticas del spike.** El chunking del spike inyecta alias léxicos
  cableados al paper "Attention" (`d_model`, `N=6`); eso no puede ir a producción: se generaliza a
  una normalización de LaTeX independiente del documento, o se elimina.
- Comparación A/B: mismos gold cases, ambos stacks, resultado registrado.

**Aceptación:** una consulta de texto sobre una figura devuelve `kind=image` en el top-k; una sobre
una tabla de resultados devuelve `kind=table`; la consulta en castellano recupera la figura de
arquitectura; el A/B queda escrito. El default cambia solo si el stack nuevo gana o empata.

Cierra #3, #4 y #5.

### F4 — Interfaces y corpus único · `refactor/single-corpus-ui`

- Un corpus activo sobre carpeta configurable; desaparece la identidad fija en/es/ca. Los PDFs
  actuales se conservan.
- Las interfaces pasan a consumir casos de uso; se elimina el shim `chat_pdfs.py`.
- Fuente de verdad única de configuración: el frontend deja de mantener copia autoritativa.
- i18n consolidado: hoy hay tres tablas divergentes (`cli/strings.py`, listas de `cli/commands.py`,
  diccionarios en `App.tsx`). Pasan a una sola.
- `App.tsx` (2022 líneas, un componente) se descompone por vista.
- Frontend migrado a `pnpm`; packaging actualizado al nuevo layout de corpus.

**Aceptación:** la web y el `.exe` empaquetado indexan y responden; cambiar un parámetro en el panel
afecta de verdad al pipeline (el bug de los argumentos por defecto está cubierto por test).

### F5 — Release · `release/v2.0`

README raíz corto, CLAUDE.md reescrito para la arquitectura nueva, tag `v2.0.0`, cierre de issues.

## 6. Mapeo de issues

| Issue | Fase | Nota |
|---|---|---|
| #3 MinerU 3.x CLI | F3 | Hard-fail, sin fallback a pymupdf |
| #4 FAISS | F3 | Puerto definido en F1 |
| #5 jina-clip-v2 | F3 | CC BY-NC aceptada y documentada |
| #6 Hard-fail | F1 | Aplicada a todos los puertos, no solo al extractor |
| #7 Modelos stock | F2 | Fine-tunes retirados en F0 |
| #8 Clean architecture | F1 | Base de todo lo demás |
| #9 Loop de auto-mejora | F2 | Gate completo real, no documental |
| #2 CLI profesional | — | Excluido por decisión del usuario |

## 7. Integridad de la verificación

- Ningún assert se debilita para que una fase pase.
- Un job de CI que no ejercita un camino no se presenta como cobertura de ese camino.
- El gate completo requiere GPU y Ollama locales: es un requisito declarado, no una excusa. Si no
  puede correr, se dice, no se marca verde.
- Los tests de caracterización de F1 se escriben **antes** de mover código, y no se editan durante
  la migración: si cambian, el comportamiento cambió.

### 7.1 Corpus de verificación con papers reales

El juez del sistema no puede ser un mock. Se construye un corpus de papers públicos de arXiv con
preguntas cuya respuesta es verificable en el texto, y se usa como criterio de aceptación en cada
fase que toque recuperación o generación.

- **Fuentes:** papers de arXiv identificados por su ID, descargados por un script idempotente a un
  directorio cacheado y no versionado. El ID fija la versión, así que la corrida es reproducible sin
  meter los PDFs en git.
- **Cobertura buscada:** hechos numéricos (valores en tablas y formulas), definiciones conceptuales,
  recuperación de figuras y de tablas, y consultas en castellano sobre documentos en inglés. La
  mezcla es deliberada: es donde el stack antiguo (caption por VLM) y el nuevo (embedding
  multimodal) deben diferenciarse.
- **Papers de fuera del conjunto de desarrollo.** Las heurísticas del spike se calibraron sobre
  "Attention is all you need"; parte de los papers se reservan como conjunto ciego, para detectar
  sobreajuste al documento en lugar de calidad de recuperación.
- **Respuesta conocida, no respuesta plausible.** Cada caso lleva los literales aceptables
  verificados a mano contra el PDF. Un caso cuya respuesta no se puede comprobar en el texto no
  entra.

### 7.2 Autosuficiencia del gate completo

El gate completo se ejecuta con un solo comando y se autoabastece: descarga los papers que falten,
reutiliza la extracción de MinerU cacheada e invalida la caché por versión de índice, comprueba que
Ollama responde y que los modelos necesarios están presentes, y falla con un mensaje accionable
cuando un prerrequisito no está. No requiere pasos manuales previos ni rutas absolutas de una
máquina concreta — el spike actual sí las tiene cableadas y eso desaparece.

### 7.3 Autorización de merge

El usuario autoriza merge a `main` de cada fase cuyo CI esté verde, incluida F0 con la eliminación
de `research/` (que además queda preservada en el tag `v1.0.0-tfg`). Verde significa: gate rápido en
verde **y** gate completo en verde sobre el corpus de papers, no solo el primero.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| La web depende de símbolos privados | El shim los reexporta hasta F4 |
| `torch` cu124 y los pins de MinerU chocan | MinerU permanece CLI externo, fuera del entorno |
| jina-clip exige CUDA y el CI de GitHub no la tiene | Dos gates; el completo es self-hosted/local |
| Reindexado obligatorio al cambiar de embedder | Los paths de índice ya se derivan por slug de modelo; se mantiene |
| Migración larga con la app rota | Strangler fig: cada fase deja la app funcional |
| Las heurísticas del spike no generalizan | F3 las generaliza o las elimina, con gold cases de otros documentos como juez |

## 9. Resultado del A/B

Medido el 2026-07-27 sobre los 51 casos gold, mismo juez, mismo generador
(`gemma4:e2b`) y el mismo camino de recuperación en ambos lados. Una sola
variable: el stack.

| Tipo de caso | `pymupdf-ollama-chroma` | `mineru-jina_clip-faiss` |
|---|---|---|
| Recuperación de tablas | 0/5 (0%) | **5/5 (100%)** |
| Recuperación de figuras | 5/10 (50%) | 6/10 (60%) |
| Conceptos factuales | 9/11 (82%) | 10/11 (91%) |
| Números factuales | 21/25 (84%) | 21/25 (84%) |
| **Total** | **35/51 (68,6%)** | **42/51 (82,3%)** |
| Duración | 24,2 min | 28,6 min |

Gana en todas las categorías y no pierde en ninguna, **cargando un handicap**: el
índice multimodal se construyó sin contextual retrieval, que el actual sí tiene.
Igualarlo exigiría reconstruir un índice con una llamada al LLM por chunk, así
que se declara en vez de ocultarse — y hace que el resultado sea conservador.

Tres de las cinco tablas recuperadas son del conjunto ciego (ResNet, BERT, ViT),
que es lo que descarta el sobreajuste a los papers de desarrollo.

**Por qué las tablas pasan de ninguna a todas.** No es ajuste fino: aplanar un
PDF no degrada una tabla, la destruye como objeto. Sus números se dispersan en
prosa y no existe ningún fragmento de tipo tabla que recuperar. Hicieron falta
las dos mitades — MinerU emitiendo HTML, y el indexado marcando como tabla el
fragmento que lo contiene. Solo con la primera, el resultado no habría cambiado.

**El default no cambia con este resultado.** El stack nuevo exige el venv aislado
y una GPU con CUDA; ponerlo por omisión dejaría un clon recién hecho con una app
que no arranca. La medición dice que gana; la instalación dice que todavía no
puede ser el default. Queda seleccionable y documentado.

**Sobre la eficiencia (objetivo 2).** La recuperación es más rápida con el stack
nuevo: 3-5 s por consulta frente a 6-7 s. Las corridas anteriores que sugerían lo
contrario medían un apaño —reranker en CPU y un worker reiniciado 51 veces—, no
el stack. La corrida completa es algo más lenta por el arranque del worker y por
generar sobre más fragmentos.

## 10. Registro de decisiones tomadas durante la ejecución

Decisiones que no estaban en el plan original y se resolvieron sobre la marcha, con su motivo. Se
anotan aquí porque varias contradicen lo que el propio documento decía antes.

| Decisión | Motivo |
|---|---|
| Los pines del stack multimodal **no** se declaran hasta tener el entorno real delante | El venv del spike tiene `torch` y `transformers` **anteriores** a los del runtime del producto; declararlos habría degradado un entorno que funciona |
| `media/` se conserva pese a no estar referenciado | Son las grabaciones originales de las demos, no regenerables, y 29 MB frente a 43 GB liberados no justifican el riesgo |
| `test_image_rag.py` se elimina en vez de repararse | No definía ninguna función de test: aportaba cero verificación y costaba un import de chromadb por corrida |
| Los dos sondeos de Ollama salen de `tests/` a `tools/diagnostics/` | Igual: parecían cobertura sin serlo. Un gate honesto no cuenta ficheros que no verifican nada |
| El lint se limita a pyflakes y errores de sintaxis | Un linter que reporta cientos de hallazgos cosméticos sobre código existente se silencia, y un linter silenciado no detecta nada |
| Los tests de adaptadores corren en el gate del motor, no en el rápido | Un adaptador **es** infraestructura: importa su librería aunque el objeto esté doblado. Verificado contra un intérprete sin esas librerías |
| La comparación de defaults contra el motor se salta si el motor no se puede importar | "Este entorno no tiene chromadb" y "los defaults divergieron" son hechos distintos; confundirlos entrena a ignorar el fallo que importa |
| `RERANKER_QUALITY` pasa a validar su valor | Hoy cualquier cadena distinta de `quality` significa `fast` en silencio. Coherente con la política hard-fail, pero **es** un cambio de comportamiento |
| La extracción de imágenes conserva su degradación silenciosa | Es enriquecimiento opcional gobernado por un flag; abortar la indexación entera por una imagen corrupta sería peor. Excepción documentada en el puerto |
| `pnpm` se ancla con `packageManager` y se valida en CI | Cambiar de gestor no sirve de nada mientras nada impida volver a `npm` y regenerar el lockfile |
| El corpus de casos endurece los literales de un solo dígito | `"6"` casaba con `"Figure 6"` en una respuesta que decía 12 capas. Se prioriza precisión sobre acreditar respuestas telegráficas |
| La recuperación converge sobre `Retrieve` antes que la generación | El gate medía `Retrieve` mientras CLI y web ejecutaban el pipeline inline: el criterio de aceptación certificaba código que ningún usuario corría. La generación no tenía esa divergencia, porque el gate ya llama a `generar_respuesta_silenciosa` |
| `Answer` se mantiene sin cablear en vez de borrarse | Cablearlo mueve el código que produce respuestas y exige el gate completo con GPU para firmarse. Borrarlo tiraría trabajo verificado que es el destino de la fase siguiente |
| `Retrieve` recupera la variante de consulta por palabras clave | Se había omitido al portarlo. Sin ella, toda pregunta por debajo del umbral de descomposición perdía en silencio su segundo ángulo de búsqueda |
| El desempate de palabras clave pasa a ser alfabético | Las de igual prioridad salían en orden de iteración de un `set`, derivado del hash aleatorizado por proceso: la consulta de respaldo, y con ella el resultado de la recuperación, variaba entre ejecuciones de la misma pregunta |
| Los banners `MODULE MAP` y las cabeceras numeradas desaparecen | Duplicaban la estructura del módulo sin que nada lo verificase, y ya habían quedado obsoletos en varios ficheros |

## 11. Fuera de alcance

Rust (issue aparte), CLI profesional (#2), reescritura del frontend más allá de descomponer
`App.tsx`, ANN aproximado (IVF/HNSW) mientras la búsqueda exacta baste, y recuperar la capa de
investigación.
