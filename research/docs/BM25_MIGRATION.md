# Migración de la búsqueda léxica a Okapi BM25

> Fecha: 2026-05-22 · Afecta a la **Etapa 2 (Recuperación híbrida)** del pipeline RAG.
> Documenta el comportamiento **antes vs. ahora** cuando `USAR_BUSQUEDA_HIBRIDA = True`.

---

## 1. Resumen

La capa léxica del pipeline tenía **dos fases redundantes** que hacían coincidencia
de texto sobre los mismos chunks:

1. **Búsqueda por keywords** (`busqueda_por_keywords`): filtro `$contains` de
   ChromaDB + fusión RRF por la *posición de devolución* de Chroma.
2. **Búsqueda exhaustiva** (`busqueda_exhaustiva_texto`): escaneo de toda la
   colección contando términos críticos.

Ninguna producía una puntuación de relevancia léxica formal. Se han **sustituido
por una única búsqueda Okapi BM25** (`busqueda_lexica_bm25`), que sí genera un
ranking de relevancia real y citable, fusionado con la búsqueda semántica por RRF.

**Consecuencia neta:** el pipeline pasa de **3 vías** (semántica + keywords +
exhaustiva) a **2 vías** (semántica + BM25).

---

## 2. Diagrama de la Etapa 2

### Antes

```
pregunta
   │
   ├── A) Query decomposition (LLM)         → variantes de query
   ├── B) Búsqueda SEMÁNTICA (vector)        → lista 1  (RRF por distancia)
   ├── C) Búsqueda por KEYWORDS ($contains)  → lista 2  (RRF por orden Chroma)
   └── D) Búsqueda EXHAUSTIVA (scan total)   → lista 3  (0.3 · nº términos)
                       │
                       ▼
        E) Fusión: score_final = PESO_SEMANTICO_RRF·sem + PESO_BM25_RRF·kw
```

### Ahora

```
pregunta
   │
   ├── A) Query decomposition (LLM)         → variantes de query
   ├── B) Búsqueda SEMÁNTICA (vector)        → lista 1  (RRF por distancia)
   └── C) Búsqueda LÉXICA BM25               → lista 2  (RRF por rango BM25)
                       │
                       ▼
        D) Fusión: score_final = PESO_SEMANTICO_RRF·sem + PESO_BM25_RRF·kw
```

---

## 3. Comparativa fase por fase

### 3.1 Extracción de keywords — `extraer_keywords()`

| | Antes | Ahora |
|---|---|---|
| ¿Se ejecuta? | Sí | Sí (**sin cambios** en la función) |
| ¿Para qué? | **Dirigía la búsqueda léxica** (`$contains` y términos críticos) | Solo alimenta una **variante de query semántica** (`query_kw`) y el campo `keywords` de métricas/debug |
| Salida | Acrónimos, bigramas, términos parentetizados y técnicos | Igual |

> Punto clave: las keywords "de siempre" se siguen extrayendo idénticas, pero **ya
> no controlan la recuperación léxica**. BM25 usa su propia tokenización.

### 3.2 Tokenización de la consulta y el corpus

| | Antes (`extraer_keywords` + variantes) | Ahora (`_tokenizar_bm25`) |
|---|---|---|
| Unidad | lista curada con n-gramas | unigramas |
| Bigramas / multipalabra | sí | no |
| Mayúsculas | conserva acrónimos (`RAGBENCH`) | todo a minúsculas |
| Parentizados | sí | no (se parten) |
| Variantes por término | sí (lower, capitalize, guion→espacio) | no |
| Filtro | stopwords + blacklist + dedup | stopwords + descarta tokens <3 sin dígitos |
| Aplicado a | solo la query | **query y corpus por igual** (requisito de BM25) |

`_tokenizar_bm25`: minúsculas → split por límites no alfanuméricos (Unicode,
conserva acentos es/ca) → descarta `STOPWORDS` → descarta tokens de menos de 3
caracteres salvo que contengan dígitos (conserva identificadores/métricas como
`q4`, `f1`).

### 3.3 Recuperación léxica

#### Antes — dos funciones

**`busqueda_por_keywords(pregunta, collection, n_results=N_RESULTADOS_KEYWORD)`**
- Para cada keyword (hasta 20) y sus variantes (lower/capitalize/guion):
  - `collection.get(where_document={"$contains": variante}, limit=40)`.
  - `$contains` es un **filtro** de subcadena, **no** un ranking. Sensible a
    mayúsculas (de ahí las variantes).
  - Se acumulan chunks no vistos, con `distancia=0.5` y `keyword_match`.
- El orden de la lista resultante es el orden en que ChromaDB devuelve los chunks
  filtrados + el orden de iteración de keywords. **No es relevancia.**

**`busqueda_exhaustiva_texto(terminos_criticos, collection, max_results=30)`**
- `terminos_criticos = _filtrar_terminos_criticos(keywords[:12])`: conserva
  multipalabra, capitalizados y acrónimos; descarta una blacklist genérica.
- Escaneo de **toda la colección** en lotes de 100; por chunk cuenta cuántos
  términos críticos aparecen (substring en minúsculas) → `num_matches`.
- Ordena por `num_matches` desc, devuelve hasta 30.

#### Ahora — una función

**`busqueda_lexica_bm25(pregunta, collection, top_n=N_RESULTADOS_KEYWORD)`**
- Requiere `rank-bm25` (`BM25_AVAILABLE`); si falta, devuelve `[]` y el pipeline
  opera solo con la vía semántica.
- `query_tokens = _tokenizar_bm25(pregunta)`; si no hay tokens, devuelve `[]`.
- Escaneo de **toda la colección** en lotes de 100; tokeniza cada chunk con
  `_tokenizar_bm25`.
- Construye el índice e infiere puntuaciones:
  ```python
  bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)   # k1=1.5, b=0.75
  scores = bm25.get_scores(query_tokens)
  ```
- Ordena por score BM25 desc, **descarta score ≤ 0**, devuelve los `top_n=40`
  mejores, cada uno con `bm25_score`.
- El índice BM25 **se reconstruye en cada consulta** (sin cache).

### 3.4 Puntuación léxica (cómo se calcula el score)

#### Antes
`score_keyword` mezclaba **dos escalas distintas**:
- De keywords: `score_keyword += 1 / (idx + RRF_K)` donde `idx` = posición en la
  lista devuelta por Chroma (no relevancia). Aporte máximo ≈ `1/21 ≈ 0.048`.
- De exhaustiva: `score_keyword += 0.3 · num_matches` (additivo, sin RRF). Con 2–3
  términos podía valer `0.6–0.9`, **dominando** numéricamente al término RRF.

#### Ahora
`score_keyword` es **RRF puro sobre un ranking de relevancia real**:
```python
score_keyword += 1 / (rank_bm25 + RRF_K)   # rank_bm25 = posición por score BM25
```
Una sola escala, coherente con el lado semántico. El `bm25_score` crudo solo se usa
para **ordenar y filtrar** (`> 0`); no entra directamente en la mezcla.

**Fórmula BM25 (Robertson & Zaragoza, 2009)** que da ese `bm25_score`:
```
score(D,Q) = Σ_t∈Q  IDF(t) · [ f(t,D)·(k1+1) ] / [ f(t,D) + k1·(1 − b + b·|D|/avgdl) ]
```
- `f(t,D)`: frecuencia del término en el chunk.
- `IDF(t)`: rareza del término en la colección (más peso a términos discriminantes).
- `|D|`, `avgdl`: longitud del chunk y longitud media → normalización por longitud.
- `k1=1.5` (`BM25_K1`): saturación de la frecuencia (repetir aporta cada vez menos).
- `b=0.75` (`BM25_B`): penalización por longitud del chunk.

### 3.5 Fusión final

| | Antes | Ahora |
|---|---|---|
| Listas fusionadas | 3 (semántica, keywords, exhaustiva) | 2 (semántica, BM25) |
| Fórmula | `score_final = PESO_SEMANTICO_RRF·score_semantic + PESO_BM25_RRF·score_keyword` | **idéntica** |
| `RRF_K` | 20 | 20 (sin cambios) |
| Umbral de paso | `UMBRAL_RELEVANCIA = 0.50` | sin cambios |

La búsqueda semántica (incluida la acumulación RRF sobre variantes de query) **no
ha cambiado**. Solo cambia el segundo sumando (`score_keyword`).

---

## 4. Flags y parámetros

| Elemento | Antes | Ahora |
|---|---|---|
| `USAR_BUSQUEDA_HIBRIDA` | activa keyword `$contains` | activa **BM25** |
| `USAR_BUSQUEDA_EXHAUSTIVA` | activa la exhaustiva | **eliminado** por completo (de `chat_pdfs.py`, `PIPELINE_RUNTIME_FLAGS` y de la capa de evaluación) |
| `RRF_K` | 20 | 20 |
| Pesos fusión | 0.55 / 0.45 | 0.55 / 0.45 por defecto (`RAG_PESO_SEMANTICO_RRF`, `RAG_PESO_BM25_RRF`) |
| `N_RESULTADOS_KEYWORD` | máximo de resultados **por variante** de keyword (`limit` de `collection.get`) | **top-N total** de fragmentos BM25 devueltos |
| `BM25_K1` | — | **1.5** (nuevo) |
| `BM25_B` | — | **0.75** (nuevo) |
| `BM25_AVAILABLE` | — | nuevo flag de disponibilidad de `rank-bm25` |

---

## 5. Funciones y dependencias

**Eliminadas:**
- `busqueda_por_keywords` (`rag/engine/lexical.py`)
- `busqueda_exhaustiva_texto` (`rag/engine/lexical.py`)
- `_filtrar_terminos_criticos` (`rag/engine/reranking.py`) — quedaba muerta

**Añadidas:**
- `_tokenizar_bm25` (`rag/engine/lexical.py`)
- `busqueda_lexica_bm25` (`rag/engine/lexical.py`)

**Conservadas sin cambios funcionales:** `extraer_keywords`, `realizar_busqueda_hibrida`
(orquestador, con la fusión reescrita internamente).

**Dependencia nueva:** `rank-bm25>=0.2.2` en `rag/requirements.txt` (Python puro +
numpy; sin GPU).

---

## 6. Métricas y archivos de debug

| Campo en `metricas` | Antes | Ahora |
|---|---|---|
| `fase_keywords` | `keywords_totales`, `keywords_encontradas`, `resultados_totales`, `errores` | `disponible`, `documentos_indexados`, `terminos_query`, `resultados_totales`, `mejor_score` |
| `fase_exhaustiva` | presente | **eliminado** |
| `terminos_criticos` | presente | **eliminado** |
| `keywords` | presente (de `extraer_keywords`) | presente (sin cambios) |

Cambios en el volcado de `guardar_debug_rag` (`rag/engine/debug.py`):
- Eliminado el bloque "Critical terms (exhaustive search)".
- Eliminada la línea "Exhaustive Search: YES/NO".
- "Keyword metrics" → **"BM25 metrics"** (docs indexados, términos de query,
  resultados, mejor score).
- "Hybrid Search (keywords)" → **"Hybrid Search (BM25)"**.
- Por fragmento: "Matched keywords" → **"Lexical match: BM25"** (el campo `matches`
  ahora solo contiene el marcador `BM25`).

---

## 7. Interfaz de usuario

- **CLI** (`rag/cli/`): el flag `exhaustive` desaparece de `/stats` y de las filas
  compactas de flags. La búsqueda híbrida sigue mostrándose (`hybrid`).
- **Web** (`rag/web/`): el antiguo toggle de "Búsqueda exhaustiva" se ha
  **reconvertido a "Indexado de imágenes"** (`USAR_EMBEDDINGS_IMAGEN`) y movido a la
  sección "1. Indexación" (es/en/ca). Matiz: ese flag es de indexado, así que solo
  surte efecto en el **próximo reindexado**, no en la recuperación en vivo. Requiere
  `npm run build` para regenerar `dist/`.

---

## 8. Lo que NO ha cambiado

- Búsqueda **semántica** (vector, variantes de query, acumulación RRF por distancia).
- **Query decomposition** con LLM.
- **Reranker** Cross-Encoder, **expansión de vecinos**, **RECOMP**, **generación**.
- **Indexación** (chunking, contextual retrieval, embeddings, OCR de imágenes).
- Fórmula y pesos de fusión, `RRF_K`, umbrales.

---

## 9. Implicaciones operativas

- **No requiere reindexar.** BM25 opera sobre el texto de los chunks ya presentes en
  ChromaDB (`collection.get`); funciona con el `vector_db/` actual tal cual.
- **No descarga ningún modelo.** BM25 es recuperación dispersa (CPU pura): sin
  embeddings, sin Ollama, sin GPU. Única dependencia: el paquete `rank-bm25`.
- **Coste por consulta** comparable o menor: antes había múltiples `collection.get`
  por keyword + un escaneo completo (exhaustiva); ahora hay un único escaneo completo
  + construcción del índice BM25.
- **Evaluación del TFG:** las métricas publicadas (Qwen3, Phi-4) se obtuvieron con el
  pipeline **antiguo** (keyword + exhaustiva). Producción ya no coincide con lo
  evaluado; si se vuelve a evaluar, regenerar con BM25.
- **Capa de evaluación alineada:** `USAR_BUSQUEDA_EXHAUSTIVA` se ha eliminado de
  `research/evaluation/_lib/pipeline_flags.py` (`BASELINE_PIPELINE_FLAGS` y el arm
  `no_exhaustive_search`). La suite `ablation` pasa de 9 a 8 variantes.
- **Consecuencia en checkpoints congelados:** los manifiestos `pipeline_flags` de
  `baseline_all_on.json` / `all_off.json` aún contienen `USAR_BUSQUEDA_EXHAUSTIVA`.
  Como `get_pipeline_flags()` ya no la devuelve, `checkpoint_pipeline_flags_match`
  dará `False` y una re-ejecución de `infer.py` contra esos run dirs **no reutilizará**
  las generaciones cacheadas (re-inferiría). Los JSON se conservan como registro
  histórico; no se re-evalúa.

---

## 10. Referencias

- **BM25** — Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond*. Foundations and Trends in IR, 3(4), 333–389.
  https://doi.org/10.1561/1500000019
- **RRF** — Cormack, G., Clarke, C. & Büttcher, S. (2009). *Reciprocal Rank Fusion
  outranks Condorcet and individual Rank Learning Methods*. SIGIR 2009.
  https://doi.org/10.1145/1571941.1572114

---

## 11. Documento a citar, librería y fórmulas

### 11.1 Documentos a citar

- **BM25** (la puntuación léxica): Robertson, S. & Zaragoza, H. (2009). *The
  Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in
  Information Retrieval, 3(4), 333–389.
  - DOI: https://doi.org/10.1561/1500000019
  - PDF abierto: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
- **RRF** (la fusión léxica+semántica): Cormack, G., Clarke, C. & Büttcher, S. (2009).
  *Reciprocal Rank Fusion outranks Condorcet and individual Rank Learning Methods*.
  SIGIR 2009.
  - DOI: https://doi.org/10.1145/1571941.1572114
  - PDF abierto: https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf

### 11.2 Librería usada

- **`rank-bm25` 0.2.2**, clase `BM25Okapi`. Recuperación dispersa en Python puro
  (solo depende de numpy). CPU; sin modelo ni reindexado.
  - PyPI: https://pypi.org/project/rank-bm25/
  - Repositorio: https://github.com/dorianbrown/rank_bm25
  - Declarada en `rag/requirements.txt` (`rank-bm25>=0.2.2`).

### 11.3 Fórmulas de puntuación

**(a) BM25 por fragmento** (lo que devuelve `BM25Okapi.get_scores`):

```
score(D,Q) = Σ_{t∈Q}  IDF(t) · [ f(t,D)·(k1+1) ] / [ f(t,D) + k1·(1 − b + b·|D|/avgdl) ]
```
- `f(t,D)`: frecuencia del término `t` en el chunk `D`.
- `|D|`: longitud del chunk en tokens; `avgdl`: longitud media de los chunks.
- `k1 = BM25_K1 = 1.5` (saturación de frecuencia); `b = BM25_B = 0.75` (normalización por longitud).
- IDF en `BM25Okapi` (Robertson–Spärck-Jones):
  `IDF(t) = ln( (N − n(t) + 0.5) / (n(t) + 0.5) )`, con `N` = nº de chunks y `n(t)` =
  chunks que contienen `t`. Los IDF negativos (términos casi omnipresentes) se
  reemplazan por `ε · IDF_medio` con `ε = 0.25`, para no restar peso.

**(b) Fusión RRF** (convierte cada ranking en una contribución; Cormack et al. 2009):
```
score_keyword(d)  += 1 / (rank_BM25(d)      + RRF_K)
score_semantic(d) += 1 / (rank_semantico(d) + RRF_K)        # acumulado sobre las variantes de query
```
con `RRF_K = 20` por defecto. El `bm25_score` crudo solo ordena y filtra (`> 0`); en la mezcla
entra su **rango**, no su valor.

**(c) Puntuación final fusionada:**
```
score_final(d) = PESO_SEMANTICO_RRF · score_semantic(d) + PESO_BM25_RRF · score_keyword(d)
```
Defaults actuales: `PESO_SEMANTICO_RRF = 0.55`, `PESO_BM25_RRF = 0.45`.

---

## 12. Traza real de una consulta (de la pregunta a la respuesta)

Ejemplo ejecutado en la Web el 2026-05-22 (dump
`rag/debug_rag/20260522_204733_What_is_the_key_insight_of_the_Direct_Pr.txt`), corpus
EN (un único PDF indexado: `2305.18290v3.pdf`, el paper de DPO), modelo `gemma4`.

**Pregunta del usuario:**
> *What is the key insight of the Direct Preference Optimization (DPO) approach that
> allows it to transform a loss function over reward functions into a loss function
> over policies, thereby bypassing the need for a standalone reward model and an RL
> training loop?*

**Paso 0 — Entrada.** La Web envía `POST /api/rag` → `rag/web/app.py` delega en el
facade `rag.chat_pdfs` (`rag_engine`) → `realizar_busqueda_hibrida(pregunta, collection)`.
Configuración activa del dump: Contextual Retrieval, Query Decomposition, Hybrid
Search (BM25), Reranker, Expand Context, Optimize Context, RECOMP Synthesis = todos YES.

**Paso 1 — Descomposición de la consulta** (`USAR_LLM_QUERY_DECOMPOSITION`, la pregunta
supera 60 caracteres). `MODELO_CHAT` (`think=False`) genera 3 sub-queries:
1. `DPO loss function policy transformation`
2. `Direct Preference Optimization bypass reward model`
3. `DPO methodology policy learning without RL loop`

**Paso 2 — Variantes de query para la semántica (5).** Se combinan: (1) la pregunta
original, (2) una versión recortada por palabras clave
(*"What is the insight of the Direct Preference Optimization (DPO) approach allows to
transform function reward functions function policies, thereby"*) y (3–5) las tres
sub-queries.

**Paso 3 — Keywords** (`extraer_keywords`): 14 términos (`RL, DPO, need, insight,
transform, reward model, loss function, training loop, policies thereby, reward
functions, standalone reward, thereby bypassing, direct preference, preference
optimization`). **Importante:** alimentan la variante semántica y el dump, **no** la
búsqueda BM25.

**Paso 4 — Búsqueda semántica + RRF.** Para cada una de las 5 variantes: prefijo de
embedding → `MODELO_EMBEDDING` → `collection.query(n_results=80)` → acumulación RRF por
distancia en `score_semantic`. Resultado: **69 fragmentos únicos**
(`fase_semantica.fragmentos_unicos = 69`).

**Paso 5 — Búsqueda léxica BM25** (`USAR_BUSQUEDA_HIBRIDA`, `BM25_AVAILABLE = true`):
1. `_tokenizar_bm25(pregunta)` → **24 tokens de query**.
2. Escaneo de toda la colección por lotes → **69 chunks indexados** → tokenizados.
3. `BM25Okapi(corpus, k1=1.5, b=0.75)`; `get_scores(query_tokens)`.
4. Top-N por score positivo: **40 resultados**, **mejor score 40.72**.
5. Fusión por RRF del rango BM25 en `score_keyword`; se marca `matches = ['BM25']`.
   (`fase_keywords`: `disponible=true, documentos_indexados=69, terminos_query=24,
   resultados_totales=40, mejor_score=40.718`).

**Paso 6 — Fusión.** `score_final = PESO_SEMANTICO_RRF·score_semantic + PESO_BM25_RRF·score_keyword` sobre los
**69 candidatos** (`candidatos_fusion = 69`), ordenados de mayor a menor.

**Paso 7 — Reranking** (`USAR_RERANKER`, CrossEncoder tier `quality`, `cuda`): 69
candidatos de entrada → **top 15** de salida en **3.35 s**; `score_final` se reemplaza
por el `score_reranker`.

**Paso 8 — Umbral de relevancia** (`UMBRAL_SCORE_RERANKER = 0.65`): de los 15, **8**
superan el umbral, con scores reranker `0.998, 0.979, 0.974, 0.962, 0.900, 0.896,
0.889, 0.871` — todos del PDF `2305.18290v3.pdf` (páginas 5, 3, 5, 7, 2, 2, 11, 6) y
marcados `Lexical match: BM25`.

**Paso 9 — Expansión de vecinos** (`EXPANDIR_CONTEXTO`, `N_TOP_PARA_EXPANSION = 3`): se
añaden chunks adyacentes a los mejores → fragmento 9 (página 4, `Final score 0.0`,
`Reranker N/A`, sin `Lexical match` = vecino). **Total: 9 fragmentos** al contexto.

**Paso 10 — Optimización de contexto** (`USAR_OPTIMIZACION_CONTEXTO`): limpieza de
artefactos de extracción del PDF.

**Paso 11 — Síntesis RECOMP** (`USAR_RECOMP_SYNTHESIS`, `MODELO_RECOMP`): en lugar de
enviar los chunks crudos, se condensan en hechos relevantes (`## Facts relevant to the
question`), p. ej.: el *insight* de DPO es un mapeo analítico de funciones de recompensa
a políticas óptimas; el cambio de variable evita ajustar un reward model explícito; la
red de política representa el LM y la recompensa implícita; objetivo equivalente a
Bradley-Terry con `r_*(x,y)=β·log π_ref^θ(y|x)`; resuelve RLHF con una simple pérdida de
clasificación.

**Paso 12 — Generación** (`MODELO_RAG = gemma4`, streaming): se envía el system prompt
RAG + el mensaje de usuario con `<context>` = los hechos de RECOMP. Respuesta final:
> *The key insight of the Direct Preference Optimization (DPO) approach is leveraging an
> analytical mapping from reward functions to optimal policies. This mapping allows for
> the transformation of a loss function … into a loss function … over policies … bypass
> … a standalone reward model and … an RL training loop … the policy network … represents
> both the language model and the (implicit) reward.*

**Paso 13 — Observabilidad.** `guardar_debug_rag` escribe el `.txt` con sub-queries,
keywords, métricas BM25, configuración, fragmentos recuperados (con `Lexical match: BM25`),
síntesis RECOMP y respuesta. Las fuentes de los fragmentos se devuelven como citas a la UI.

> Contraste con la ejecución previa (dump `20260522_204216`, antes de instalar
> `rank-bm25` en el entorno correcto): `fase_keywords.disponible = false`, 0 docs
> indexados, 0 resultados BM25 → el pipeline operaba solo con semántica + reranker. La
> respuesta salía bien igualmente, pero **BM25 no aportaba**. Tras instalar `rank-bm25`
> en el entorno `(base)` y reiniciar el servidor, BM25 ya fusiona (ver Paso 5).
