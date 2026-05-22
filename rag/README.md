# Pipeline RAG — Documentación Técnica

MonkeyGrab implementa un pipeline RAG (Retrieval-Augmented Generation) completamente local sobre Ollama + ChromaDB. Este documento describe cada etapa del pipeline, las funciones que la implementan y los parámetros que la controlan.

---

## Índice

1. [Arquitectura general](#1-arquitectura-general)
2. [Etapa 1 — Indexación](#2-etapa-1--indexación)
3. [Etapa 2 — Recuperación híbrida](#3-etapa-2--recuperación-híbrida)
4. [Etapa 3 — Reranking y expansión de contexto](#4-etapa-3--reranking-y-expansión-de-contexto)
5. [Etapa 4 — Construcción del contexto](#5-etapa-4--construcción-del-contexto)
6. [Etapa 5 — Generación](#6-etapa-5--generación)
7. [Módulos transversales](#7-módulos-transversales)
8. [Configuración global y flags](#8-configuración-global-y-flags)
9. [Apéndice A — Metadatos ChromaDB](#apéndice-a--metadatos-chromadb)
10. [Apéndice B — Sincronización runtime](#apéndice-b--sincronización-runtime)
11. [Apéndice C — Flujo completo de ejemplo](#apéndice-c--flujo-completo-de-ejemplo)

---

## 1. Arquitectura general

```
PDF corpus
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ETAPA 1: INDEXACIÓN                                │
│  chunking → contextual enrichment → embeddings      │
│  → ChromaDB  (+ OCR de imágenes, opcional)          │
└──────────────────────┬──────────────────────────────┘
                       │ collection ChromaDB
    ┌──────────────────▼──────────────────────────────┐
    │  ETAPA 2: RECUPERACIÓN HÍBRIDA                  │
    │  query decomposition → semántica + BM25         │
    │  → RRF fusion                                   │
    └──────────────────┬──────────────────────────────┘
                       │ candidatos con scores
    ┌──────────────────▼──────────────────────────────┐
    │  ETAPA 3: RERANKING + EXPANSIÓN                 │
    │  Cross-Encoder → top-K → neighbor expansion     │
    └──────────────────┬──────────────────────────────┘
                       │ fragmentos finales
    ┌──────────────────▼──────────────────────────────┐
    │  ETAPA 4: CONSTRUCCIÓN DE CONTEXTO              │
    │  optimización PDF → RECOMP synthesis (opt)      │
    └──────────────────┬──────────────────────────────┘
                       │ contexto listo
    ┌──────────────────▼──────────────────────────────┐
    │  ETAPA 5: GENERACIÓN                            │
    │  Ollama streaming → respuesta + debug dump      │
    └─────────────────────────────────────────────────┘
```

### Modos de operación

| Modo | Descripción |
|------|-------------|
| **CHAT** | Conversación libre con historial persistente (no usa documentos) |
| **RAG**  | Consultas documentales: ejecuta el pipeline completo de 5 etapas |

### Estructura de módulos

```
rag/
├── chat_pdfs.py          — Fachada pública + toda la configuración global
└── engine/
    ├── runtime.py        — Sincronización de globals entre módulos
    ├── indexing.py       — Orquestación de indexación
    ├── chunking.py       — División en fragmentos
    ├── contextual.py     — Contextual retrieval + detección de idioma
    ├── images.py         — Extracción OCR de imágenes PDF
    ├── retrieval.py      — Orquestación de búsqueda híbrida
    ├── reranking.py      — Query decomposition + Cross-Encoder
    ├── lexical.py        — Búsqueda por keywords
    ├── context.py        — Construcción y optimización del contexto
    ├── generation.py     — Generación de respuestas + evaluación silenciosa
    ├── history.py        — Persistencia del historial de chat
    └── debug.py          — Volcado de interacciones RAG
```

---

## 2. Etapa 1 — Indexación

### 2.1 Función principal: `indexar_documentos()`

**Archivo**: `rag/engine/indexing.py`

```python
def indexar_documentos(
    carpeta: str,
    collection: chromadb.Collection,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int
```

Por cada PDF encontrado en `carpeta`:

1. Extrae texto con `pymupdf4llm` (Markdown); si falla, fallback a `pypdf`.
2. Detecta idioma del documento (`_detectar_idioma`).
3. Divide el texto en chunks (`dividir_en_chunks`).
4. Opcionalmente enriquece cada chunk con contexto situacional (`generar_contexto_situacional`) si `USAR_CONTEXTUAL_RETRIEVAL = True`.
5. Calcula embeddings vía Ollama (`MODELO_EMBEDDING`) prefijando el texto con `EMBED_PREFIX_DOC`.
6. Almacena en ChromaDB con metadatos de página, chunk e índice.
7. Opcionalmente extrae imágenes y las indexa como chunks especiales (`USAR_EMBEDDINGS_IMAGEN`).

**Parámetros de configuración relevantes** (definidos en `rag/chat_pdfs.py`):

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 2000 | Tamaño máximo de un chunk en caracteres |
| `CHUNK_OVERLAP` | 400 | Solapamiento entre chunks consecutivos (~20%) |
| `MIN_CHUNK_LENGTH` | 150 | Descarta artefactos más cortos |
| `CONTEXTUAL_DOC_CHARS` | 24000 | Muestra del documento para generar el contexto situacional |
| `EMBED_PREFIX_DOC` | `"search_document: "` o `""` | Prefijo de documento para el modelo de embeddings activo; auto-configurado en `chat_pdfs.py` según `MODELO_EMBEDDING`; vacío si el modelo no lo requiere |
| `progress_callback` | `None` | Callable opcional `(info: dict) → None`; recibe `{"file", "file_index", "total_files"}` en cada PDF procesado; usado por la web para mostrar progreso en tiempo real |

---

### 2.2 Chunking: `dividir_en_chunks()`

**Archivo**: `rag/engine/chunking.py`

```python
def dividir_en_chunks(
    texto: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, str]]
```

**Estrategia recursiva** con separadores jerárquicos:

```
\n\n  →  \n  →  ". "  →  ", "  →  " "
```

Para cada chunk resultante:
- Preserva el header Markdown (`# …`) más cercano previo al fragmento.
- Solapamiento: toma los últimos `overlap` caracteres del chunk anterior y los antepone al siguiente.
- Filtra fragmentos con longitud < `MIN_CHUNK_LENGTH`.
- Limpia markup de presentación: tachado, asteriscos de énfasis, backticks.
- Colapsa `\n{3,}` a `\n\n`.

Retorna `[{"text": "...", "header": "..."}]`.

---

### 2.3 Contextual retrieval: `generar_contexto_situacional()`

**Archivo**: `rag/engine/contextual.py`

```python
def generar_contexto_situacional(
    chunk_text: str,
    texto_base: str,
    idioma_doc: str = "",
) -> str
```

Controlado por el flag `USAR_CONTEXTUAL_RETRIEVAL`. Cuando está activo, antes de indexar un chunk se llama a `MODELO_CONTEXTUAL` (contexto de `OLLAMA_CONTEXTUAL_NUM_CTX` = 32768 tokens) con el prompt:

```
System: "Escribe exactamente 2-3 frases sobre cómo este fragmento
         encaja en el documento global."
User:   <document>{texto_base[:CONTEXTUAL_DOC_CHARS]}</document>
        <excerpt>{chunk_text}</excerpt>
```

La salida se antepone al texto del chunk usando el separador literal `\\n\\n` (6 bytes, no un salto de línea real), lo que permite distinguir contexto situacional de cuerpo de chunk al recuperar sin ambigüedad con los saltos naturales del PDF.

El texto indexado final tiene la forma:

```
<2-3 frases situacionales>\\n\\n<cuerpo del chunk>
```

**Parámetros**:

| Constante | Valor |
|-----------|-------|
| `MODELO_CONTEXTUAL` | Configurable via `OLLAMA_CONTEXTUAL_MODEL` |
| `OLLAMA_CONTEXTUAL_NUM_CTX` | 32768 |
| `CONTEXTUAL_DOC_CHARS` | 24000 |

---

### 2.4 Detección de idioma: `_detectar_idioma()`

**Archivo**: `rag/engine/contextual.py`

```python
def _detectar_idioma(texto: str) -> str  # → 'Spanish' | 'Catalan' | 'English'
```

Cuenta tokens indicativos de cada idioma (conjunciones, artículos, preposiciones específicas) y devuelve el idioma con mayor puntuación. El resultado se propaga a `generar_contexto_situacional` para que el LLM responda en el idioma del documento.

---

### 2.5 OCR de imágenes: `extraer_imagenes_pdf()` + `describir_imagen_con_llm()`

**Archivo**: `rag/engine/images.py`

Controlado por `USAR_EMBEDDINGS_IMAGEN`.

#### Extracción

```python
def extraer_imagenes_pdf(
    ruta_pdf: str,
    max_por_pagina: int = MAX_IMAGENES_POR_PAGINA,   # 5
    min_size_px: int = MIN_IMAGEN_SIZE_PX,            # 100
) -> Dict[int, List[Dict[str, Any]]]
```

Usa PyMuPDF (`fitz`) para iterar páginas y extraer imágenes como bytes PNG/JPEG. Filtra:
- Imágenes más pequeñas que `MIN_IMAGEN_SIZE_PX` (100 px) en cualquier dimensión.
- Más de `MAX_IMAGENES_POR_PAGINA` (5) por página.
- Detecta caption buscando texto en los `CAPTION_MARGIN_PX` (80 px) inmediatamente bajo la imagen.

#### Descripción con OCR

```python
def describir_imagen_con_llm(
    image_bytes: bytes,
    caption: str = "",
    idioma_doc: str = "English",
) -> str
```

Envía la imagen (base64) a `MODELO_OCR` con un prompt estructurado que guía la descripción según el tipo visual:
- **Diagrama**: bloques, entradas/salidas, flujo de datos.
- **Tabla**: estructura de filas y columnas, valores clave.
- **Gráfico**: ejes, leyendas, tendencias.

Las descripciones degeneradas se descartan mediante tres filtros:

| Función | Qué detecta |
|---------|-------------|
| `_es_descripcion_spam()` | Léxico repetitivo (<35% palabras únicas) o >20% tokens "no"/"text" |
| `_es_prompt_echo()` | El modelo repite fragmentos del propio prompt |
| `_es_solo_caption()` | >85% de solapamiento con el caption sin información adicional |

Las imágenes válidas se indexan como chunks con `format = "image"` y un `chunk_id` desplazado en `_IMAGEN_CHUNK_OFFSET` (10 000) para no colisionar con chunks de texto.

---

## 3. Etapa 2 — Recuperación híbrida

### 3.1 Orquestador: `realizar_busqueda_hibrida()`

**Archivo**: `rag/engine/retrieval.py`

```python
def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]
```

Devuelve `(fragmentos_ordenados, tiempo_total, stats)`. Internamente ejecuta los pasos A–D en orden.

---

### 3.2 A) Query decomposition: `generar_queries_con_llm()`

**Archivo**: `rag/engine/reranking.py`

```python
def generar_queries_con_llm(pregunta: str) -> List[str]
```

Activa si `USAR_LLM_QUERY_DECOMPOSITION = True` **y** `len(pregunta) > 60`. Llama a `MODELO_CHAT` con `think=False` para generar **3 sub-queries** que cubren aspectos distintos de la pregunta original, en el mismo idioma. Las sub-queries se añaden a la lista de queries que entran en el paso semántico.

---

### 3.3 B) Búsqueda semántica + RRF

Para cada query (original + sub-queries):

1. Prefija el texto con `EMBED_PREFIX_QUERY` (auto-configurado en `chat_pdfs.py` según `MODELO_EMBEDDING`; vacío si el modelo no requiere prefijo).
2. Calcula embeddings vía Ollama (`MODELO_EMBEDDING`).
3. Consulta ChromaDB: `collection.query(n_results=N_RESULTADOS_SEMANTICOS)` donde `N_RESULTADOS_SEMANTICOS = 80`.
4. Acumula en un diccionario de candidatos:

```python
score_semantic[doc_id] += 1.0 / (rank + RRF_K)   # RRF_K = 20
```

---

### 3.4 C) Búsqueda léxica BM25: `busqueda_lexica_bm25()`

**Archivo**: `rag/engine/lexical.py`

```python
def busqueda_lexica_bm25(
    pregunta: str,
    collection: chromadb.Collection,
    top_n: int = N_RESULTADOS_KEYWORD,   # 40
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
```

Activa si `USAR_BUSQUEDA_HIBRIDA = True`. Requiere `rank-bm25` (flag
`BM25_AVAILABLE`); si no está instalado, devuelve lista vacía y el pipeline opera
solo con la vía semántica.

Implementa recuperación dispersa clásica **Okapi BM25** (Robertson & Zaragoza,
2009): puntúa cada fragmento por frecuencia de término, rareza del término en la
colección (IDF) y normalización por longitud, produciendo un **ranking de
relevancia real** en lugar de un filtro de subcadena. Sustituye a la antigua
búsqueda por `$contains` y a la búsqueda exhaustiva, que eran redundantes.

#### Tokenización: `_tokenizar_bm25()`

Tokenizador **único** para corpus y query (requisito de BM25): minúsculas, split
por límites no alfanuméricos (Unicode, conserva acentos), descarta `STOPWORDS`
multiidioma (es/ca/en) y tokens de menos de 3 caracteres salvo que contengan
dígitos (se conservan identificadores y métricas).

#### Búsqueda

El índice BM25 se **reconstruye por consulta**: se escanea toda la colección en
batches, se tokeniza el corpus, se construye `BM25Okapi(corpus, k1=BM25_K1,
b=BM25_B)` (`BM25_K1 = 1.5`, `BM25_B = 0.75`) y se puntúa la query con
`get_scores()`. Se devuelven los `top_n` fragmentos con score positivo,
ordenados de mayor a menor. La fusión RRF usa ese **rango real**:

```python
score_keyword[doc_id] += 1.0 / (rank + RRF_K)   # rank = posición por score BM25
```

> `extraer_keywords()` se mantiene para métricas/depuración y para la
> decomposición de query, pero ya no dirige la recuperación léxica.

---

### 3.5 D) Fusión RRF final

Una vez recogidos los candidatos de ambas vías (semántica + BM25):

```python
score_final = score_semantic * 0.55 + score_keyword * 0.45
```

Los candidatos se ordenan descendentemente por `score_final`. El umbral mínimo para pasar a la siguiente etapa es `UMBRAL_RELEVANCIA = 0.50`.

---

## 4. Etapa 3 — Reranking y expansión de contexto

### 4.1 Reranking: `rerank_resultados()`

**Archivo**: `rag/engine/reranking.py`

```python
def rerank_resultados(
    pregunta: str,
    documentos_recuperados: List[Dict[str, Any]],
    top_k: int = TOP_K_AFTER_RERANK,   # 15
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
```

Activa si `USAR_RERANKER = True` (auto-detectado según disponibilidad de `sentence-transformers`).

**Tier configurable** (controlado por `RERANKER_QUALITY`):

| Calidad | Uso previsto |
|---------|--------------|
| `"quality"` | Mayor precision, mas coste |
| `"fast"` | Menor latencia, menor coste |

**Flujo**:

1. Para cada fragmento, extrae el cuerpo de texto limpio: si el texto contiene `\\n\\n` (separador contextual), toma la parte posterior al separador.
2. Construye pares `(pregunta, texto_cuerpo)` para los `TOP_K_RERANK_CANDIDATES` (200) mejores candidatos por `score_final`.
3. Ejecuta `CrossEncoder.rank()` en CPU (FP32) o CUDA (FP16, auto-detectado).
4. Sustituye `score_final` por `score_reranker` (rango [0, 1]).
5. Filtra por `UMBRAL_SCORE_RERANKER = 0.65` y retiene el top `TOP_K_AFTER_RERANK` (15).

---

### 4.2 Expansión de vecinos: `expandir_con_chunks_adyacentes()`

**Archivo**: `rag/engine/retrieval.py`

```python
def expandir_con_chunks_adyacentes(
    chunk_id: str,
    metadata: Dict[str, Any],
    n_vecinos: int = 1,
) -> List[str]
```

Activa si `EXPANDIR_CONTEXTO = True`. Para los `N_TOP_PARA_EXPANSION` (3) fragmentos con mayor score, construye los IDs de los chunks adyacentes (misma página, página anterior, página siguiente) y los recupera de ChromaDB si existen. Esto repara la continuidad de párrafos truncados en los límites de chunk.

---

## 5. Etapa 4 — Construcción del contexto

### 5.1 Optimización de texto: `optimizar_texto_contexto()`

**Archivo**: `rag/engine/context.py`

```python
def optimizar_texto_contexto(texto: str) -> str
```

Activa si `USAR_OPTIMIZACION_CONTEXTO = True`. Elimina artefactos comunes en extracciones PDF que consumen tokens sin aportar información:
- Headers Markdown (`^#{1,6}\s+`)
- Patrones de footer (autor/fecha)
- Múltiples espacios consecutivos
- Espacios trailing por línea
- Párrafos huérfanos de un solo dígito (números de página)
- Triples saltos de línea o más → doble salto

El módulo registra el ahorro: `"Optimized context: 12000 -> 8500 chars (29.2%)"`.

La función auxiliar `_es_continuacion_parrafo()` detecta párrafos quebrados por la extracción del PDF aplicando heurísticas (¿la línea anterior termina en `.?!`? ¿la línea actual empieza en minúscula?) y `_reunir_parrafos()` los reconstituye.

---

### 5.2 Ensamblaje del contexto: `construir_contexto_para_modelo()`

**Archivo**: `rag/engine/context.py`

```python
def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str
```

Ordena los fragmentos por `(source, page, chunk)` y genera el bloque de contexto con el siguiente formato por fragmento:

```
--- [Fragment N] ---
[Fragment Context]
<contexto situacional, si existe>

[Source Text]
<cuerpo del fragmento, optimizado>

[excerpt ends mid-sentence]   ← solo si no termina con .?!:")]
```

El total de caracteres se limita a `MAX_CONTEXTO_CHARS = 24000`; si se supera, los fragmentos de menor score se truncan.

---

### 5.3 Síntesis RECOMP: `sintetizar_contexto_recomp()`

**Archivo**: `rag/engine/context.py`

```python
def sintetizar_contexto_recomp(
    fragmentos: List[Dict[str, Any]],
    query_usuario: str = "",
) -> str
```

Activa si `USAR_RECOMP_SYNTHESIS = True`. En lugar de los chunks en bruto, envía los fragmentos a `MODELO_RECOMP` con el prompt:

```
System:
  "Comprimes fragmentos en un briefing para un modelo downstream.
   SOLO información de los fragmentos. Sin conocimiento externo.
   Si la pregunta pide lista/conteo, ENUMERA TODOS los ítems."

User:
  ## User question
  <pregunta>

  ## Evidence excerpts
  <fragmentos>

  Produce: ## Facts relevant to the question
  - (hecho 1)
  - (hecho 2)
  ...
```

**Condiciones de fallback a raw chunks** (se descarta la síntesis si):
- Salida < 20 caracteres.
- La salida no contiene el encabezado `## Facts relevant to the question`.
- Error de comunicación con Ollama.

Antes de retornar, aplica `_strip_ollama_think_blocks()` para eliminar bloques `<think>…</think>` que algunos modelos de razonamiento emiten.

---

## 6. Etapa 5 — Generación

### 6.1 Función principal: `generar_respuesta()`

**Archivo**: `rag/engine/generation.py`

```python
def generar_respuesta(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    metricas: Optional[Dict[str, Any]] = None,
    on_token=None,
) -> str
```

**Flujo**:

1. `_preparar_mensaje_usuario_rag()`: construye el mensaje de usuario final intercalando la pregunta y el contexto dentro de etiquetas `<context>…</context>`.
2. `_generar_respuesta_stream()`: llama a Ollama con streaming y emite tokens al caller vía `on_token`.
3. `guardar_debug_rag()`: vuelca el dump completo de la interacción.

---

### 6.2 Streaming: `_ollama_generate_stream()` / `_generacion_chat_stream()`

**Archivo**: `rag/engine/generation.py`

```python
def _ollama_generate_stream(
    model: str,
    prompt: str,
    options: dict,
    system: Optional[str] = None,
)  # yields str (JSON lines de /api/generate)
```

**Opciones de generación en modo RAG**:

```python
{
    "temperature":    0.15,
    "top_p":          0.9,
    "repeat_penalty": 1.15,
    "repeat_last_n":  64,                   # pinned to avoid Ollama default drift
    "num_predict":    -1,                   # no token cap; relies on num_ctx
    "num_ctx":        OLLAMA_RAG_NUM_CTX,   # 16384
}
```

Adicionalmente el payload fuerza `think=False` para que los modelos con razonamiento (Qwen3, Gemma 4) no consuman `num_predict` en una traza interna antes de emitir la respuesta.

Si el nombre del modelo contiene `"finetuned"`, el system prompt está horneado en el Modelfile y **no** se envía vía API. En caso contrario se envía `SYSTEM_PROMPT_RAG` explícitamente.

El timeout total de Ollama es `OLLAMA_REQUEST_TIMEOUT = 900` segundos (15 minutos).

---

### 6.3 Evaluación silenciosa: `evaluar_pregunta_rag()`

**Archivo**: `rag/engine/generation.py`

```python
def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection,
) -> Tuple[str, List[str]]
```

Camino exclusivo para las evaluaciones RAGAS. Ejecuta el pipeline completo pero:
- No imprime nada en terminal.
- No genera debug dumps.
- Aplica `UMBRAL_RELEVANCIA` y `UMBRAL_SCORE_RERANKER` normalmente.
- Si `EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK = True`, relaja el umbral de reranker cuando no hay fragmentos con score suficiente.

Retorna `(respuesta, lista_de_contextos_utilizados)`.

---

## 7. Módulos transversales

Estos módulos no forman parte de una etapa concreta del pipeline pero son consumidos por varias etapas.

---

### 7.1 Historial de conversación: `history.py`

**Archivo**: `rag/engine/history.py`

```python
def cargar_historial() -> List[Dict[str, str]]
def guardar_historial(historial: List[Dict[str, str]]) -> None
def limpiar_historial(historial: List[Dict[str, str]]) -> None
```

- `cargar_historial` lee `HISTORIAL_PATH` (JSON); retorna `[]` si no existe o está corrupto.
- `guardar_historial` persiste la lista truncada a los últimos `MAX_HISTORIAL_MENSAJES = 40` mensajes.
- `limpiar_historial` vacía la lista in-place y persiste el estado vacío.

Los mensajes siguen el formato `{"role": "user"|"assistant", "content": "..."}`.

---

### 7.2 Debug de interacciones RAG: `debug.py`

**Archivo**: `rag/engine/debug.py`

```python
def guardar_debug_rag(
    pregunta: str,
    mensaje_usuario: str = "",
    respuesta: str = "",
    fragmentos: list | None = None,
    motivo_interrupcion: str | None = None,
    metricas: dict | None = None,
) -> None
```

Condicionado a `GUARDAR_DEBUG_RAG = True`. Escribe un fichero de texto en `CARPETA_DEBUG_RAG` (por defecto `rag/debug_rag/`) con naming `TIMESTAMP_SLUG.txt`. El volcado incluye:

- Sub-queries generadas, keywords extraídas y términos críticos.
- Flags del pipeline activos en el momento de la llamada.
- System prompt, contexto inyectado y mensaje de usuario completo.
- Respuesta del modelo y scores de cada fragmento (`score_final`, `score_reranker`).

---

## 8. Configuración global y flags

Toda la configuración se centraliza en `rag/chat_pdfs.py`. Los valores se leen desde variables de entorno con defaults embebidos.

### Modelos Ollama

El pipeline se describe por roles configurables. Cada rol se resuelve desde una
variable de entorno y puede apuntar a cualquier modelo compatible disponible en
Ollama.

| Constante | Variable configurable | Rol |
|-----------|-----------------------|-----|
| `MODELO_RAG` | `OLLAMA_RAG_MODEL` | Generación de respuestas RAG |
| `MODELO_CHAT` | `OLLAMA_CHAT_MODEL` | Modo CHAT + query decomposition |
| `MODELO_EMBEDDING` | `OLLAMA_EMBED_MODEL` | Embeddings de documentos y queries |
| `MODELO_CONTEXTUAL` | `OLLAMA_CONTEXTUAL_MODEL` | Generación de contexto situacional |
| `MODELO_RECOMP` | `OLLAMA_RECOMP_MODEL` | Síntesis RECOMP |
| `MODELO_OCR` | `OLLAMA_OCR_MODEL` | Descripción de imágenes |
| `RERANKER_MODEL_QUALITY` | `RERANKER_QUALITY` | Tier del reranker local: `"quality"` carga `BAAI/bge-reranker-v2-m3`; `"fast"` carga `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### Context windows

| Constante | Valor | Modelo que la usa |
|-----------|-------|-------------------|
| `OLLAMA_NUM_CTX` | 8192 | General |
| `OLLAMA_RAG_NUM_CTX` | 16384 | `MODELO_RAG` |
| `OLLAMA_AUX_NUM_CTX` | 8192 | Modelos auxiliares |
| `OLLAMA_QUERY_NUM_CTX` | 8192 | `MODELO_CHAT` (query decomposition) |
| `OLLAMA_CONTEXTUAL_NUM_CTX` | 32768 | `MODELO_CONTEXTUAL` |
| `OLLAMA_RECOMP_NUM_CTX` | 8192 | `MODELO_RECOMP` |
| `OLLAMA_OCR_NUM_CTX` | 8192 | `MODELO_OCR` |
| `OLLAMA_REQUEST_TIMEOUT` | 900 | Todos |

### Parámetros de chunking y retrieval

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 2000 | Tamaño máximo de chunk (chars) |
| `CHUNK_OVERLAP` | 400 | Solapamiento entre chunks (~20%) |
| `MIN_CHUNK_LENGTH` | 150 | Descarta chunks demasiado cortos |
| `CONTEXTUAL_DOC_CHARS` | 24000 | Muestra para contexto situacional |
| `N_RESULTADOS_SEMANTICOS` | 80 | Resultados por query semántica |
| `N_RESULTADOS_KEYWORD` | 40 | Resultados por búsqueda keyword |
| `TOP_K_RERANK_CANDIDATES` | 200 | Candidatos que entran al reranker |
| `TOP_K_AFTER_RERANK` | 15 | Fragmentos tras reranking |
| `TOP_K_FINAL` | 8 | Fragmentos enviados al LLM |
| `N_TOP_PARA_EXPANSION` | 3 | Fragmentos que reciben expansión de vecinos |
| `RRF_K` | 20 | Factor de amortiguamiento RRF |
| `UMBRAL_RELEVANCIA` | 0.50 | Score RRF mínimo para pasar |
| `UMBRAL_SCORE_RERANKER` | 0.65 | Score Cross-Encoder mínimo (subido desde 0.55 tras sonda 2026-05-14) |
| `MAX_CONTEXTO_CHARS` | 24000 | Máximo de chars de contexto al LLM |
| `MIN_LONGITUD_PREGUNTA_RAG` | 10 | Mínimo de caracteres para activar el pipeline RAG |
| `MAX_IMAGENES_POR_PAGINA` | 5 | Máximo de imágenes extraídas por página PDF |
| `CAPTION_MARGIN_PX` | 80 | Píxeles debajo de la imagen donde se busca el caption |

### Flags booleanos del pipeline

| Flag | Default | Efecto cuando `True` |
|------|---------|----------------------|
| `USAR_CONTEXTUAL_RETRIEVAL` | `True` | Enriquece cada chunk con contexto situacional en indexación |
| `USAR_LLM_QUERY_DECOMPOSITION` | `True` | Genera 3 sub-queries para recuperación multi-aspecto |
| `USAR_BUSQUEDA_HIBRIDA` | `True` | Añade búsqueda léxica Okapi BM25 (rank-bm25) fusionada por RRF |
| `USAR_RERANKER` | auto | Aplica Cross-Encoder tras fusión RRF |
| `EXPANDIR_CONTEXTO` | `True` | Añade chunks adyacentes a los fragmentos top |
| `USAR_OPTIMIZACION_CONTEXTO` | `True` | Limpia artefactos PDF del contexto |
| `USAR_RECOMP_SYNTHESIS` | `True` | Sintetiza el contexto antes de enviar al LLM |
| `USAR_EMBEDDINGS_IMAGEN` | `True` | Indexa imágenes de los PDFs con OCR |
| `EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK` | `False` | Relaja umbral reranker en evaluaciones |
| `LOGGING_METRICAS` | `True` | Imprime métricas de cada etapa |
| `GUARDAR_DEBUG_RAG` | `True` | Guarda dump de cada interacción RAG |

---

## Apéndice A — Metadatos ChromaDB

Cada fragmento indexado lleva los siguientes metadatos:

```python
{
    "source":               "paper.pdf",        # nombre del archivo PDF
    "page":                 3,                   # página (0-indexed)
    "chunk":                1,                   # índice del chunk en la página
    "total_chunks_in_page": 5,                  # total de chunks en esa página
    "format":               "markdown",          # "markdown" | "plain_text" | "image"
    "section_header":       "## Metodología",   # header Markdown más cercano
    # Solo para chunks de imagen:
    "image_width":          800,
    "image_height":         600,
}
```

Los IDs de chunks de imagen se calculan como `chunk_num + _IMAGEN_CHUNK_OFFSET` (10 000) para evitar colisiones con los chunks de texto.

---

## Apéndice B — Sincronización runtime

**Archivo**: `rag/engine/runtime.py`

```python
def sync_runtime_globals(namespace: MutableMapping[str, Any]) -> None
```

Todos los módulos de `rag/engine/` llaman a esta función al inicio de cada función pública para copiar los valores actuales de `rag/chat_pdfs.py` en su propio espacio de nombres. Esto permite que los toggles aplicados vía CLI o API web (que modifican variables en `chat_pdfs`) se propaguen inmediatamente sin reiniciar el proceso.

`_RUNTIME_NAMES` contiene ~150 nombres: modelos, flags, constantes numéricas, prefijos de embedding y referencias a funciones auxiliares compartidas.

---

## Apéndice C — Flujo completo de ejemplo

Consulta: `"¿Qué componentes tiene una arquitectura Transformer?"`

```
1. QUERY DECOMPOSITION  (len > 60, USAR_LLM_QUERY_DECOMPOSITION=True)
   Sub-queries generadas:
     a) "Componentes principales del modelo Transformer"
     b) "Mecanismo de atención multi-head en Transformer"
     c) "Codificador y decodificador en la arquitectura Transformer"

2. BÚSQUEDA SEMÁNTICA  (4 queries × 80 resultados)
   Embeddings vía MODELO_EMBEDDING con prefix "search_query: "
   ChromaDB.query() → RRF con k=20
   Candidatos acumulados con score_semantic

3. BÚSQUEDA LÉXICA BM25  (USAR_BUSQUEDA_HIBRIDA=True)
   Tokeniza toda la colección y la query con _tokenizar_bm25()
   BM25Okapi(corpus, k1=1.5, b=0.75).get_scores(query_tokens)
   Top-N fragmentos por score BM25; RRF acumulado en score_keyword

4. FUSIÓN RRF
   score_final = score_semantic × 0.55 + score_keyword × 0.45
   Filtro: score_final >= 0.50
   Resultado: ~50-80 candidatos ordenados

5. RERANKING  (USAR_RERANKER=True)
   Entrada: top 200 candidatos por score_final
   CrossEncoder.rank(pairs=[(pregunta, texto_chunk)])
   Umbral: score_reranker >= 0.65
   Salida: top 15 fragmentos con score_reranker

6. EXPANSIÓN DE VECINOS  (EXPANDIR_CONTEXTO=True)
   Para los 3 fragmentos con mayor score: recupera chunks adyacentes
   Añade vecinos al conjunto de fragmentos finales

7. CONSTRUCCIÓN DE CONTEXTO
   USAR_RECOMP_SYNTHESIS=True → síntesis con MODELO_RECOMP:
     "## Facts relevant to the question
      - El Transformer consta de un codificador y un decodificador...
      - El mecanismo de atención multi-head..."
   USAR_OPTIMIZACION_CONTEXTO=True → limpieza de artefactos PDF

8. GENERACIÓN
   Mensaje usuario: pregunta + <context>síntesis</context>
   System prompt: SYSTEM_PROMPT_RAG (si modelo no tiene baked)
   Ollama streaming: temperature=0.15, num_ctx=16384
   Tokens emitidos en tiempo real al terminal/web

9. DEBUG
   Archivo: rag/debug_rag/YYYYMMDD_HHMMSS_que_componentes_tiene.txt
   Contenido: flags, sub-queries, keywords, scores por fragmento,
              contexto enviado, respuesta completa
```
