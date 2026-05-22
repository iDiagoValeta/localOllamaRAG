# Defensa de parametros del pipeline RAG

Fecha: 2026-05-22. Este documento clasifica los valores numericos que afectan a
recuperacion, ranking, filtrado y construccion de contexto. La regla de defensa
para el TFG es simple: no presentar como "valor teorico" lo que en realidad es
un hiperparametro operativo del sistema.

## Cambios de trazabilidad

Los valores de ranking ya no quedan como literales anonimos en la fusion. En
`rag/chat_pdfs.py` se exponen como defaults reproducibles y, cuando procede,
como variables de entorno:

| Parametro | Default | Env var | Papel |
| --- | ---: | --- | --- |
| `BM25_K1` | 1.5 | `RAG_BM25_K1` | Saturacion de frecuencia de termino en BM25 |
| `BM25_B` | 0.75 | `RAG_BM25_B` | Normalizacion por longitud en BM25 |
| `RRF_K` | 20 | `RAG_RRF_K` | Amortiguacion de la fusion por ranking |
| `PESO_SEMANTICO_RRF` | 0.55 | `RAG_PESO_SEMANTICO_RRF` | Peso de la rama semantica tras RRF |
| `PESO_BM25_RRF` | 0.45 | `RAG_PESO_BM25_RRF` | Peso de la rama BM25 tras RRF |
| `UMBRAL_SCORE_RERANKER` | 0.65 | `RAG_UMBRAL_SCORE_RERANKER` | Filtro minimo del Cross-Encoder |
| `UMBRAL_RELEVANCIA` | 0.50 | `RAG_UMBRAL_RELEVANCIA` | Puerta de relevancia sobre escala reranker |

`rag/engine/retrieval.py` usa esos nombres en lugar de literales:

```python
score_final = (
    score_semantic * PESO_SEMANTICO_RRF
    + score_keyword * PESO_BM25_RRF
)
```

## Clasificacion de defensa

### 1. Parametros defendibles por literatura o implementacion

`BM25_K1 = 1.5` y `BM25_B = 0.75` pertenecen a Okapi BM25. En este proyecto se
usan porque son los defaults de la clase `BM25Okapi` de `rank-bm25`, no porque
se hayan optimizado sobre el conjunto de evaluacion. `b = 0.75` tambien coincide
con un default comun en motores de busqueda.

Formula usada por BM25:

```text
score(D,Q) = sum_t IDF(t) * f(t,D) * (k1 + 1)
             / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))
```

Defensa recomendada: "Se adopta Okapi BM25 como recuperador disperso clasico.
Los parametros `k1` y `b` se fijan a los valores por defecto de la implementacion
empleada (`rank-bm25`) para evitar ajuste ad hoc sobre el test."

Fuentes:

- Robertson, S. y Zaragoza, H. (2009), *The Probabilistic Relevance Framework:
  BM25 and Beyond*, DOI: https://doi.org/10.1561/1500000019.
- `rank-bm25`, `BM25Okapi(corpus, tokenizer=None, k1=1.5, b=0.75,
  epsilon=0.25)`: https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py.
- Vespa BM25 docs: `k1` y `b` son parametros configurables; `b` default 0.75:
  https://docs.vespa.ai/en/ranking/bm25.html.

### 2. Parametros teoricos con default del proyecto

RRF procede de Cormack, Clarke y Buettcher (2009). La formula original suma:

```text
RRFscore(d) = sum_r 1 / (k + r(d))
```

El paper fija `k = 60` tras una investigacion piloto. El pipeline actual usa
`RRF_K = 20`; por tanto, no debe decirse que `20` es el valor canonico del
paper. Es un hiperparametro del proyecto que da mas peso relativo a las primeras
posiciones, razonable cuando las listas son cortas (`80` semanticos por query y
`40` BM25), pero defendible al maximo solo si se reporta una sensibilidad
`RAG_RRF_K=20` vs `RAG_RRF_K=60`.

Fuente RRF:

- Cormack, G. V., Clarke, C. L. A. y Buettcher, S. (2009), *Reciprocal Rank
  Fusion outperforms Condorcet and individual Rank Learning Methods*:
  https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf.

### 3. Hiperparametros del sistema que conviene presentar como defaults

Los pesos `0.55` / `0.45` no son parte de BM25 ni de RRF. Son una decision de
fusion del sistema:

```text
score_final(d) = 0.55 * score_semantic(d) + 0.45 * score_BM25(d)
```

Defensa recomendada: "La fusion ponderada mantiene una ligera prioridad para la
recuperacion semantica, porque el objetivo principal del sistema es recuperar
contenido conceptualmente relacionado, mientras BM25 actua como ancla lexica.
Los pesos se mantienen fijos durante todas las variantes comparadas y se exponen
como variables de entorno para reproducibilidad/sensibilidad."

Para blindarlo empiricamente, comparar al menos:

```powershell
$env:RAG_PESO_SEMANTICO_RRF="0.50"; $env:RAG_PESO_BM25_RRF="0.50"
$env:RAG_PESO_SEMANTICO_RRF="0.55"; $env:RAG_PESO_BM25_RRF="0.45"
$env:RAG_PESO_SEMANTICO_RRF="0.70"; $env:RAG_PESO_BM25_RRF="0.30"
```

### 4. Umbrales calibrados

`UMBRAL_SCORE_RERANKER = 0.65` es el valor mejor defendido del grupo porque ya
tiene protocolo de sonda en `research/evaluation/probe_reranker_scores.py` y
documentacion en `research/docs/EVALUACIONES_PIPELINE.md`: se subio desde `0.55`
tras observar una banda de ruido `0.40-0.60` y evitar chunks irrelevantes.

`UMBRAL_RELEVANCIA = 0.50` no se aplica a RRF puro. El codigo solo lo usa cuando
`USAR_RERANKER` esta activo, porque entonces `score_final` ya ha sido sustituido
por `score_reranker`. No debe describirse como "umbral RRF".

### 5. Presupuestos operativos, no ranking teorico

Estos valores controlan coste, latencia y limite de contexto. No deben venderse
como optimos teoricos:

| Parametro | Default | Defensa |
| --- | ---: | --- |
| `N_RESULTADOS_SEMANTICOS` | 80 | Presupuesto de recall antes de fusion/reranking |
| `N_RESULTADOS_KEYWORD` | 40 | Presupuesto BM25 antes de fusion |
| `TOP_K_RERANK_CANDIDATES` | 200 | Limite de coste del Cross-Encoder |
| `TOP_K_AFTER_RERANK` | 15 | Retencion post-reranker antes de contexto |
| `TOP_K_FINAL` | 8 | Fragmentos base enviados al generador |
| `N_TOP_PARA_EXPANSION` | 3 | Vecinos para continuidad textual |
| `MAX_CONTEXTO_CHARS` | 24000 | Presupuesto de entrada al LLM |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 2000 / 400 | Granularidad de chunks; overlap aproximado del 20% |

Defensa recomendada: "Son presupuestos de computo/contexto fijados antes de la
evaluacion y mantenidos constantes entre variantes; no son parametros aprendidos."

## Riesgos argumentales y respuesta

| Critica posible | Respuesta defendible |
| --- | --- |
| "Por que `BM25_K1=1.5`?" | Es el default de `rank-bm25` para `BM25Okapi`; no se ajusto contra el test. |
| "Por que `RRF_K=20` si el paper usa 60?" | No se presenta como valor del paper; es el default del proyecto para listas cortas. Para defensa fuerte, reportar sensibilidad 20 vs 60. |
| "Por que 55/45?" | Es un peso fijo de fusion que prioriza levemente la rama semantica. Debe reportarse como hiperparametro, no como resultado teorico. |
| "El umbral 0.65 es arbitrario" | Tiene sonda especifica (`probe_reranker_scores.py`) y decision documentada en `EVALUACIONES_PIPELINE.md`. |
| "BM25 no usa sub-consultas" | Correcto: BM25 se aplica solo a la pregunta original; la descomposicion LLM alimenta la rama semantica. |

## Recomendacion para el texto del TFG

No escribir: "se usa RRF con los parametros originales" si se mantiene
`RRF_K=20`.

Si no se repite evaluacion de sensibilidad, escribir:

> La recuperacion hibrida combina una rama densa y una rama dispersa BM25. BM25
> utiliza los valores por defecto de la implementacion `rank-bm25` (`k1=1.5`,
> `b=0.75`). Las listas semantica y lexica se transforman mediante RRF y se
> combinan con pesos fijos `0.55/0.45`, definidos como hiperparametros del
> sistema y mantenidos constantes en todas las variantes comparadas. El umbral
> del reranker (`0.65`) se fijo tras una sonda de distribucion de scores previa
> a la evaluacion final.

Para defensa mas fuerte, anadir una tabla de sensibilidad con `RRF_K=60` y
pesos `0.50/0.50` frente al baseline actual.
