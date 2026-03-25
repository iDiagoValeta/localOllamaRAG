# Evaluación sobre Datasets Públicos — Trabajo Futuro

Ampliación planificada de la evaluación de MonkeyGrab usando datasets públicos con documentos
indexables, para complementar el dataset de ground truth construido manualmente.

La ventaja principal de usar datasets públicos es la **reproducibilidad**: cualquier revisor
puede verificar los resultados sin depender del corpus privado del TFG.

---

## Requisito funcional

Los datasets seleccionados deben incluir **documentos fuente indexables** (no solo pares QA),
de forma que el pipeline completo pueda ser evaluado end-to-end:

```
Documentos fuente → ChromaDB (indexación) → Retrieval híbrido → Generación → Comparación con ground truth
```

Datasets que solo contienen pares QA sin documentos no son válidos para este propósito,
ya que solo permitirían evaluar el modelo generador de forma aislada.

---

## Datasets identificados

### 1. `vectara/open_ragbench` — Recomendado principal

| Atributo | Detalle |
|---|---|
| **Documentos fuente** | ~1000 PDFs de papers de arXiv |
| **Pares QA** | 3045 (abstractivo + extractivo) |
| **Formato** | URLs de PDF en `pdf_urls.json`, descargables directamente |
| **HuggingFace** | `vectara/open_ragbench` |

**Por qué es el más adecuado:** parte de PDFs reales de arXiv, lo que encaja directamente
con el caso de uso de MonkeyGrab. Incluye queries sobre texto, tablas e imágenes, lo que
permite estresar todos los modos de indexación del pipeline (incluyendo el módulo de visión
con `llama3.2-vision:11b`). Los PDFs se descargan y se colocan directamente en `rag/pdfs/`.

**Flujo de integración:**
```bash
# 1. Descargar PDFs desde pdf_urls.json
# 2. Colocar en rag/pdfs/
# 3. /reindex
# 4. Lanzar queries desde el dataset contra el pipeline
# 5. Comparar con RAGAS usando las respuestas de referencia
```

---

### 2. `yixuantt/MultiHopRAG` — Estrés del retriever

| Atributo | Detalle |
|---|---|
| **Documentos fuente** | Corpus de artículos de noticias (descargable completo) |
| **Pares QA** | 2556 queries multi-hop |
| **Complejidad** | Evidencia distribuida en 2–4 documentos por pregunta |
| **HuggingFace** | `yixuantt/MultiHopRAG` |
| **Publicación** | COLM 2024 |

**Por qué es útil:** las preguntas multi-hop requieren cruzar información de varios
fragmentos del corpus. Permite evaluar específicamente si el RRF (55% semántico + 45%
léxico) y el reranker BAAI/bge son capaces de recuperar evidencia distribuida. Justifica
las decisiones de diseño del retrieval híbrido en la memoria del TFG.

---

### 3. `rag-datasets/rag-mini-wikipedia` — Validación rápida

| Atributo | Detalle |
|---|---|
| **Documentos fuente** | Pasajes de Wikipedia (corpus separado del QA, mapeado por ID) |
| **Pares QA** | 4118 |
| **Tamaño** | ~852 KB |
| **HuggingFace** | `rag-datasets/rag-mini-wikipedia` |

**Por qué es útil:** el dataset más ligero de la lista. Sirve como validación rápida
del pipeline sin coste computacional elevado, o como conjunto de warmup antes de
lanzar una evaluación completa con `open_ragbench`.

---

### 4. `dwb2023/ragas-golden-dataset-documents` — Integración directa con RAGAS

| Atributo | Detalle |
|---|---|
| **Documentos fuente** | Incluidos en el dataset |
| **Formato** | Nativo de RAGAS (`question`, `ground_truth`, `contexts`, `answer`) |
| **HuggingFace** | `dwb2023/ragas-golden-dataset-documents` |

**Por qué es útil:** el esquema de este dataset coincide directamente con el que espera
`evaluation/run_eval.py`, lo que minimiza el trabajo de adaptación. Buena opción si
se prioriza la velocidad de integración sobre el tamaño del corpus.

---

## Estrategia de evaluación propuesta

| Nivel | Dataset | Propósito |
|---|---|---|
| Evaluación end-to-end | `vectara/open_ragbench` | Benchmark principal sobre PDFs reales |
| Estrés del retriever | `yixuantt/MultiHopRAG` | Validar RRF + reranker en preguntas multi-hop |
| Validación rápida | `rag-mini-wikipedia` | Smoke test barato antes de evaluaciones largas |
| Integración RAGAS | `dwb2023/ragas-golden-dataset` | Warm-up con formato nativo, sin adaptación |
| Dominio propio | Dataset manual del TFG | Evaluación sobre el dominio objetivo real |

---

## Adaptación a `run_eval.py`

El formato que espera el script de evaluación existente es:

```json
{
  "question": "...",
  "ground_truth": "...",
  "contexts": ["fragmento recuperado 1", "fragmento recuperado 2"],
  "answer": "respuesta generada por el sistema"
}
```

`open_ragbench` y `dwb2023/ragas-golden-dataset` se mapean directamente a este esquema.
`MultiHopRAG` y `rag-mini-wikipedia` requieren un script de adaptación ligero para
generar el campo `contexts` ejecutando el retriever de MonkeyGrab sobre cada pregunta.

---

*Identificado durante la fase de evaluación del TFG — pendiente de implementación.*
