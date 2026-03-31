# Splits de datasets — MonkeyGrab

## Datasets empleados

El sistema usa 3 fuentes de datos, que se tratan como **5 datasets independientes**
(Aina se divide por idioma):

| # | Dataset | HuggingFace ID |
|---|---------|----------------|
| 1 | Neural-Bridge RAG | `neural-bridge/rag-dataset-12000` |
| 2 | Dolly QA | `databricks/databricks-dolly-15k` |
| 3 | Aina-EN | `projecte-aina/RAG_Multilingual` (lang=en) |
| 4 | Aina-ES | `projecte-aina/RAG_Multilingual` (lang=es) |
| 5 | Aina-CA | `projecte-aina/RAG_Multilingual` (lang=ca) |

---

## Filtros aplicados

### F1 — Campos obligatorios no vacios

Se descartan filas en las que cualquiera de los tres campos del esquema RAG
(`instruction`, `context`, `response`) sea una cadena vacia o ausente.

Impacto: practicamente nulo en Neural-Bridge y Aina (datasets ya limpios).
Solo elimina 1-2 filas en Neural-Bridge.

### F3 — Categoria RAG (solo Dolly)

Dolly contiene tareas heterogeneas (escritura creativa, brainstorming,
clasificacion, etc.) que no son formato RAG. F3 retiene unicamente las filas
cuya categoria sea una de las tres RAG-relevantes **y** que tengan contexto
no vacio:

- `closed_qa`
- `information_extraction`
- `summarization`

Impacto: elimina el **70 %** del dataset Dolly, de 15.011 a 4.467 filas.

---

## Splits naturales en HuggingFace y tamaño tras filtros

```
Dataset                HF split        HF original   tras F1+F3
---------------------- -------------- ------------ ------------
Neural-Bridge RAG      train                 9.600        9.598
Neural-Bridge RAG      test                  2.400        2.399
Dolly QA               train                15.011        4.467  (solo train en HF)
---------------------- -------------- ------------ ------------
Aina-EN                train                14.997       14.997
Aina-ES                train                11.263       11.263
Aina-CA                train                16.043       16.043
Aina-EN                validation            2.999        2.999
Aina-ES                validation            2.252        2.252
Aina-CA                validation            3.208        3.208
Aina-EN                test                  2.000        2.000
Aina-ES                test                  1.503        1.503
Aina-CA                test                  2.140        2.140
```

---

## Esquema en dos fases

El criterio general es: **dev para explorar, test congelado hasta el final.**

### Fase 1 — Evaluacion rapida de modelos base

El objetivo es comparar modelos base sin entrenamiento, de la forma mas barata posible.
Se usa unicamente un subconjunto reducido de dev, fijo y reproducible.
**No se usa train. No se toca test.**

Tamanos recomendados para dev en fase 1:

| Dataset | Dev fase 1 |
|---------|----------:|
| Neural-Bridge RAG | 200–300 |
| Dolly QA | 300–400 |
| Aina-EN | 200–300 |
| Aina-ES | 200–300 |
| Aina-CA | 200–300 |

Criterio por dataset:
- **Neural-Bridge** esta limpio y es homogeneo; con 200-300 muestras se ven diferencias claras entre modelos.
- **Dolly** es el mas heterogeneo incluso despues de F3; conviene darle algo mas de peso para que la comparacion no sea ruidosa.
- **Aina** esta dividido por idioma; cada subconjunto necesita muestra suficiente para que el ranking no dependa de casos anecdoticos.

Que se obtiene: un ranking preliminar de modelos, una idea de que familia merece la pena explorar, y una estimacion barata del rendimiento sin ningun coste de entrenamiento.

### Fase 2 — Entrenamiento y evaluacion formal

Una vez identificados uno o dos modelos candidatos en fase 1, se pasa a la estructura completa.
**Test permanece congelado y se evalua solo una vez, al final.**

| Dataset | Train | Dev | Test |
|---------|------:|----:|-----:|
| Neural-Bridge RAG | 4.000–6.000 | 500–800 | 400–600 |
| Dolly QA | 2.500–4.000 | 400–600 | 400–600 |
| Aina-EN | 4.000–8.000 | 800–1.200 | 800–1.200 |
| Aina-ES | 3.000–6.000 | 600–900 | 600–900 |
| Aina-CA | 4.000–8.000 | 800–1.200 | 800–1.200 |

Uso de cada split:
- **Train**:  adaptación del modelo (SFT con LoRA).
- **Dev**: seleccion de hiperparametros, epoca optima, early stopping.
- **Test**: metrica final real. Se evalua una sola vez.

En esta fase se puede asumir mas coste porque el espacio de busqueda ya esta reducido.

### Resumen operativo

| | Fase 1 | Fase 2 |
|--|--------|--------|
| Train | No se usa | Completo o cap amplio |
| Dev | Submuestra pequena (~250 por dataset) | Submuestra media (500–1.200) |
| Test | Congelado | Congelado; se evalua solo al final |
| Objetivo | Ranking de modelos base | Entrenamiento y evaluacion formal |
| Coste | Bajo | Alto (justificado) |
