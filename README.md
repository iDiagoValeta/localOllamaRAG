<p align="center">
  <img src="logo.png" alt="MonkeyGrab Logo" width="180" />
</p>

<h1 align="center">🐒 MonkeyGrab</h1>

<p align="center">
  <strong>Asistente local de consulta documental con RAG académico</strong><br>
  Conversación inteligente sobre PDFs indexados · 100% local · Sin APIs externas
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/RAG-Híbrido-28A745?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Licencia-MIT-6B4C9A?style=for-the-badge" alt="License">
</p>

---

## ¿Qué es MonkeyGrab?

MonkeyGrab es un sistema RAG (Retrieval-Augmented Generation) local diseñado para consulta académica de documentos PDF. Combina recuperación híbrida, reranking y enriquecimiento general del contexto orquestando diversos modelos ejecutados localmente mediante Ollama.

## Modos de uso

- `CHAT` — Conversación general, guía y soporte operativo
- `RAG`  — Respuestas fundamentadas en PDFs indexados


## Ejecución (CLI)

```bash
cd rag
python chat_pdfs.py
```

Comandos principales en la CLI:

- `/rag`
- `/chat`
- `/docs`
- `/temas`
- `/stats`
- `/reindex`
- `/help`
- `/salir`

## Pipeline RAG

1. Extracción y chunking de PDFs → embeddings → ChromaDB
2. Recuperación: semántica y búsqueda léxica
3. Fusión RRF → Reranking
4. Expansión de contexto y síntesis RECOMP
5. Inferencia final con LLM

## Obligatorio vs Opcionales

Obligatorio (mínimo para que el sistema funcione como RAG):

- Indexación básica: extracción de texto y `dividir_en_chunks()` → embeddings (`ollama.embeddings`) y persistencia en ChromaDB.
- Recuperación semántica: consultas vectoriales (`collection.query`) en `realizar_busqueda_hibrida`.
- Generación con LLM RAG: usar `MODELO_RAG` para producir la respuesta final basada en el contexto recuperado.

Opcionales (activables por flags o que requieren dependencias externas):

- `USAR_CONTEXTUAL_RETRIEVAL`: enriquece chunks con un modelo contextual al indexar.
- `USAR_LLM_QUERY_DECOMPOSITION`: genera sub-queries con un LLM auxiliar para ampliar la cobertura de búsqueda.
- `USAR_BUSQUEDA_HIBRIDA`: añade búsqueda por keywords/where_document junto a la búsqueda semántica.
- `USAR_BUSQUEDA_EXHAUSTIVA`: escaneo profundo por términos críticos.
- `USAR_RERANKER`: reranking mediante Cross-Encoder.
- `USAR_OPTIMIZACION_CONTEXTO`: limpieza y normalización de texto extraído antes de generar la respuesta.
- `USAR_RECOMP_SYNTHESIS`: síntesis RECOMP para resumir contextos antes de la generación.
- `EXPANDIR_CONTEXTO`: recuperar chunks vecinos para dar contexto continuo al LLM.

Muchas de las opciones opcionales mejoran la calidad pero añaden coste computacional o dependencias (GPU, `sentence-transformers`, `pymupdf4llm`, etc.).

<p align="center">Hecho con 🐒 para el TFG · UPV · 2025–2026</p>
