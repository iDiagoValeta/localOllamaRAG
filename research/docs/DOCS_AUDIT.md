# Inventario y auditoría de documentación

> Único índice de toda la documentación del repositorio. Cada fila indica
> propósito, audiencia y la última fecha en que se verificó contra el código.
> Se actualiza tras cada cambio en el doc correspondiente.
>
> **Política de qué se documenta dónde** (para evitar drift entre docs):
>
> - `CLAUDE.md` → reglas operativas para Claude Code (símbolos públicos, flags
>   prohibidas, política git).
> - `README.md` (raíz) → instalación y uso del producto (CLI + Web).
> - `rag/README.md` → referencia técnica del pipeline RAG (módulos, signaturas, parámetros).
> - `research/README.md` → mapa del workspace experimental; los detalles viven en sub-docs.
> - `research/docs/EVALUACIONES_PIPELINE.md` → protocolo TFG de evaluación.
> - `research/docs/REINFERENCIA_FINAL.md` → protocolo de la segunda pasada de inferencia.
> - Model cards en `research/models/gguf/<modelo>/README.md` → auto-contenidas, HF Hub.

| Doc | Propósito | Audiencia | Última verificación | Riesgo de drift |
|---|---|---|---|---|
| `CLAUDE.md` | Reglas operativas del proyecto (símbolos públicos, flags, git) | Claude Code / autor | 2026-05-14 | Alto |
| `README.md` | Instalación y uso del producto (CLI + Web) | Usuario final | 2026-05-14 | Medio |
| `rag/README.md` | Referencia técnica del pipeline RAG | Desarrollador | 2026-05-14 | Alto |
| `rag/web/frontend/README.md` | Comandos `npm` del frontend React | Desarrollador frontend | 2026-05-14 | Bajo |
| `research/README.md` | Mapa del workspace experimental TFG | Tutor / autor | 2026-05-14 | Medio |
| `research/conversion/GEMMA3_CONVERSION_ISSUE.md` | Post-mortem de Gemma-3 (abandonado) | Autor / tutor | 2026-05-14 | Bajo (congelado) |
| `research/docs/EVALUACIONES_PIPELINE.md` | Protocolo TFG de evaluación RAGAS | Tutor / autor | 2026-05-14 | Alto |
| `research/docs/PROJECT_LAYOUT.md` | Árbol top-level del repo | Cualquiera | 2026-05-14 | Medio |
| `research/docs/USER_SPARSE_CHECKOUT.md` | Cómo clonar sólo `rag/` con sparse checkout | Usuario final | 2026-05-14 | Bajo |
| `research/docs/REINFERENCIA_FINAL.md` | Protocolo de segunda pasada de inferencia | Autor | 2026-05-14 | Medio |
| `research/docs/DOCS_AUDIT.md` | Este índice | Autor | 2026-05-14 | — |
| `research/models/gguf/gemma-3/README.md` | Estado final de Gemma-3 (abandonado) | HF Hub / tutor | 2026-05-14 | Bajo |
| `research/models/gguf/phi-4/CONVERSION.md` | Checklist merge LoRA → GGUF → Ollama | Autor | 2026-05-14 | Bajo |
| `research/models/gguf/phi-4/README.md` | Model card Phi-4 RAG fine-tuned | HF Hub / tutor | 2026-05-14 | Medio |
| `research/models/gguf/qwen-3/CONVERSION.md` | Checklist merge LoRA → GGUF → Ollama | Autor | 2026-05-14 | Bajo |
| `research/models/gguf/qwen-3/README.md` | Model card Qwen3-14B RAG fine-tuned | HF Hub / tutor | 2026-05-14 | Medio |

## Historial de borrados (2026-05-14)

- `research/docs/palabras.md` — formulario administrativo TFG, fuera del repo.
- `research/docs/investigacionMetricas.md` — ya plasmado en la memoria del TFG.
- `research/docs/splits.md` — ya plasmado en la memoria del TFG.

## Cómo usar este índice

Cuando cambies código (signaturas en `rag/chat_pdfs.py`, CLIs en
`research/evaluation/*.py`, métricas en `training-output/`), revisa la columna
**Riesgo de drift**:

- **Alto** → el doc casi seguro necesita actualización.
- **Medio** → revisar comandos/paths que toquen tu cambio.
- **Bajo** → improbable, pero un vistazo no cuesta nada.

Actualiza la columna **Última verificación** al terminar.
