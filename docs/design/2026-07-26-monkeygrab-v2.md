# Diseño actual de MonkeyGrab

Este documento describe la arquitectura que ejecutan la aplicación y su
evaluación. No es un plan de migración.

## Objetivos

MonkeyGrab responde preguntas sobre PDFs sin enviar el corpus a servicios
externos. Debe recuperar texto, tablas e imágenes, mantener citas a la página
original y medir regresiones con respuestas producidas por el sistema real.

La aplicación tiene una sola composición de indexación y recuperación. La CLI,
la interfaz web, la aplicación de escritorio y el gate de evaluación comparten
los mismos casos de uso y adaptadores.

## Composición multimodal

La ruta de producción combina cuatro tecnologías:

1. [MinerU](https://github.com/opendatalab/MinerU) extrae el orden de lectura,
   el texto, las tablas estructuradas y las figuras del PDF.
2. [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) representa texto
   e imágenes en el mismo espacio semántico. MonkeyGrab usa vectores
   Matryoshka de 512 dimensiones.
3. [FAISS](https://github.com/facebookresearch/faiss) persiste los vectores y
   ejecuta búsqueda exacta. Los vectores se normalizan y se comparan mediante
   producto interno, equivalente aquí a similitud coseno.
4. [BGE Reranker v2 M3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
   reordena los candidatos leyendo conjuntamente la pregunta y cada fragmento.

No existen selectores de extractor, modelo de embeddings, almacén vectorial o
reranker. Cambiar una de estas piezas exige una decisión de arquitectura, una
evaluación completa y una modificación del punto de composición.

Los pesos descargables de Jina CLIP v2 se distribuyen bajo
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Esta
composición es válida para el uso no comercial del proyecto. Un despliegue
comercial debe obtener otra licencia o sustituir el modelo tras medir la nueva
composición.

## Indexación

MinerU produce texto estructurado por página y conserva las tablas como HTML.
Los fragmentos de tabla se etiquetan como `table`; el resto del contenido
textual como `markdown`.

Las figuras y gráficos se extraen como imágenes. Jina CLIP los representa
directamente junto con su leyenda, si existe. No se genera una descripción
intermedia con un modelo de visión.

El enriquecimiento contextual es opcional. Cuando está activo, un modelo
Ollama antepone a cada fragmento textual un resumen situacional. Las tablas
siguen siendo recuperables como contenido estructurado y las imágenes mantienen
su vector visual.

Cada corpus de idioma tiene su propio índice FAISS. Los índices son
reproducibles y no se versionan.

## Recuperación y respuesta

La pregunta se descompone en variantes cuando corresponde. Cada variante se
busca en el espacio compartido de Jina CLIP y, si la búsqueda híbrida está
activa, también mediante BM25.

Reciprocal Rank Fusion combina ambas ramas. BGE vuelve a puntuar los candidatos,
se aplica el umbral de relevancia y pueden añadirse fragmentos vecinos.
Opcionalmente RECOMP sintetiza la evidencia antes de la generación.

Ollama se usa para conversación, descomposición de consultas, enriquecimiento
contextual, síntesis RECOMP y respuesta final. No se usa para extraer PDFs,
generar embeddings, buscar vectores ni describir figuras.

## Arquitectura hexagonal

`src/monkeygrab/` contiene dominio, puertos, configuración, casos de uso y
adaptadores. `rag/` contiene las interfaces y el cableado.

La regla de dependencias es unidireccional:

- `application` puede importar `domain`, `ports` y `config`.
- `ports` puede importar `domain`.
- `domain` y `config` no importan otras capas internas.
- `adapters` implementa puertos, pero ninguna capa interior importa
  adaptadores concretos.

`tests/unit/test_architecture_boundaries.py` comprueba esta regla analizando los
imports. `monkeygrab.composition` es el único punto que construye MinerU, Jina
CLIP y FAISS.

Los tres casos de uso son:

- `IndexCorpus`, para extracción, fragmentación e indexación multimodal.
- `Retrieve`, para recuperación semántica, BM25, fusión y reranking.
- `Answer`, para seleccionar evidencia y generar la respuesta.

Todas las interfaces llaman a estos casos de uso mediante `rag/engine/`.

## Aislamiento de dependencias

MinerU se ejecuta como herramienta externa. Jina CLIP se carga en el intérprete
aislado `.venv-mineru` y se comunica con la aplicación mediante un proceso
persistente con mensajes JSON por línea.

Este límite evita mezclar versiones incompatibles de Torch y Transformers con
el entorno principal. La carga del modelo se paga una vez y el worker se
reutiliza durante la sesión.

## Política de fallos

Los adaptadores fallan de forma visible. No cambian de backend, modelo o
dispositivo en silencio.

La extracción de imágenes es la excepción limitada: una figura corrupta puede
omitirse sin perder el texto completo del documento. Un fallo global de MinerU,
Jina CLIP, FAISS o BGE deja la operación inconclusa.

Los fallos de infraestructura durante la evaluación no cuentan como respuestas
correctas y no actualizan el baseline.

## Evaluación y CI

El gate rápido se ejecuta en cada pull request y push a `main`. Comprueba lint,
límites arquitectónicos, pruebas unitarias, caracterización y compilación del
frontend sin descargar modelos.

El gate completo se lanza manualmente en un runner propio con GPU y Ollama.
Ejecuta la composición real sobre casos verificados del corpus de desarrollo y
un conjunto ciego de artículos de arXiv. Incluye preguntas factuales,
recuperación de figuras y recuperación de tablas.

El grader es determinista. El resultado solo es comparable con el baseline
cuando usa el modelo generador calibrado y completa todos los casos. El baseline
actual es `0.77`; solo puede subir mediante la opción explícita del runner.

Este sistema es un ratchet de medición, no un optimizador autónomo: detecta
regresiones y registra mejoras, pero no modifica código ni prompts por sí solo.

## Decisiones vigentes

- Una sola composición de producción.
- Búsqueda FAISS exacta para los corpus actuales.
- Vectores Jina CLIP truncados a 512 dimensiones.
- Tablas indexadas como contenido estructurado, no como capturas.
- Figuras indexadas por sus píxeles, sin descripción generada.
- BGE Reranker v2 M3 como único reranker.
- Sin fallbacks silenciosos entre tecnologías.
- Baselines ligados a una configuración comparable.
