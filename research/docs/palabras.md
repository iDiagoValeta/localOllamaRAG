# Propuesta TFG: Campos para la plataforma UPV
 
## Titulación
Grado en Ingeniería Informática
 
## Título del trabajo
 
**Castellano:**
Adaptación de modelos de lenguaje de pesos abiertos para un sistema multilingüe local de generación aumentada por recuperación
 
## Resumen

El avance de los modelos de lenguaje ha impulsado herramientas como NotebookLM de Google, que permiten consultar documentos propios de forma conversacional. Sin embargo, estas soluciones comerciales imponen barreras económicas, dependen de servidores externos y presentan problemas de privacidad que las hacen inviables cuando se trabaja con material académico o datos confidenciales. Este trabajo propone MonkeyGrab, una alternativa completamente basada en modelos de lenguaje de pesos abiertos, con soporte multilingüe para inglés, castellano y valenciano/catalán, y que operará íntegramente en local sin ceder datos a terceros.

El trabajo se desarrollará en varias fases experimentales. En primer lugar se evaluarán y compararán diversos modelos base para identificar los candidatos más prometedores en la tarea. A continuación, se adaptarán mediante adaptadores de bajo rango (LoRA) sobre un corpus multilingüe en inglés, castellano y valenciano/catalán, evaluando cada modelo antes y después del entrenamiento para cuantificar el efecto de la adaptación de forma aislada. Sobre la base de estos resultados, se implementará un sistema para la generación de respuestas fundamentadas exclusivamente en el contenido aportado por el usuario. Para ello se estudiará la integración en MonkeyGrab de diferentes técnicas de recuperación de información, tales como indexación de documentos, búsqueda semántica y léxica, o reordenación de fragmentos recuperados. Finalmente, el pipeline completo se evaluará desde múltiples perspectivas, incluyendo la calidad de la recuperación y de la generación en los tres idiomas de trabajo.
 
## Tutor/a
Adrià Giménez Pastor
 
## Departamento de adscripción del tutor
DSIC
 
## Estimación de horas
330
 
## Palabras clave
 
Modelos de lenguaje grandes abiertos; Generación Aumentada por Recuperación; Aprendizaje profundo; LoRA; Fine-tuning; Multilingüe
