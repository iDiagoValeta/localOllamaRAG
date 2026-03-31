# Propuesta TFG: Campos para la plataforma UPV
 
## Titulación
Grado en Ingeniería Informática
 
## Título del trabajo
 
**Castellano:**
Adaptación de modelos de lenguaje de pesos abiertos multilingües para un sistema de generación aumentada por recuperación enfocado en documentos académicos

**Valenciano:**
Adaptació de models de llenguatge de pesos oberts multilingües per a un sistema de generació augmentada per recuperació enfocat en documents acadèmics
 
**Inglés:**
Fine-tuning of multilingual open-weight language models for a retrieval-augmented generation system focused on academic documents
 
## Resumen

El avance de los modelos de lenguaje ha impulsado herramientas como NotebookLM de Google, que permiten consultar documentos propios de forma conversacional. Sin embargo, estas soluciones comerciales imponen barreras económicas, dependen de servidores externos y presentan problemas de privacidad que las hacen inviables cuando se trabaja con material académico o datos confidenciales. Este trabajo propone MonkeyGrab, una alternativa completamente local basada en modelos de lenguaje de pesos abiertos, con soporte multilingüe para inglés, castellano y valenciano/catalán, que opera íntegramente sin ceder datos a terceros.

El trabajo se desarrolla en varias fases experimentales. En primer lugar, se evalúan y comparan diversos modelos base para identificar los candidatos más prometedores en la tarea. A continuación, se adaptan mediante adaptadores de bajo rango (LoRA) sobre un corpus multilingüe en inglés, castellano y valenciano/catalán, evaluando cada modelo antes y después del entrenamiento para cuantificar el efecto de la adaptación de forma aislada. En paralelo, se estudian y comparan diferentes técnicas RAG, como estrategias de indexación, recuperación híbrida y reordenación semántica, para determinar las configuraciones más adecuadas para el sistema. Sobre la base de todos estos resultados, se implementa un sistema que indexa documentos académicos, combina búsqueda semántica y léxica, reordena los fragmentos recuperados y genera respuestas fundamentadas exclusivamente en el contenido aportado por el usuario. Finalmente, el pipeline completo se evalúa desde múltiples perspectivas, incluyendo la calidad de la recuperación y de la generación en los tres idiomas de trabajo.

## Resum

L'avanç dels models de llenguatge ha impulsat ferramentes com NotebookLM de Google, que permeten consultar documents propis de forma conversacional. No obstant això, estes solucions comercials imposen barreres econòmiques, depenen de servidors externs i presenten problemes de privacitat que les fan inviables quan es treballa amb material acadèmic o dades confidencials. Este treball proposa MonkeyGrab, una alternativa completament local basada en models de llenguatge de pesos oberts, amb suport multilingüe per a l'anglés, el castellà i el valencià/català, que opera íntegrament sense cedir dades a tercers.

El treball es desenrotlla en diverses fases experimentals. En primer lloc, s'avaluen i comparen diversos models base per a identificar els candidats més prometedors en la tasca. A continuació, s'adapten mitjançant adaptadors de baix rang (LoRA) sobre un corpus multilingüe en anglés, castellà i valencià/català, avaluant cada model abans i després de l'entrenament per a quantificar l'efecte de l'adaptació de forma aïllada. En paral·lel, s'estudien i comparen diferents tècniques RAG, com ara estratègies d'indexació, recuperació híbrida i reordenació semàntica, per a determinar les configuracions més adequades per al sistema. Sobre la base de tots estos resultats, s'implementa un sistema que indexa documents acadèmics, combina cerca semàntica i lèxica, reordena els fragments recuperats i genera respostes fonamentades exclusivament en el contingut aportat per l'usuari. Finalment, el pipeline complet s'avalua des de múltiples perspectives, incloent-hi la qualitat de la recuperació i de la generació en els tres idiomes de treball.

## Abstract

The rapid progress of language models has driven the development of tools such as Google’s NotebookLM, which allow users to query their own documents conversationally. However, these commercial solutions impose economic barriers, depend on external servers, and raise privacy concerns that make them unsuitable for working with academic material or confidential data. This work proposes MonkeyGrab, a fully local alternative based on open-weight language models, with multilingual support for English, Spanish, and Valencian/Catalan, operating entirely without transferring data to third parties.

The work is structured into several experimental phases. First, various base models are evaluated and compared to identify the most promising candidates for the task. Next, they are adapted using Low-Rank Adapters (LoRA) on a multilingual corpus in English, Spanish, and Valencian/Catalan, and each model is evaluated before and after training to quantify the effect of the adaptation in isolation. In parallel, different RAG techniques are studied and compared, such as indexing strategies, hybrid retrieval, and semantic reranking, to determine the most suitable configurations for the system. Based on all these results, a system is implemented that indexes academic documents, combines semantic and lexical search, reranks retrieved fragments, and generates responses grounded exclusively in the user’s documents. Finally, the complete pipeline is evaluated from multiple perspectives, including retrieval and generation quality across the three working languages.
 
## Tutor/a
Adrià Giménez Pastor
 
## Departamento de adscripción del tutor
DSIC
 
## Estimación de horas
200-300
 
## Palabras clave
 
`Large Language Models` · `Retrieval-Augmented Generation` · `Hybrid Retrieval` · `LoRA` · `Fine-tuning` · `Multilingual NLP` · `Open-weight models`