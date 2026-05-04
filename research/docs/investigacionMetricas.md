# Justificación de las métricas de evaluación (TFG)

El trabajo compara modelos de lenguaje en tareas de respuesta condicionada a documento. El contexto se toma de los propios conjuntos de evaluación (contexto de referencia), no del recuperador del sistema RAG, para que los resultados reflejen sobre todo la **calidad de la generación** y el **anclaje al texto aportado**, sin mezclar en esa fase los errores del módulo de recuperación. En la literatura este acotamiento suele describirse como evaluación con **pasajes de referencia** o **contexto oráculo**, o como medición del comportamiento del **lector** en un esquema recuperador–lector.

---

## Métricas por tipo de evaluación

| Evaluación | Métricas |
|---|---|
| **Modelos base** (varios LLM, mismo protocolo) | Token F1; ROUGE-L F1; BERTScore (P, R, F1); longitud media de respuesta †; proporción de respuestas que terminan en frase completa (`.`, `!`, `?`) †. |
| **Modelo base frente al mismo modelo adaptado** (fine-tuning para RAG) | Las mismas que arriba, calculadas en particiones **desarrollo** y **prueba** fijas, para cuantificar el efecto del entrenamiento sin cambiar el protocolo de contexto. |
| **Sistema RAG completo** (recuperación + generación) | RAGAS (p. ej. fidelidad y relevancia respecto al contexto recuperado mediante verificación asistida por modelo) y **BERTScore** (P, R, F1) frente a la respuesta de referencia, para alinear la evaluación *end-to-end* con la dimensión semántica ya usada en las fases con contexto de referencia. |

Las dos primeras filas responden a la misma pregunta de investigación (¿cómo mejora la adaptación la generación anclada al documento?); la tercera responde a otra (¿cómo se comporta el sistema en condiciones reales de uso con recuperación).

† Indicadores auxiliares de formato y verbosidad; no participan en comparaciones principales ni en las tablas de resultados del TFG.

---

## Fórmulas y cálculo (definición operativa en este TFG)

### 1) Token F1

Token F1 es el estándar heredado de SQuAD para medir solapamiento léxico predicción–referencia en QA extractiva (Rajpurkar et al., 2016).

Para cada ejemplo, se normalizan predicción y referencia (minúsculas, eliminación de puntuación y de artículos frecuentes en EN/ES/CA) y se tokenizan por espacios. Con los multiconjuntos de tokens de predicción $T_p$ y referencia $T_r$:

$$
\mathrm{Precisión} = \frac{|T_p \cap T_r|}{|T_p|},
\qquad
\mathrm{Recall} = \frac{|T_p \cap T_r|}{|T_r|}
$$

$$
\mathrm{F1} = \frac{2 \cdot \mathrm{Precisión} \cdot \mathrm{Recall}}{\mathrm{Precisión} + \mathrm{Recall}}
$$

Si no hay solapamiento, F1 = 0; si ambas secuencias quedan vacías tras normalizar, F1 = 1. El valor reportado en tablas es la media por conjunto, expresada en porcentaje. En estos conjuntos se usa una referencia por ejemplo; si hubiera múltiples referencias, se tomaría el máximo F1 por ejemplo (criterio habitual en SQuAD).

### 2) BERTScore (P, R, F1)

BERTScore se propuso específicamente como métrica para evaluación de generación de texto y mostró mejor correlación con juicios humanos que métricas basadas en n-gramas como BLEU (Zhang et al., 2020).

Para cada par predicción-referencia se obtienen embeddings contextuales por token y se calcula similitud coseno. Si $\hat{y}$ es la predicción y $y$ la referencia:

$$
P_{\mathrm{BERT}} = \frac{1}{|\hat{y}|} \sum_{i \in \hat{y}} \max_{j \in y} \cos(\mathbf{h}_i, \mathbf{h}_j),
\qquad
R_{\mathrm{BERT}} = \frac{1}{|y|} \sum_{j \in y} \max_{i \in \hat{y}} \cos(\mathbf{h}_j, \mathbf{h}_i)
$$

$$
F1_{\mathrm{BERT}} = \frac{2 \cdot P_{\mathrm{BERT}} \cdot R_{\mathrm{BERT}}}{P_{\mathrm{BERT}} + R_{\mathrm{BERT}}}
$$

En la implementación se usa `microsoft/deberta-xlarge-mnli`, con `rescale_with_baseline=True` y sin ponderación IDF explícita (equivalente a `idf=False` en la librería). Se calcula por ejemplo y después se promedia por conjunto (en %).

### 3) ROUGE-L F1

ROUGE-L (Lin, 2004) mide el solapamiento entre predicción y referencia a través de la subsecuencia común más larga (*Longest Common Subsequence*, LCS), capturando tanto coincidencia de tokens como preservación del orden relativo sin exigir contigüidad. Sea $c$ la predicción (candidato) de longitud $|c|$ y $r$ la referencia de longitud $|r|$:

$$
P_{L} = \frac{\mathrm{LCS}(c,\,r)}{|c|}, \qquad R_{L} = \frac{\mathrm{LCS}(c,\,r)}{|r|}
$$

$$
F1_{L} = \frac{2 \cdot P_{L} \cdot R_{L}}{P_{L} + R_{L}}
$$

donde $\mathrm{LCS}(c,r)$ es la longitud en tokens de la subsecuencia común más larga. La tokenización y normalización siguen el mismo preprocesado que en Token F1 (minúsculas, eliminación de puntuación y artículos frecuentes en EN/ES/CA). El valor reportado es la media aritmética por conjunto, expresada en porcentaje.

---

## Nivel de cálculo: media por muestra

Las tres métricas cuantitativas (Token F1, ROUGE-L F1 y BERTScore) se calculan **por ejemplo** y el resultado reportado en tablas es la **media aritmética** de esos valores individuales, expresada en porcentaje. Esta elección de media de métricas por muestra, y no una métrica agregada sobre el corpus completo es el estándar en QA generativa con conjuntos de referencia y sigue el esquema usado en SQuAD (Rajpurkar et al., 2016). Una consecuencia directa de este esquema es que cada ejemplo contribuye por igual a la media, **independientemente de la longitud de la respuesta**: una respuesta corta y una larga tienen el mismo peso en el promedio final. Esto vale tanto para Token F1 como para ROUGE-L F1 y BERTScore.

---

## Por qué estas métricas y no otras

**Token F1** es el estándar heredado de SQuAD para solapamiento léxico predicción–referencia en QA extractiva (Rajpurkar et al., EMNLP 2016) y permite comparación directa con esa tradición de evaluación.

**ROUGE-L F1** (Lin, 2004) es la variante de ROUGE basada en la subsecuencia común más larga y es métrica de referencia en evaluación de resumen automático y generación de texto. A diferencia de ROUGE-N (que mide solapamiento de n-gramas contiguos), ROUGE-L captura coherencia en el orden sin exigir contigüidad, lo que la hace más flexible para respuestas de longitud variable. Aparece explícitamente en los artículos: Pointer-Generator Networks (See et al., ACL 2017) la usa como métrica principal junto a ROUGE-1/2 y METEOR, y Newsroom (Grusky et al., NAACL 2018) reporta las variantes F1 de ROUGE-1/2/L como estándar de evaluación. En este TFG, ROUGE-L F1 complementa Token F1 añadiendo la dimensión de orden y continuidad textual, y alinea la evaluación con esa tradición de summarization.

**BERTScore** mide similitud semántica contextual entre generación y referencia y fue propuesta específicamente para evaluar generación de texto, mostrando mejor correlación con juicios humanos que métricas de n-gramas como BLEU (Zhang et al., ICLR 2020). Al operar con embeddings contextuales, compensa la rigidez de Token F1 y ROUGE-L cuando el estilo de respuesta es más libre. Se usa como backbone `microsoft/deberta-xlarge-mnli`, recomendado en el repositorio oficial de BERTScore como "currently, the best model" por su mayor correlación con evaluación humana frente a otras configuraciones disponibles (Tiiiger, BERTScore GitHub).

**Longitud media** y **cierre de frase** son indicadores auxiliares de formato y verbosidad; no sustituyen a las métricas anteriores pero ayudan a interpretar cambios de estilo tras el fine-tuning.

**BLEU** (Papineni et al., ACL 2002) se diseñó originalmente para traducción automática con múltiples referencias y salidas relativamente cercanas a la referencia. En tareas de generación más libres, como simplificación con división de oraciones, se ha mostrado que BLEU puede correlacionar débilmente o incluso negativamente con la simplicidad y dejar de ser informativo como métrica de calidad (Sulem et al., EMNLP 2018). En QA generativa y RAG, donde suele haber una única referencia y la formulación puede variar mucho, estas limitaciones lo hacen poco adecuado como métrica principal, por lo que en este trabajo no se emplea BLEU.

**METEOR** (Banerjee & Lavie, 2005) fue diseñada para traducción automática como alternativa a BLEU: considera stems y sinónimos, y aplica una penalización por fragmentación para capturar fluidez. Varios trabajos de resumen, incluido Pointer-Generator (See et al., ACL 2017), la reportan junto a ROUGE. No obstante, en este TFG no se incluye como métrica principal por dos motivos: (1) en el plano léxico, Token F1 y ROUGE-L F1 cubren ya la comparación con los artículos de referencia; (2) en el plano semántico, BERTScore ofrece mayor correlación con juicios humanos gracias a las representaciones contextuales modernas, que es el papel que METEOR buscaba originalmente. METEOR se menciona en el marco teórico como parte del conjunto clásico de métricas de generación, pero no forma parte del conjunto de evaluación experimental del TFG.

**COMET** (Rei et al., EMNLP 2020) está diseñada para evaluación de traducción con un esquema fuente–hipótesis–referencia que no encaja de forma natural en síntesis documental monolingüe; su uso fuera de MT no tiene todavía una tradición consolidada para este tipo de sistema (Zouhar et al., WMT 2024, sobre limitaciones y usos indebidos de COMET).

---

## Referencias (orden alfabético por apellido del primer autor)

- Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization 2005*. [https://aclanthology.org/W05-0909/](https://aclanthology.org/W05-0909/)
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024* (demos). [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
- Grusky, M., Naaman, M., & Artzi, Y. (2018). Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies. *NAACL-HLT 2018*. [https://arxiv.org/abs/1804.11283](https://arxiv.org/abs/1804.11283)
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop on Text Summarization Branches Out 2004*. [https://aclanthology.org/W04-1013/](https://aclanthology.org/W04-1013/)
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL 2002*. [https://aclanthology.org/P02-1040/](https://aclanthology.org/P02-1040/)
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP 2016*. [https://arxiv.org/abs/1606.05250](https://arxiv.org/abs/1606.05250)
- Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A Neural Framework for MT Evaluation. *EMNLP 2020*. [https://arxiv.org/abs/2009.09025](https://arxiv.org/abs/2009.09025)
- See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL 2017*. [https://arxiv.org/abs/1704.04368](https://arxiv.org/abs/1704.04368)
- Sulem, E., Abend, O., & Rappoport, A. (2018). BLEU is Not Suitable for the Evaluation of Text Simplification. *EMNLP 2018*. [https://arxiv.org/abs/1810.05995](https://arxiv.org/abs/1810.05995)
- Tiiiger. (2019-2025). BERTScore: BERT score for text generation. GitHub repository. [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*. [https://arxiv.org/abs/1904.09675](https://arxiv.org/abs/1904.09675)
- Zouhar, V., Chen, P., et al. (2024). Pitfalls and Outlooks in Using COMET. *WMT 2024 Metrics Shared Task*. [https://arxiv.org/abs/2408.15366](https://arxiv.org/abs/2408.15366)
