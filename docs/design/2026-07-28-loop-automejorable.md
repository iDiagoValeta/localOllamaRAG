# Loop automejorable sobre el pipeline RAG

Diseño validado el 2026-07-28. Define un arnés que busca automáticamente
configuraciones mejores del pipeline, y los criterios que dicen si está
conseguido.

Antecedente: la auditoría de `8f83c5f` (migración multimodal) encontró que tres
de sus siete hallazgos son defectos del **medidor**, no del runtime: índice que
no se invalida, baseline sin evidencia versionada, métrica de figuras que puntúa
recuperación y se lee como respuesta. Este diseño los absorbe como criterios de
aceptación en vez de tratarlos como deuda aparte.

---

## 1. Objetivo

Un arnés que, partiendo de una configuración dada, busca configuraciones mejores
del pipeline, donde **mejor** significa más casos gold acertados sin empeorar la
latencia por encima de un techo, y donde cada candidato queda registrado de forma
que se pueda reconstruir exactamente.

**Función objetivo:** maximizar la tasa de acierto sobre el conjunto de búsqueda,
sujeta a una restricción dura de latencia. No es un escalar combinado: no se fija
ningún tipo de cambio entre puntos de acierto y minutos. Un candidato que se pasa
del techo queda descalificado aunque acierte más.

**Espacio de acción, primera etapa:** sólo configuración, es decir flags de pipeline,
tamaño de fragmento, top-k, pesos de fusión, parámetros BM25, prompts. Todo lo
que `AppConfig` ya modela como objeto inmutable y serializable. El espacio se
amplía a código cuando el arnés haya demostrado que mide bien, no antes.

> [!IMPORTANT]
> El orden no es cautela decorativa. Un agente que edita código contra una
> función de fitness defectuosa optimiza contra los defectos del medidor antes
> que contra el problema, y lo hace con confianza.

---

## 2. Criterios de aceptación

Los cuatro primeros son del medidor. Un loop sobre una medida que miente no
produce mejoras: produce basura reproducible.

| # | Criterio | Cómo se comprueba |
|---|---|---|
| 1 | **Repetibilidad** (medido 2026-07-29, ver nota abajo) | La misma configuración corrida dos veces aprueba el mismo conjunto de casos. Si no, la dispersión observada queda declarada como suelo de ruido y ningún delta por debajo cuenta como mejora. |
| 2 | **Sensibilidad** (medido 2026-07-29, ver nota abajo) | Un sabotaje conocido y acotado (recuperar un solo fragmento; apagar el reranker) hace caer el gate de forma inequívoca. |
| 3 | **No engañable** (evidencia de campo 2026-07-29, ver nota abajo) | Alterar el troceado obliga a reindexar en el run siguiente, en vez de reutilizar el índice por nombre de fichero. |
| 4 | **Métricas separadas** | Un caso de figura que acierta la recuperación no cuenta como respuesta correcta. |
| 5 | **Búsqueda efectiva** | Partiendo de una configuración deliberadamente empeorada, el loop recupera al menos hasta el rendimiento actual sin intervención. |
| 6 | **Techo respetado** | Un candidato que acierta más pero excede la latencia se rechaza, y el informe registra el motivo. |
| 7 | **Reproducibilidad** | Con el informe de una iteración se reconstruye esa configuración y se obtiene el mismo resultado, dentro del suelo de ruido del criterio 1. |
| 8 | **Terminación** | Para por presupuesto o por falta de progreso, y en ambos casos deja informe. |

El criterio 5 se corre con dos proponentes (ver §4) para saber si el razonamiento
del LLM aporta algo sobre un control determinista.

> [!NOTE]
> **Criterio 1, medido el 2026-07-29.** Dos runs del gate completo, misma
> configuración y mismo código, mismo índice (los dos registraron `cache hit`
> en el conjunto dev y en el ciego, así que ninguno reindexó):
> `tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json` y
> `tests/eval/runs/20260729T040824Z_mineru-jina_clip-faiss.json`. Son
> artefactos locales: `tests/eval/runs/` está en `.gitignore`, así que nadie
> que clone el repo puede verificarlos por sí mismo. Ambos 44/51 = 0.8627
> global. `compare_runs.py` sobre el par: `identical: 51 case(s) unchanged`,
> delta de tasa de acierto +0.0000, cero vuelcos. `compare_runs.py` compara el
> vector de aprobado/fallado por caso, no el texto generado, así que eso acota
> la **clasificación**, no la salida del generador. Prueba directa en el
> propio par: `planck-sigma8-es` falla en los dos runs y, al fallar, guarda la
> respuesta generada -- los dos textos difieren (uno termina en "Planck
> lensing", el otro añade una frase completa sobre las preferencias de
> amplitud de Planck). El generador corre a temperatura 0.15 y varió, como
> cabía esperar; lo medido es que el criterio de acierto de `grade.py` absorbe
> esa variación, no que la salida fuera idéntica. Eso acota el suelo de ruido
> de la clasificación por debajo de un caso; no demuestra que el pipeline sea
> determinista en general, y dos runs es una muestra pequeña, así que una
> afirmación más amplia exige más pares. El suelo queda medido para este
> `grade.py`: un cambio en las reglas de puntuación puede moverlo y exigiría
> remedirlo. Bajo este suelo, un vuelco de un solo caso deja de ser explicable
> por ruido -- lo cual no es lo mismo que una diferencia entre dos
> configuraciones sea demostrable: la sección 3 sitúa ese segundo umbral en
> unos seis vuelcos netos, con un margen aprovechable de unos cinco casos
> descontadas las figuras. Esta nota no retracta esa cuenta, sólo fija el
> suelo bajo el que se interpreta; la inferencia, además, descansa en un único
> par de runs.
>
> **Criterio 2, medido el 2026-07-29.** Un run degradado a propósito
> (`RAG_TOP_K_FINAL=1`, recuperación devuelve un solo fragmento) comparado
> contra el run sano del mismo par que mide el criterio 1: sano
> `tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json` (44/51 =
> 0.8627), degradado
> `tests/eval/runs/20260729T081129Z_mineru-jina_clip-faiss.json` (39/51 =
> 0.7647, 121.1 min). Los dos registraron `cache hit`, lo cual acota el
> índice, no el código: que entre ambos runs sólo medie un commit de
> documentación es cierto, pero nada citado aquí lo respalda, y ningún informe
> registra el valor de `RAG_TOP_K_FINAL` con el que corrió cada uno, así que la
> configuración degradada queda afirmada, no capturada -- exactamente el hueco
> que el libro de evidencias (§4) y el criterio 7 existen para cerrar. Son
> artefactos locales, igual que en la nota del criterio 1:
> `tests/eval/runs/` está en `.gitignore`, nadie que clone el repo puede
> verlos directamente. `compare_runs.py` sobre el par: **2 vuelcos a PASS, 7
> vuelcos a FAIL, 42 sin cambio, delta de tasa de acierto -0.0980**. Los siete
> que empeoraron son todos casos de sólo recuperación: `att-arch-figure`,
> `att-arch-figure-es`, `att-bleu-table`, `dpo-pipeline-figure`,
> `resnet-block-figure`, `vit-comparison-table`, `vit-overview-figure`. Los
> dos que mejoraron: `planck-sigma8-es`, `resnet-top1-34layer`. Por métrica:
> recuperación sola pasa de 11/15 (0.7333) a 4/15 (0.2667); respuesta pasa de
> 33/36 (0.9167) a 35/36 (0.9722). El run degradado también incumple el suelo
> de `baseline_min_pass_rate.txt` (0.7647 < 0.77), que es el gate
> comportándose correctamente ante una degradación conocida -- pero por un
> margen de 0.0053, frente a los 0.0196 que vale un solo caso: con 40/51 =
> 0.7843 (recuperación en 5/15 en vez de 4/15, igual de hundida) el gate
> habría aprobado la misma degradación catastrófica. El agregado cayó 0.098
> mientras que la recuperación cayó 0.47; es la evidencia más nítida del punto
> que ya hace el párrafo de consecuencia más abajo: un agregado que apenas
> nota un desplome de recuperación de esta magnitud es justo lo que vuelve
> peligroso a un loop que lo maximiza.
>
> Esta medición sólo es interpretable porque el suelo de ruido que fija la
> nota del criterio 1, con el mismo par de runs sanos, es cero vuelcos. Siete
> vuelcos contra un suelo de cero es señal inequívoca; contra un suelo
> distinto de cero habría que descontarlo antes de leer nada.
>
> Recuperación y respuesta se movieron en direcciones opuestas: recuperación
> se desploma con 7 vuelcos -- por encima del umbral de unos seis vuelcos
> netos que la nota del criterio 1 cita (fijado en la sección 3) para que una
> diferencia sea demostrable --, mientras que respuesta mejora con sólo 2
> vuelcos netos: por encima del suelo de ruido de cero pero por debajo de ese
> mismo umbral, así que esta medición no demuestra que la mejora en respuesta
> sea real. Una lectura plausible es que menos contexto distrae menos en
> preguntas factuales concretas -- pero eso es una hipótesis, no un hallazgo:
> nada en esta medición la puso a prueba.
>
> Se probó una única configuración degradada (`RAG_TOP_K_FINAL=1`), no un
> barrido; el otro sabotaje que enumera el criterio (apagar el reranker) sigue
> sin medir. Esta nota usa la separación entre recuperación y respuesta que ya
> reporta `run_eval.py`, pero mide sólo el criterio 2: no declara cerrado el
> criterio 4 por usarla.
>
> Consecuencia para el arnés: la función objetivo de la sección 1 maximiza un
> único número agregado. Un loop que maximizara sólo ese agregado podría
> favorecer configuraciones que cambian calidad de recuperación por acierto
> factual sin que nadie lo note -- el agregado por sí solo no distingue esa
> compensación de una mejora real. Las métricas separadas del criterio 4 son
> las que hacen visible el intercambio; hoy la función objetivo del diseño
> sigue apuntando al agregado. Cinco de los siete vuelcos a FAIL son casos de
> figura, y la sección 3 (Margen inalcanzable) ya deja parte de esos casos
> fuera del escalar que el loop maximiza -- lo que hace la advertencia más
> fuerte, no menos.
>
> **Criterio 3, evidencia de campo del mismo par de runs.** Los dos runs
> registraron `cache hit` en ambos corpus; la línea `chunks={store.count()}`
> que sigue a cada cache hit en el log es la que reportó 884 y 218 fragmentos
> respectivamente -- el mensaje de cache hit en sí no cuenta fragmentos. Con
> la configuración sin cambios, un cache hit ejerce la ruta de
> **coincidencia** de la huella del índice, no la ruta de **detección** que
> pide el criterio 3 (que alterar el troceado fuerce un reindexado en el run
> siguiente). Es evidencia de que la huella no invalida en falso un índice
> real, cobertura que hasta ahora sólo daban los dobles de prueba; no
> ejercita el camino que el criterio exige. No se marca el criterio como
> cerrado por esto solo; sólo añade esa evidencia de campo a la que ya daban
> los tests unitarios.

---

## 3. El corpus, derivado del objetivo

### Por qué hay que ampliarlo

De los 51 casos actuales fallan 9. Ese es el margen de mejora completo: el loop
no puede ganar más de nueve casos porque no hay más que ganar.

La comparación entre dos configuraciones es **pareada** sobre los mismos casos:
se miran los casos que vuelcan de estado, no las tasas globales. Con estas
cantidades hacen falta del orden de seis vuelcos a favor sin ninguno en contra
para que la diferencia no sea atribuible al azar.

Nueve de margen, seis para demostrar algo. Y cuatro de esos nueve son casos de
figura, cuya métrica hoy mide sólo recuperación (criterio 4). El margen real en
el que el loop puede demostrar algo es de unos cinco casos.

> [!NOTE]
> El corpus crece porque el medidor necesita resolución, no por completitud
> temática. Ésa es la única justificación que fija el tamaño.

> [!CAUTION]
> **Medido el 2026-08-12: el conjunto de búsqueda no sólo tiene poco margen,
> es que tampoco detecta un sabotaje conocido.** La cuenta de arriba mira el
> gate entero. Lo que el loop optimiza es el conjunto de búsqueda, y ahí los
> números son peores.
>
> Los tres artefactos que ya citan las notas del criterio 1 y 2, cruzados por
> `id` contra `gold_cases.jsonl` (el campo `source` vive sólo en el fichero de
> casos, no en los registros del run), separan un conjunto de búsqueda de 32
> casos `source: corpus` sobre 6 papers y un ciego de 19 casos `source: arxiv`
> sobre 3. Esa partición ya es **por documento**, que es lo que pide la sección
> de Partición, así que sirve hasta que el bloque B entregue una propia.
>
> - Del run sano `20260729T020233Z`: búsqueda **27/32**, ciego 17/19. **Cinco
>   fallos disponibles en el conjunto de búsqueda**, no nueve. Tres de esos
>   cinco son casos de figura.
> - Sano contra sano, restringido al conjunto de búsqueda: **0 vuelcos**. El
>   suelo de ruido de la nota del criterio 1 se sostiene también aquí, que es
>   lo que hace legible el punto siguiente.
> - Sano contra el saboteado `RAG_TOP_K_FINAL=1`, restringido al conjunto de
>   búsqueda: **4 vuelcos a FAIL y 1 a PASS, 3 netos** -- frente a los 7 y 2
>   que da el gate entero.
>
> Tres vuelcos netos contra un umbral de demostrabilidad de unos seis, y la
> configuración saboteada colapsa la recuperación a un solo fragmento: es el
> cambio más destructivo que se ha probado. El conjunto de búsqueda está
> infrapotenciado frente a un efecto enorme y conocido, no sólo frente a
> mejoras sutiles.
>
> Son dos cotas distintas y conviene no confundirlas: cinco fallos acotan lo
> que el loop puede **ganar**; tres vuelcos frente a un sabotaje catastrófico
> acotan lo que el loop puede **ver**. La segunda es la que obliga a que el
> arnés declare su propio límite de resolución en cada informe, en vez de
> dejar la advertencia en un hilo de issues.
>
> Ninguno de los cinco fallos del conjunto de búsqueda es margen inalcanzable:
> los tres de figura fallan con `wanted one of ['image'], got ['text']`, es
> decir por **ranking**, y son casos `figure_retrieval` que nunca llaman al
> generador. Los otros dos son un número presente en el resumen del paper que
> no sobrevive a la respuesta. Todos caen dentro del espacio de acción de la
> primera etapa, así que la lista de casos inalcanzables nace vacía y eso es
> una afirmación con evidencia, no un hueco por rellenar.
>
> Artefactos locales, igual que en las notas anteriores: `tests/eval/runs/`
> está en `.gitignore` y nadie que clone el repo puede verificarlos.

### Tamaño

Para que el loop demuestre varias mejoras sucesivas antes de agotarse, el pozo de
casos fallidos debe ser varias veces el umbral de detección: alrededor de treinta.

Los casos nuevos vienen de documentos más duros y fallarán más, así que se asume una
tasa de fallo del orden de cuatro de cada diez, frente a dos de cada diez en el
corpus actual. Con esa proporción:

- **~55 casos nuevos**, hasta un total en torno a **110**.
- **~20 documentos**, frente a los 9 actuales.
- ~6 casos por documento.

### Partición

Un loop es un optimizador: todo aquello contra lo que optimiza deja de ser una
muestra independiente. Si el loop usara el gate completo como función objetivo,
el conjunto ciego se consumiría en la primera iteración.

> [!WARNING]
> **Los tiempos de esta sección estaban mal por un factor de ~4.** Medido la
> noche del 2026-07-29 (issue #27): con `OLLAMA_KEEP_ALIVE=0` (el default del
> producto), Ollama descarta los pesos del generador tras cada llamada, así que
> cada caso factual paga una carga en frío completa en la fase de generación:
> 195 a 210 s por caso, de forma uniforme. La estimación original asumía
> ~32 s/caso, extrapolados de un informe (2026-07-27) cuya cifra por caso no
> aislaba generación de recuperación. Los números de la tabla y del diagrama
> siguientes quedan corregidos con ese factor ~4, aplicado de forma plana a
> las cifras originales del diseño -- incluidos los casos `figure_retrieval`
> y `table_retrieval`, que nunca llaman al generador, así que la cifra
> corregida es un límite superior, no una medición por tipo de caso. Son una
> corrección, no una medición directa de este conjunto de búsqueda (que aún
> no existe): la cifra real, una vez aplicado el Cambio 1 (keep-alive del
> generador acotado a la fase de evaluación, `tests/eval/run_eval.py`), queda
> pendiente de medir.

> [!NOTE]
> **Medido el 2026-08-12 (issue #54): el Cambio 1 sí surte efecto, y la
> corrección de arriba hay que deshacerla.** Run completo del gate sobre `main`
> en `8924110`, `gemma4:e2b` (el mismo modelo que los runs de referencia), los
> dos corpus con cache hit:
> `tests/eval/runs/20260812T194812Z_mineru-jina_clip-faiss.json`.
>
> | Bucket | 2026-07-29 | 2026-08-12 |
> |---|---|---|
> | Mediana de casos respondidos (n=36) | 201,0 s | **28,5 s** |
> | Mediana de sólo recuperación (n=15) | 4,1 s | 4,1 s |
> | Total | 124,0 min | **20,4 min** |
> | Casos `source: corpus` (32) | 78,9 min | **12,6 min** |
>
> La comprobación de coherencia es la segunda fila: los casos de sólo
> recuperación nunca llaman al generador, no tenían nada que ganar, y no
> ganaron nada. Toda la mejora cae exactamente donde la hipótesis de la carga
> en frío decía que caería, así que los 195-210 s que midió el issue #27 eran
> carga en frío y el keep-alive la elimina.
>
> Eso resuelve la ambigüedad que dejaba abierta la nota de arriba: los runs de
> referencia se lanzaron desde un árbol anterior a `d36c943`, pese a que el
> commit aterrizó cuatro minutos antes del primero.
>
> El resultado no se movió: `compare_runs.py` contra el run sano del
> 2026-07-29 da `identical: 51 case(s) unchanged`, delta +0.0000. Es además el
> tercer run sano consecutivo con cero vuelcos, lo que refuerza el suelo de
> ruido del criterio 1 más allá del único par sobre el que descansaba.
>
> Sigue sin registrarse el valor de `OLLAMA_KEEP_ALIVE` con el que corre cada
> run: es el mismo hueco que hizo esta pregunta irresoluble desde los
> artefactos, y el que el criterio 7 y el libro de evidencias existen para
> cerrar.

| Conjunto | Papel | Cuándo se toca | Tamaño |
|---|---|---|---|
| **Búsqueda** | Función objetivo del loop | Cada iteración | ~11 docs · ~60 casos · **~24 min** (extrapolado de los 12,6 min medidos sobre 32 casos) |
| **Validación** | Detecta sobreajuste al conjunto de búsqueda | Al cerrar una tanda | ~4 docs · ~22 casos |
| **Ciego** | Acepta una versión | Sólo para aceptar; nunca dentro del loop | ~5 docs · ~28 casos |

> [!IMPORTANT]
> Lo que el loop paga en cada iteración es el **conjunto de búsqueda**, no el
> corpus entero. Correr los 110 casos por iteración consumiría el conjunto ciego
> en la primera vuelta y lo convertiría en datos de entrenamiento.

> [!WARNING]
> La partición es **por documento, no por caso**. Dos preguntas sobre el mismo
> PDF comparten extracción, troceado y fragmentos; repartirlas entre búsqueda y
> validación filtra información de un lado al otro y la validación deja de
> validar.

### Composición

Tres ejes, escalonados. Un documento entra si produce **fallos informativos**: el
pipeline lo hace mal y podría plausiblemente hacerlo bien. Un documento que el
extractor destroza no aporta margen, aporta ruido que ninguna configuración
arregla.

| Eje | Qué estresa | Coste de aprovisionamiento |
|---|---|---|
| **Idioma** | Dos de los tres stores que se shipean, hoy sin medir | Cero: los PDFs ya están en `rag/docs/es/` y `rag/docs/ca/` |
| **Dominio** | Vocabulario no compartido con ML ni física; rama léxica | Bajo: arXiv cubre cualquier área |
| **Forma** | Extractor, troceado y ruta multimodal | Alto e incierto: escaneados y tablas densas con licencia abierta |

Las tablas van 5/5 porque las cinco son tablas académicas en LaTeX, donde MinerU
brilla. Un tipo de caso saturado no distingue mejora de empeoramiento: sólo cuesta
minutos por iteración. Documentos con tablas estadísticas densas devuelven
gradiente a ese tipo; hasta entonces, sale del nivel rápido.

### Sonda previa

Antes de invertir autoría humana en ~55 casos, un lote diagnóstico decide, por
eje, si produce fallos informativos o destructivos. Seis documentos, dos o tres
casos cada uno, indexados en **su propia colección aislada** para no alterar los
stores contra los que se mide el producto.

Los casos de la sonda se redactan para sobrevivir: si el eje resulta viable, son
la semilla de la fase completa, no material desechable.

Veredicto por eje, explícito: *viable* · *viable tras arreglo* · *inviable con
esta fuente*.

> [!NOTE]
> **Eje idioma, medido el 2026-08-13.** Primera corrida de
> `tests/eval/run_probe_lang.py --models gemma4:e2b` sobre `origin/main`
> (`5f136cb`). Colección aislada `probe_docs_lang` (426 fragmentos); no tocó
> `docs_es`/`docs_ca` ni `dev_docs`/`blind_docs`. Artefacto local y gitignored:
> `tests/eval/probe_runs_lang/20260813T213824Z_lang_probe.json`.
>
> | | |
> |---|---|
> | Total | **17/18 = 0.944** |
> | Castellano (`lang: es`, 9 casos, 3 docs) | 9/9 |
> | Valencià (`lang: ca`, 9 casos, 3 docs) | 8/9 |
> | Retrieval-only | 4/4 |
> | Answered | 13/14 |
> | Indexación (primera vez) | 630 s |
> | Wall | 20.1 min |
>
> El único fallo es `jaume-edat-mort`: el generador respondió que la
> introducción no especifica la muerte. El sidecar de la colección aislada
> *sí* guarda el literal en `jaume-conqueridor.pdf_pag0_chunk1` («regnà 58
> anys i morí a l'edat de 68 anys»), y la recuperación devolvió fragmentos
> de la página 0. Es un fallo de generación sobre evidencia recuperada, no
> de extracción ni de idioma — el mismo tipo de margen que la primera etapa
> del loop podría cerrar. Los tres `figure_retrieval` pasaron; sigue en pie
> la limitación ya anotada en esos casos: `grade.grade_retrieval` acepta
> cualquier hit `image` en el top-k, así que un PASS no prueba por sí solo
> que se recuperó el árbol y no una foto vecina.
>
> **Veredicto: *viable tras arreglo*.** El eje no es *inviable con esta
> fuente*: MinerU no destroza estos PDFs y el generador contesta en
> castellano y en valenciano. Tampoco es *viable* tal cual como pozo de
> fallos para las ~55 casos del bloque B. Diecisiete aciertos sobre
> dieciocho, contra la tasa de fallo de ~4/10 que la sección Tamaño asume
> para dimensionar el corpus, satura el medidor igual que las tablas LaTeX
> a 5/5 del párrafo anterior: un caso que casi siempre aprueba no distingue
> mejora de empeoramiento. El arreglo no es el pipeline ni el idioma; es la
> fuente. Estos seis artículos de Wikipedia ya en `rag/docs/es|ca/` cubren
> el hueco «dos stores que se shipean, hoy sin medir», pero no son el pozo.
> Hace falta documentos más duros en es/ca (o dejar este eje como cobertura
> de store y buscar el pozo en dominio/forma). Los 18 casos de la sonda no
> se promueven a `gold_cases.jsonl`: la autoría humana del diseño sigue
> pendiente, y promover un lote saturado gastaría minutos de cada iteración
> a cambio de casi ningún vuelco.
>
> **Eje dominio, medido el 2026-08-23.** Primera corrida de
> `tests/eval/run_probe_domain.py --models gemma4:e2b` sobre `main`
> (`c22a07d`). Colección aislada `probe_docs_domain` (278 fragmentos, tres
> papers recientes de econometría y biomatemáticas, léxico ajeno al corpus);
> no tocó los stores del producto ni del gate. Artefacto local y gitignored:
> `tests/eval/probe_runs_domain/20260823T133223Z_domain_probe.json`.
>
> | | |
> |---|---|
> | Total | **9/10 = 0.900** |
> | Retrieval-only (figura y tabla) | 2/2 |
> | Answered | 7/8 |
> | Indexación (primera vez, 3 PDFs) | 454 s |
> | Wall | 15.0 min |
>
> El único fallo es `singleton-dominates-tests`: pregunta conceptual cuya
> respuesta exige dos métodos aceptados y el generador no dio ninguno de los
> dos literales. Fallo de generación sobre evidencia recuperada, no de
> extracción ni de recuperación. La señal importante del eje es la ausencia
> de colapso léxico: 0.900 fuera de dominio contra 0.8824 del gate dentro de
> dominio el mismo día; extracción, troceado, recuperación y generación
> generalizan sin ajustes.
>
> **Veredicto: *viable tras arreglo*.** Como demostración de que arXiv
> cualquiera alimenta el pipeline, el eje es viable y cierra la duda de la
> tabla de ejes («Bajo: arXiv cubre cualquier área» queda confirmado). Pero
> como pozo de fallos hereda la misma advertencia de saturación que el eje
> idioma: 9/10 tampoco distingue mejora de empeoramiento, y el dominio por
> sí solo no fabrica la dificultad. La palanca está en la forma dentro del
> dominio (tablas estadísticas densas, figuras escaneadas), que es justo el
> eje forma y su riesgo alto ya anotado; los diez casos de esta sonda son
> semilla válida para esa fase, no desechables, y tampoco se promueven a
> `gold_cases.jsonl` mientras la autoría humana siga pendiente.

### Margen inalcanzable

Separar la métrica de figura (criterio 4) hará aflorar casos que fallan porque el
generador nunca ve el contenido visual: se almacena la leyenda o el literal
`[figure] page=N`. Ese margen es real y se mide, pero **ninguna configuración
puede cerrarlo**: exige un cambio de código fuera del espacio de acción de la
primera etapa.

Esos casos se reportan pero quedan **fuera del escalar que el loop maximiza**,
para que no gaste iteraciones persiguiendo casos imposibles. Entran cuando el
espacio de acción se amplíe a código.

---

## 4. Arquitectura del arnés

```mermaid
flowchart LR
    P[Proponente] -->|configuración| F[Nivel rápido<br/>~15 casos · ~6 min]
    F -->|regresión| X[Descartado]
    F -->|sin regresión| G[Conjunto de búsqueda<br/>~60 casos · ~24 min]
    G --> L[(Libro de evidencias<br/>versionado)]
    L --> P

    classDef cheap fill:#2d5016,stroke:#4a7c23,color:#fff
    classDef costly fill:#5c3a00,stroke:#8a5a00,color:#fff
    classDef store fill:#1a3a5c,stroke:#2d5f8a,color:#fff
    class F cheap
    class G costly
    class L store
```

**Espacio de búsqueda declarado como dato.** Un fichero enumera qué campos son
ajustables y con qué valores. Escrito a mano, no inferido por introspección:
añadir un parámetro debe ser una decisión visible, porque cada uno multiplica el
espacio contra un presupuesto de evaluaciones muy corto.

**Proponente.** El argumento original decía: a ~2,8 horas por evaluación
completa una noche da tres o cuatro candidatos, la búsqueda ciega es inútil con
ese presupuesto, luego el proponente natural es un LLM que lea los fallos y
razone qué mover.

> [!IMPORTANT]
> **Ese argumento ya no se sostiene (medido el 2026-08-12, ver la nota de la
> sección 3).** El conjunto de búsqueda tarda **12,6 min** sobre los 32 casos
> actuales, no 136. Una evaluación completa es el nivel rápido más eso, del
> orden de un cuarto de hora, así que una noche da **decenas** de candidatos.
> El presupuesto que volvía inútil la búsqueda ciega, y con ello obligatorio el
> proponente LLM, desapareció.

Esto **no** demuestra que el proponente determinista baste, y no es motivo para
retirar el LLM. Lo que cambia es que la comparación controlada entre ambos que
ya exige el criterio 5 pasa a ser barata, así que la pregunta se zanja con
evidencia en vez de con un argumento de presupuesto. El arnés mantiene los dos
proponentes justamente para eso.

**Evaluador.** Aplica la configuración, corre el nivel rápido, descarta si hay
regresión, y sólo entonces paga el conjunto de búsqueda completo. Devuelve el
registro en vez de sólo imprimirlo.

> [!IMPORTANT]
> **El evaluador debe fallar ante una configuración que no puede aplicar.**
> Al extraer la API de biblioteca (§6.1) se encontró que los `config_overrides`
> llegan a recuperación e indexado, pero no a la generación: ésta construye su
> propio `AppConfig` por llamada leyendo los globales de `rag/chat_pdfs.py` y el
> entorno del proceso. Un override de `context.max_context_chars` o de
> `flags.usar_recomp_synthesis` que se ignore en silencio hace que el loop mida
> "este parámetro no hace nada" y lo escriba en el libro de evidencias como si
> fuera un hallazgo. Por la política de fallo duro del repo, un override que no
> se pueda honrar de extremo a extremo debe levantar excepción, y el arnés debe
> comprobar al arrancar que cada clave de su espacio declarado es una que el
> evaluador acepta. Un espacio de búsqueda con un mando inerte es peor que uno
> que no lo incluye.

El nivel rápido es un **subconjunto fijo del conjunto de búsqueda** (unos 15
casos elegidos por dar señal: figuras, dominios nuevos, formas difíciles), no una
muestra rotatoria. Una muestra que cambia entre iteraciones introduce varianza en
la comparación, y un ratchet que sólo sube acabaría envenenado por la suerte del
muestreo.

> [!NOTE]
> El nivel rápido **nunca declara una mejora**: no tiene resolución para eso. Es
> un filtro de regresión, no un criterio de aceptación.

**Dónde vive cada umbral.** El ratchet que el loop usa como referencia vive en el
conjunto de búsqueda. El umbral que acepta una versión para integrar se comprueba
contra el conjunto ciego, y esa comprobación ocurre fuera del loop.

**Libro de evidencias.** Registro append-only versionado en git, una entrada por
iteración: configuración exacta, casos aprobados y fallados uno a uno, tiempos, y
veredicto con motivo. Hace posible el criterio 7 y le da expediente al baseline,
que hoy es un `0.77` suelto cuyo run de calibración está fuera del repo.

**Huella del índice.** Resumen de todo lo que afecta al contenido del índice
(extractor, parámetros de troceado, modelo de embedding, flags de indexado),
guardado junto al índice. Si no coincide, se reindexa. Es el criterio 3.

**Ubicación.** El arnés no es producto: no entra en el núcleo hexagonal ni en la
capa de interfaces. Vive en su propio directorio y **consume** el gate como
biblioteca.

> [!CAUTION]
> El arnés puede leer y ejecutar la evaluación, nunca redefinir cómo se puntúa.
> Un optimizador que puede tocar su propio criterio deja de medir.

---

## 5. Supuestos

Adoptados por defecto; cualquiera es revisable.

- **El loop propone, el humano integra.** Nunca fusiona ni despliega solo
  (regla 1 del repo).
- **Techo de latencia:** ningún candidato puede empeorar la latencia mediana por
  consulta más de un 20 % sobre la configuración vigente. **La mediana se toma
  por tipo de caso, no sobre el conjunto entero** (medido 2026-08-12): el coste
  por caso es bimodal -- `factual_number` 201,1 s y `factual_concept` 198,4 s
  frente a `figure_retrieval` 3,9 s y `table_retrieval` 4,5 s, un factor de
  cincuenta --, así que una mediana única sobre casos mezclados informa poco más
  que de qué tipo era el caso central. Un candidato que hunda la calidad de
  recuperación apenas la movería: en el par sano contra saboteado, la mediana de
  casos respondidos pasó de 201,0 s a 195,8 s y la de sólo recuperación de 4,1 s
  a 4,8 s, ambas holgadamente dentro del 20 %, mientras la recuperación se
  desplomaba. Cada bucket se compara contra su propio techo y basta con que uno
  lo incumpla para descalificar al candidato.

> [!IMPORTANT]
> **Recuperación contra referencia degradada (decisión del 2026-08-23,
> issue #92).** El emparejamiento de regresiones contra la referencia del
> propio run es correcto mientras la referencia sea sana; pero el criterio 5
> arranca con una referencia saboteada a propósito, y medido en el pipeline
> real (2026-08-22, issue #89) el sabotaje k=1 tuvo la suerte de pasar
> `planck-sigma8-es`, caso que toda configuración sana falla. Ese pase de
> suerte hizo que el filtro de regresión del nivel rápido rechazara a todos
> los candidatos que restauraban el comportamiento sano: recuperar es fallar
> de nuevo el caso con el que el sabotaje tuvo fortuna. Decisión: cuando el
> libro de evidencias contiene un estado **comparable** (mismos modelos,
> troceado y flags de indexado; ver `_comparable_config_view`) con objetivo
> estrictamente mayor que la referencia actual, el arnés arranca en **modo
> recuperación** y las dos comprobaciones de regresión se emparejan contra el
> vector de pases de esa marca de agua histórica en vez de contra la
> referencia degradada. Perder un caso que sólo pasó la referencia degradada
> no es perder terreno ganado. Dos propiedades quedan fijadas: el ratchet
> sigue arrancando en el objetivo de la referencia, así que aceptar exige
> superarla; y la latencia sigue emparejada contra la referencia del run,
> porque es presupuesto duro y no calidad ganada. Sin historia comparable el
> comportamiento es exactamente el de antes: el límite queda documentado como
> límite, no se adivina una salida. Procedencia: cada entrada puntuada
> registra en `regression_baseline_iteration` (schema v2) contra qué iteración
> del libro se emparejó. Implementación: `harness/loop.py`
> (`_historical_high_water`, `_comparable_config_view`).
- **Terminación:** presupuesto de iteraciones, o parada tras varias iteraciones
  seguidas sin superar el suelo de ruido.
- **Comparación pareada** sobre los mismos casos, nunca entre tasas de runs
  distintos.
- **El grading sigue siendo determinista.** No hay juez LLM en ninguna etapa. La
  asistencia de LLM se limita a la autoría de casos y a proponer configuraciones.
- **Autoría de casos:** se redactan leyendo las páginas reales del PDF, con
  `verified_pages` anotado; auditoría humana sobre una muestra antes de que el
  lote entre.
- **Procedencia:** todo documento nuevo se reproduce desde un identificador
  estable y abierto, y queda fuera del repo. El corpus del producto se congela en
  sus 6 papers actuales para no engordar el bundle de escritorio.

---

## 6. Decisiones que requieren permiso explícito

Ninguna se ejecuta sin confirmación.

1. **Modificar `tests/eval/run_eval.py`** para que acepte una configuración y un
   subconjunto de casos, y devuelva sus resultados. La regla 9 del repo declara
   `tests/eval/` intocable sin acuerdo previo. No se toca ninguna regla de
   puntuación: `grade.py` queda intacto.

2. **Introducir la huella del índice**, que cambia comportamiento observable del
   producto: documentos antes reutilizados pasarán a reindexarse. Puede hacer
   saltar tests de caracterización. Si ocurre, la regla del repo obliga a parar y
   consultar, no a actualizar el test.

---

## 7. Descomposición

El diseño es coherente como una pieza, pero no cabe en un solo plan de
implementación. Se parte en tres, con una dependencia real entre ellas:

| Bloque | Contenido | Criterios que cierra |
|---|---|---|
| **A · Medición fiable** | Huella del índice, separación de métricas de figura, medida del suelo de ruido, prueba de sensibilidad | 1 · 2 · 3 · 4 |
| **B · Corpus** | Sonda diagnóstica, veredicto por eje, tandas de casos, partición por documento | Habilita 2 y 5 con resolución suficiente |
| **C · Arnés** | Espacio declarado, proponentes, evaluador, libro de evidencias | 5 · 6 · 7 · 8 |

**A va primero y no es negociable.** Es lo que hace que los números de B y C
signifiquen algo. B y C pueden solaparse: el arnés se puede construir y probar
contra el corpus actual, aceptando que su resolución es baja hasta que B entregue.

Cada bloque recibe su propio plan de implementación.

---

## 8. Fuera de alcance

- Edición de código por parte del loop, y todo lo que exigiría: aislamiento por
  rama, revisión automática del diff, criterio de reversión.
- Generación multimodal: que el generador interprete el contenido visual de una
  figura. Es la vía para cerrar el margen inalcanzable de §3, y necesita su propio
  diseño.
- Los tres hallazgos de runtime de la auditoría (worker no thread-safe, VRAM sin
  liberar en web/CLI, script del worker ausente del bundle). Bloquean el producto,
  no el loop, y se tratan aparte.
