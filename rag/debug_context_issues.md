# Observaciones sobre la presentación del contexto al generador

Análisis basado en el dump de debug:
`rag/debug_rag/20260405_182746_Explain_the_transformers_architecture_an.txt`

Query: *"Explain the transformers architecture and why it's so important."*

---

## A. `[incomplete fragment]` visible al modelo generador

El Fragment 2 llega al prompt con este texto literal al final del Source Text:

```
2 Background The goal of reducing sequential computation also forms the
foundation of the Extended Neural GPU
[incomplete fragment]
```

**Origen:** `_marcar_fragmento_incompleto` (chat_pdfs.py:1555) añade el label cuando
el chunk no termina en puntuación de cierre (`.?!:`). Es una decisión de diseño
intencionada para indicar al modelo que el texto está cortado.

**Impacto observado:** ninguno — la respuesta generada fue correcta y completa.

**Decisión:** dejar como está. Si en el futuro se observa que el label degrada
respuestas, se puede eliminar quitando la llamada a `_marcar_fragmento_incompleto`
en `construir_contexto_para_modelo` (línea ~1637).

---

## B. Fragments con score 0.0 que llegan al contexto

En el debug, Fragments 3 y 4 tienen `score: 0.0` y `Reranker score: N/A`, pero aun
así se incluyen en el contexto enviado al generador.

**Origen:** `expandir_con_chunks_adyacentes` añade chunks adyacentes a los
top-ranked *después* del reranking. Al añadirse en esta fase, los chunks expandidos
no pasan por el reranker y heredan score 0.0, bypasseando el umbral de relevancia
`UMBRAL_RELEVANCIA = 0.50`.

**Coste real:** ~800 tokens de contenido redundante enviados al generador, solapado
con Fragment 2 (misma región del documento, pp. 2–3).

**Impacto observado:** ninguno en esta query — la respuesta fue buena.

**Posible mejora:** en `expandir_con_chunks_adyacentes`, asignar a los chunks
expandidos el score del fragmento que los origina (en vez de 0.0) y filtrarlos por
`UMBRAL_RELEVANCIA` antes de añadirlos. No implementado — el tradeoff (chunks
adyacentes añaden contexto estructural útil en otros casos) merece evaluación
con más queries antes de cambiar.

---

## C. Duplicación de contenido entre Fragments 2, 3 y 4

Los tres fragmentos proceden de la misma zona del paper (páginas 2–3). El texto:

> *"Recent work has achieved significant improvements in computational efficiency
> through factorization tricks [21]..."*

aparece en el Source Text de los tres. Es consecuencia combinada del overlap de
350 chars en el chunking + la expansión de contexto añadiendo adyacentes.

**Impacto observado:** ninguno apreciable — el generador supo ignorar la repetición.

**Relación con B:** es el mismo problema visto desde el contenido en lugar de los
scores. Si se corrige B, C se resuelve en gran medida.

---

## Conclusión

La respuesta generada por phi4 fue correcta y completa a pesar de los tres issues.
El pipeline de recuperación es robusto. Los tres puntos quedan documentados aquí
como deuda técnica menor a revisar si en el futuro se observan degradaciones en
la calidad de respuesta o un uso excesivo del contexto.
