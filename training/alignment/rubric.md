# RAG answer rubric

You judge one candidate answer to a question about a document. You receive: `id`, the question, its language (`en`, `es` or `ca`), the reference span, the source passage the reference comes from, and the candidate. The reference is ground truth. The candidate was written from a larger context you cannot see, so judge only what the passage lets you check.

## Criteria, most important first

| Key | Definition | Fails when |
|---|---|---|
| `correct` | The candidate states the fact in the reference. | The reference value, term or name is missing, wrong, misspelled, has a different digit, decimal position, sign or order of magnitude, a wrong unit, or the candidate offers two or more values for the asked quantity without committing to the reference. |
| `faithful` | Nothing in the candidate contradicts the passage or changes the answer. | A claim contradicts the passage, or a claim absent from the passage replaces, qualifies or casts doubt on the answer. |
| `refusal` | The candidate answers whenever the passage holds the answer. | It says the information is unavailable, asks for more context or declines, while the passage contains the reference. |
| `language` | The answer is written in the question's language. | Any sentence is in another language, or Catalan and Spanish are mixed. Technical terms, proper names, titles and short quotations may stay as the passage writes them. |
| `direct` | The answer comes first, in complete sentences, with nothing extra. | The answer is not in the first sentence; or a factual question (who, what, when, how many) gets more than 2 sentences or about 50 words; or there are headings, bullet lists or bold-heavy formatting for one fact; or there is filler ("Great question", "In summary", restating the question), a `<think>` block, or a cut-off final sentence. A bare value with no sentence ("1872.") also fails. Why/how questions may take up to about 120 words. |
| `notation` | Numbers, units and formulas are copied faithfully. | A formula, variable or expression is not in LaTeX `$...$` (or `$$...$$`), or uses `\(...\)` / `\[...\]`; a unit present in the reference is dropped; a value is rounded or reformatted beyond the allowed variants below. |
| `aside` | Minor unsupported detail. | A short claim not in the passage that does not touch the answer (background, a definition, a guess at a cause). |
| `reference` | Data flag, not a candidate defect. | The reference does not answer the question according to the passage: it echoes the question's own words, or it is a fragment of a fuller answer that the passage gives elsewhere. |

Allowed variants, never a failure: decimal comma or dot ("0,5" or "0.5"), thousands separators ("250 000", "250,000", "250.000" in es/ca), a space or not before `%` or a unit, `x` or `\times`, Unicode or LaTeX superscripts, Markdown bold around the answer, a fuller form that contains the reference ("12-14" for "12"; "el 3 de març de 1911" for "3 de març"). Extracted PDFs often space out digits ("3 4 8 . 5"): reading that as "348.5" is correct; splitting it into another number ("34.85", "7.1" for "7 1") is a `correct` failure. Ignore fragment or table numbers you cannot check.

## Verdicts

- **good**: no key failed; usable as a training target exactly as written.
- **fixable**: `correct`, `faithful` and `refusal` pass, and only `language`, `direct`, `notation` or `aside` fail; a rewrite can fix it without new facts.
- **bad**: `correct`, `faithful` or `refusal` fails, or `reference` is flagged; the item is discarded.

Edge cases: if the candidate gives a different value that the passage also supports and the reference is a partial span, use `bad` with `["reference"]`. If the answer is right but the candidate also misquotes the passage on a point unrelated to the question, that is `aside` (fixable), not `faithful`.

## Output

One JSON object per item, nothing else:

```json
{"id": "<copied from input>", "verdict": "good|fixable|bad", "failed": ["<criterion keys, in table order>"], "reason": "<= 20 words"}
```

`failed` is `[]` for good. The reason names the concrete defect, not the rule.

## Worked examples (invented)

**1. good.** Lang `en`. Q: How many stations does the Orvenne tramway have? Reference: `14 stations`. Passage: "...the Orvenne tramway, with 14 stations along the river..." Candidate: "The Orvenne tramway has 14 stations."

```json
{"id": "ex1", "verdict": "good", "failed": [], "reason": "Exact value in one sentence, correct language."}
```

**2. good.** Lang `es`, English passage. Q: ¿Qué tasa de aprendizaje usa Kestrel en el preentrenamiento? Reference: `$3 \times 10^{-4}$`. Candidate: "Kestrel usa una tasa de aprendizaje de $3 \times 10^{-4}$ durante el preentrenamiento."

```json
{"id": "ex2", "verdict": "good", "failed": [], "reason": "Value copied exactly in LaTeX, answered in Spanish."}
```

**3. fixable.** Lang `ca`. Q: Quin any es va inaugurar el far de Punta Merla? Reference: `1872`. Candidate: "El faro de Punta Merla se inauguró en 1872."

```json
{"id": "ex3", "verdict": "fixable", "failed": ["language"], "reason": "Correct year but answered in Spanish, not Catalan."}
```

**4. fixable.** Lang `en`. Q: What dropout rate does the Kestrel encoder use? Reference: `0.15`. Candidate: "Great question! The Kestrel encoder uses a dropout rate of 0.15. Dropout was invented to prevent overfitting and is used in most modern networks. In summary, the rate is 0.15."

```json
{"id": "ex4", "verdict": "fixable", "failed": ["direct", "aside"], "reason": "Right value; filler, repeated summary and unsupported background."}
```

**5. bad.** Lang `es`. Q: ¿Cuántos kilómetros mide el canal de Vessa? Reference: `47 km`. Passage: "...el canal de Vessa mide 4 7 km..." Candidate: "El canal de Vessa mide 4,7 km."

```json
{"id": "ex5", "verdict": "bad", "failed": ["correct"], "reason": "Spaced digits read as 4.7 instead of 47 km."}
```

**6. bad.** Lang `en`. Q: In which year did the Orvenne tramway open? Reference: `1908`. Passage: "...opened in 1908 and was electrified in 1931..." Candidate: "The Orvenne tramway opened in 1931, when its first line was electrified."

```json
{"id": "ex6", "verdict": "bad", "failed": ["correct", "faithful"], "reason": "Gives the electrification year; contradicts passage's 1908 opening."}
```

## Rewrite instructions

For the agent that turns a `fixable` answer into a `good` one. You receive the same inputs plus the judge's `failed` keys. Output only the rewritten answer text.

You may:
- Translate into the question's language, keeping technical terms, proper names, titles and quotations as the passage writes them.
- Cut filler, repeated summaries, `<think>` blocks, lists, headings, fragment or table numbers, and any `aside`.
- Turn a bare value into one complete sentence; put the answer in the first sentence.
- Put formulas and variables in `$...$`, replace `\(...\)` with `$...$`, restore a unit the reference carries.

You must:
- Keep the answer value exactly as in the reference: same digits, decimal position, unit and term spelling. You may use the decimal comma in `es`/`ca`.
- Use only facts present in the passage. Add nothing from memory, even if true.
- Keep factual answers to 1 or 2 sentences and under about 50 words; why/how answers under about 120 words.
- Never change which value or entity answers the question. If a fix would require that, return the single word `REJECT` instead.
