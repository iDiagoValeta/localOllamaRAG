# Synthetic RAG questions: writer and verifier instructions

The data trains and tests a small model that answers questions over retrieved
document chunks (MonkeyGrab). Its main failure is picking the wrong number,
especially from tables, so precise numeric and table questions are the most
valuable items.

## Writer

Input: a JSONL batch, one chunk per line, with `chunk_id`, `doc`, `doc_lang`,
`format` (`markdown` or `table`), `text` and `question_lang` (`en`, `es`, `ca`).

For every chunk, write the items its `text` supports (skip bibliography,
references, navigation, captions without facts, or garbled extraction):

- `number`: the answer is a number, quantity, percentage, date or year stated
  in the text. For a `table` chunk, ask for one cell, naming its row and its
  column by their labels ("What F1 does Model X reach on SQuAD v2.0 dev?").
- `term`: the answer is a specific name, term or short phrase of at most six
  words from the text.

Rules:

- Write the question in `question_lang` (Castilian Spanish for `es`,
  Valencian/Catalan for `ca`). When it differs from `doc_lang`, prefer a
  `number` item, since a term would have to be translated.
- The question is self-contained: name the subject explicitly (method, model,
  person, place, event). Never say "the passage", "the text", "the table", "the
  document", "the article", "according to the above".
- Exactly one value in the text answers it. If the text has several plausible
  candidates (two years, two scores), the question must pin down which one.
- `answer`: copied verbatim from `text`, character for character, as short as
  possible (a number with its unit, nothing else). It must not appear in the
  question.
- `ideal`: the answer a perfect assistant would give, in `question_lang`:
  the value first, in one or two complete sentences, at most about 40 words,
  no filler, no headings or lists, the value written exactly as `answer`
  (formulas in LaTeX `$...$`). Do not add facts the text does not state.

Output: one JSON object per line, nothing else, written with the Write tool to
the output path you are given:

```json
{"chunk_id": "<copied>", "item_id": "<chunk_id>:<n>", "question_lang": "<copied>", "kind": "number|term", "question": "...", "answer": "...", "ideal": "...", "writer": "<your model: opus|sonnet|haiku>"}
```

`n` counts the items of that chunk from 1.

## Verifier

Input: the same chunk batch and the writer's items for it. For each item,
answer `keep` only if all hold:

1. `answer` is the correct and complete answer to `question` according to the
   chunk text, and is the only value in the text that answers it.
2. The question is self-contained and in `question_lang`, and does not refer to
   "the passage/text/table/document".
3. `ideal` states `answer` correctly in its first sentence, in
   `question_lang`, with nothing the text contradicts or does not support.

Otherwise `drop`. Output one JSON object per line with the Write tool:

```json
{"item_id": "<copied>", "verdict": "keep|drop", "reason": "<= 15 words"}
```
