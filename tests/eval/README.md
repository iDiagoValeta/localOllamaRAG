# Gold eval corpus

Deterministic judge for the RAG pipeline: known-answer questions over real arXiv papers,
graded without an LLM. Built per `docs/design/2026-07-26-monkeygrab-v2.md` section 7.1-7.2.
This step only builds the judge and its corpus — nothing here runs the pipeline yet.

## Files

| File | Role |
|---|---|
| `gold_cases.jsonl` | One JSON object per line: a question with verified accepted answers. |
| `fetch_papers.py` | Idempotent arXiv PDF downloader (id -> cached, gitignored PDF). |
| `grade.py` | Deterministic scoring: `grade_answer`, `grade_retrieval`. |
| `test_grade.py` | pytest suite for the grader + a schema check over `gold_cases.jsonl`. |
| `papers_cache/` | Downloaded blind-set PDFs. Gitignored — reproduced from `arxiv_id`. |

## Corpus

Two sets of papers, mixed deliberately:

- **Dev set** (`source: "corpus"`): the six papers already in `rag/docs/en/`. Their retrieved
  chunks and reranking behavior are what the pipeline's heuristics were tuned against.
- **Blind set** (`source: "arxiv"`, `arxiv_id` set): three papers never used to tune anything
  — ResNet (`1512.03385`), BERT (`1810.04805`), ViT (`2010.11929`). A judge calibrated only on
  the dev set can't tell "the pipeline retrieves well" from "the pipeline overfits these six
  PDFs"; the blind set is what makes that distinction possible.

Every `accepted_answers` / `expect_kind_any` value was checked by hand against the PDF (see
each case's `verified_pages` — the physical page numbers, 1-indexed, as read directly from the
source PDF) before being added. A case whose answer could not be confirmed in the text does not
go in the file.

## Schema (`gold_cases.jsonl`)

One case per line:

| Field | Meaning |
|---|---|
| `id` | Unique, `<paper-slug>-<short-name>`. |
| `paper` | Paper slug. Matches the corpus filename stem for `source: "corpus"`. |
| `source` | `"corpus"` (already in `rag/docs/en/`) or `"arxiv"` (fetch first). |
| `arxiv_id` | Only when `source == "arxiv"`; passed to `fetch_papers.py`. |
| `case_type` | `factual_number` \| `factual_concept` \| `figure_retrieval` \| `table_retrieval`. |
| `lang` | `en` \| `es` — the question's language, not the document's. |
| `question` | The query text (a retrieval query for the two `*_retrieval` types). |
| `accepted_answers` | List of literals, any one of which counts as correct. Required for the two factual types. |
| `expect_kind_any` | List among `text`/`table`/`image` — the content kind expected in the top-k. Required for the two retrieval types. |
| `verified_pages` | PDF pages (1-indexed) where the fact/figure/table was confirmed. |
| `notes` | Optional: nuance worth flagging (ambiguous source values, grading caveats). |

`table`/`image` follow the target content taxonomy from the design doc, section 4 — today's
pipeline only tags `text`/`image`, so `table_retrieval` cases will fail until table-aware
extraction lands. That is expected, not a case bug.

## Adding a case

1. Pick a paper already in the corpus, or a new `arxiv_id`.
2. Read the actual page with the `Read` tool's `pages` parameter (or any PDF viewer) — do not
   trust a remembered fact.
3. Write the question and the literal(s) exactly as confirmed, plus `verified_pages`.
4. Prefer literals that start and end with a letter or digit — `grade.py`'s word-boundary match
   anchors on alphanumeric transitions, so a literal starting with punctuation matches looser
   than intended (documented in `grade._contains_token`).
5. Run `pytest tests/eval/test_grade.py` — the schema test catches malformed fields.

## Running

```bash
python -m pytest tests/eval/                       # grader unit tests, no network
python tests/eval/fetch_papers.py                   # fetch every arxiv_id in gold_cases.jsonl
python tests/eval/fetch_papers.py 1512.03385         # fetch one paper by id
```

`fetch_papers.py` is idempotent: a cached, header-validated PDF is never re-downloaded.
