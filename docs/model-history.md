# Model history — what has been measured, on what, and how fast

Append-only record of every model this project has run, so a later choice is
made from numbers rather than from memory. Each row names the machine and the
run artifact it came from; without those a figure is an anecdote.

**Read the caveat before the tables.** The gold set has 32 search-set cases
and 23 of them reach a generator. The design doc puts the threshold for a
difference not attributable to chance at about **six net flips**, so a gap of
one or two cases between two generators is inside the noise this project has
already measured. These tables are a log, not a ranking.

Run artifacts live under `tests/eval/runs/`, which is gitignored: the paths
are cited so the machine that holds them can reproduce the reading, not
because a reader can open them.

---

## The fixed stack

Three of the five model roles are not configurable, and they decide more than
the generator does. A generator swap moved two cases out of 23; the extractor
version moved three (see "Stack drift" below).

| Role | Model | Where it runs | Notes |
|---|---|---|---|
| PDF extraction | **MinerU 3.4.5** | `.venv-mineru`, GPU | Downloads its own models on first use. Its 2.x line produced a different output layout and different block types — not interchangeable, see #118. |
| Text+image embedding | **jinaai/jina-clip-v2**, 512-dim Matryoshka | `.venv-mineru`, GPU, ~2.8 GiB | Fixed by design. CC BY-NC 4.0: local use is non-commercial. |
| Reranking | **BAAI/bge-reranker-v2-m3** | product venv, GPU, ~1.5 GiB | Cross-encoder. Not an Ollama model. |
| Query decomposition | Ollama, `chat` role | Ollama | The gate pins this to its own `AUX_MODEL` so a `--models` sweep does not also swap the decomposer. |
| Answer generation | Ollama, `rag` role | Ollama | The only role the tables below vary. |

---

## Generators measured

`gemma4:e4b` is the repo default. Cases are the 23 answered ones; the other
42 never call a generator.

| Model | Params / size | Answered cases passed | Median s/case | tokens/s | tokens/answer | s/answer | Run |
|---|---|---|---|---|---|---|---|
| `qwen3-coder-30b:latest` | 30B MoE, 10 GB | **21 / 23** | 41.7 | not recorded | not recorded | not recorded | `20260901T023915Z` |
| `gemma4:e2b` | 7.2 GB | 20 / 23 | 38.7 | not recorded | not recorded | not recorded | `20260901T023915Z` |
| `gemma4:e4b` *(default)* | 9.6 GB | 19 / 23 | 39.8 | not recorded | not recorded | not recorded | `20260901T023915Z` |
| `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` | 3B dense, 2.0 GB | **49 / 63** (77.8%) | 8.2 | 98.5 | 25 | 0.25 | `20260903T115934Z` |
| `hf.co/noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF:MXFP4_MOE` | MoE, 4.9 GB | **56 / 63** (88.9%) | 8.1 | 109.3 | 18 *(of 50)* | 0.15 | `20260903T124819Z` |
| `qwen3:30b-a3b` | 30B MoE, 3B active, 18 GB | pending | | | | | in progress |
| `qwen3:8b` | 8B dense, 5.2 GB | pending | | | | | in progress |
| `mistral-small3.2:24b` | 24B dense, 15 GB | pending | | | | | in progress |
| `gpt-oss:20b` | 20B MoE, 13 GB | not yet run | | | | | |
| `granite4:small-h` | MoE, 19 GB | not yet run | | | | | |

**Speed is not latency, and this is the column that says so.** `tokens/s` is
how fast a model decodes; `s/answer` is how long the user waits, and it is
tokens divided by that rate. A model that reasons inline spends its budget
before it starts answering, so the two columns can rank the same pair of models
in opposite orders. Measured 2026-09-01, one representative prompt, `think:
false`, `num_predict: 96`, `temperature: 0`:

| Model | tokens/s | tokens used | s/answer |
|---|---|---|---|
| `qwen3-coder-30b:latest` | 43.5 | 5 | **0.11** |
| `qwen3:30b-a3b` | 38.2 | 96 | **2.51** |

Fourteen per cent apart on the column this table used to show; twenty-three
times apart on the wait. Both answered correctly. `think: false` controls
Ollama's *thinking channel* — the separate field a model emits its trace into —
and does not stop a model whose chat template reasons inline from doing it in
the body of the answer. Qwen3 does exactly that. Full analysis in issue #146.

The second consequence is worse than the wait. At `num_predict: 96` the
reasoning consumed the whole budget, and the answer survived by luck; a longer
preamble truncates into something that grades as a plain `FAIL`,
indistinguishable from the model not knowing. So a high `tokens/answer` is not
only a cost, it is a **truncation risk**, and that is the number to check first
when a capable model grades badly.

**Why the tokens/s column is empty above.** The gate recorded
`elapsed_seconds` per case, which includes the retrieval every model shares —
so it mostly measured how busy the card was, not how fast the model decoded.
Ollama reports `eval_count` and `eval_duration` on its final chunk and the
streaming path always collected them; the silent path the gate uses dropped
them. Fixed 2026-09-01: every record now carries `eval_count`,
`eval_duration_s`, `generation_seconds` and `tokens_per_second`, and the
column fills from the next run onward. Earlier rows stay empty rather than
being back-filled with a number nobody measured.

### What the three-model comparison actually showed

Paired over identical retrieved fragments, **21 of 23 cases gave the same
verdict under all three models**. Only two discriminated:

| Case | `gemma4:e2b` | `gemma4:e4b` | `qwen3-coder-30b` |
|---|---|---|---|
| `dpo-model-scale` | PASS | fail | PASS |
| `planck-h0` | fail | fail | **PASS** |

Two more (`planck-h0-tension`, `planck-sigma8-es`) fail under all three, which
makes them system failures rather than model failures.

The practical reading: **the generator is not where the remaining failures
are**, and a 30B MoE that does not fit in 8 GB costs about 2 s per case more
than a 2B that does. Full analysis in issue #134.

---

## Stack drift, which moved more than any generator swap

Same generator (`gemma4:e4b`), same 32 search-set cases, different extractor
version:

| When | Stack | Search set |
|---|---|---|
| Reference (per #30's comment) | MinerU 2.x line | 27 / 32 |
| 2026-09-01 | MinerU 3.4.5 | 26 / 32 |

Three flips: one recovered (`dpo-pipeline-figure-es`), two lost
(`dpo-model-scale`, `planck-h0`). The measured noise floor of this gate is
**zero flips**, so three is not noise.

One honest qualification, added after the three-model comparison: both "lost"
cases are among the two that discriminate between generators, so they sit near
a decision boundary. The drift is real; attributing those two specifically to
MinerU's version is not supportable. See issue #107.

---

## Hardware these numbers came from

RTX 4060 Laptop, 8 GB (7.6 GiB usable), Ryzen 9 8945HS, 30 GB RAM. Every run
above used `OLLAMA_KEEP_ALIVE=0`, which was required to fit the pipeline on
this card before #143 and changes timing — a run made that way is not
comparable with ledger history that was not.

---

## How to add a row

Run the gate, then read the artifact rather than the terminal:

```bash
python tests/eval/run_eval.py --models <model> [<model>...]
```

Then let the tool read the artifact and print the row:

```bash
python tools/diagnostics/model_history_row.py tests/eval/runs/<artifact>.json
```

It reports the median of `tokens_per_second`, `eval_count` and their quotient
per model, skipping records that lack the counts — a missing key means the
model reported none, and treating it as zero drags the average toward "faster
and cheaper" for a reason that is not real. When a model's medians come from
fewer records than it answered, the row says so in the `tokens/answer` cell:
three medians over two generations deserve less trust than over twenty-three,
and a table that hides its denominator stops saying which it is.

The median rather than the mean, because one runaway generation should not
move a figure that goes into an append-only table.
