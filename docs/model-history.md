# Model history -- what has been measured, on what, and how fast

Append-only record of every model this project has run, so a later choice is
made from numbers rather than from memory. Each row names the run artifact it
came from; without that a figure is an anecdote.

**Conditions are not restated here.** What has to match between two runs for
their numbers to be comparable is written once, in
[`tests/eval/README.md`](../tests/eval/README.md) under "Experiment standard",
together with the procedure for adding a row. A row that cannot point at an
artifact produced under that standard belongs in "Superseded measurements"
below, not in the current table.

Run artifacts live under `tests/eval/runs/`, which is gitignored: the
identifiers are cited so the machine that holds them can reproduce the
reading, not because a reader can open them.

---

## How to read the current table

Four qualifications, all of them measured rather than assumed. Skipping them
turns this log into a ranking, which it is not.

1. **A row is one run, and one run moves by up to five cases.** No stage
   of the pipeline fixes a seed and three of them sample at non-zero
   temperature, one of which runs *before* retrieval and so changes which
   fragments the generator ever sees (issue #223). Measured once on this set
   by running tranche 1 twice under identical conditions (`20260911T032105Z`
   against `20260911T230513Z`, four models, see "Tranche 1 repeated" below):
   sampling alone flipped **28 of 536** comparable (case, model) pairs, 4 to
   11 per model, with a net movement per model between -2 and +5 cases. That
   is the error bar of the `Answered` column. A gap of five cases between two
   rows is inside it; the `Runs` column says how many runs a row rests on.
2. **`s/answer` is the wall time of the generation call**, not tokens divided
   by the decode rate. The older form of this column understated the wait by
   a factor of 36 -- 0.22 s against a measured 7.89 s -- because it omitted
   prompt evaluation, which dominates when the prompt is retrieved context
   (issue #233). `tokens/s` still describes decoding only.
3. **Placement is part of the measurement.** Two models do not fit in 7.6 GiB
   alongside the auxiliary, so Ollama may load one into system RAM instead,
   silently. Where that was observed the row says so, and its speed columns
   describe the degraded regime (issue #235).
4. **The budget column is mostly not a model property, and it counts
   against the model anyway.** A generation is capped at 180 s (issue #229)
   and an exhausted budget is scored as a failure of the row's model. The
   repeat of tranche 1 showed that 19 of its 24 exhaustions did not
   reproduce: they sat in two contiguous windows of that run, where every
   generation of every model ran past the cap while the decode rates on
   either side were unchanged. The 5 that did reproduce are two `study_summary`
   cases over whole documents (`ricci-study-summary-es`, 96 chunks, in all
   four models both times; `att-study-summary-ca` at the edge, 171 s for the
   one model that passed it). So tranche 1's rows carry about 5 cases of
   run-level loss the other tranches do not, and a budget count of 4-6 on a
   small model is that, not the model (issue #234).

---

## Generators on the current gold set

156 cases: 136 reach a generator, 20 are retrieval-only and shared by every
model in a run. Three corpora of 17 documents (en/es/ca) plus the blind set.
Budget 180 s, `AUX_MODEL` = `Ling-3.0-tiny` for every row.

| Model | Size | Answered (of 136) | Overall (of 156) | tokens/s | tokens/answer | s/answer | Budget hit | Infra | Placement | Run | Runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3-coder-30b:latest` | 10 GB | **125 (91.9%)** | 142 (91.0%) | 45.4 | 90 | 15.03 | 0 | 1 | **~50% CPU** | `20260911T094107Z` | 1 |
| `qwen3:30b-a3b` | 18 GB | 121 (89.0%) | 138 (88.5%) | 39.1 | **335** | 24.93 | 6 | 2 | **~70% CPU** | `20260911T171038Z` | 1 |
| `gemma4:e2b` | 7.2 GB | 120 (88.2%) | 137 (87.8%) | 103.2 | 28 | 7.97 | 0 | 0 | GPU | `20260911T071629Z` | 1 |
| `qwen3:8b` | 5.2 GB | 120 (88.2%) | 137 (87.8%) | 35.8 | 35 | 8.94 | 0 | 0 | **CPU offload** | `20260911T071629Z` | 1 |
| `gemma4:e4b` *(shipped default)* | 9.6 GB | 119 (87.5%) | 136 (87.2%) | 59.9 | 28 | 11.84 | 0 | 0 | GPU | `20260911T094107Z` | 1 |
| `hf.co/noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF:MXFP4_MOE` | 4.9 GB | 116 (85.3%) | 133 (85.3%) | 116.3 | 27 | 7.52 | 6 | 0 | GPU | `20260911T032105Z` | 2 |
| `granite4:small-h` | 19 GB | 116 (85.3%) | 133 (85.3%) | 17.4 | 70 | 20.52 | 5 | 0 | **~72% CPU** | `20260911T171038Z` | 1 |
| `hf.co/noctrex/Granite-4.0-H-Tiny-MXFP4_MOE-GGUF:...` | 4.2 GB | 112 (82.4%) | 129 (82.7%) | 116.9 | 72 | 7.15 | 6 | 2 | GPU | `20260911T032105Z` | 2 |
| `hf.co/noctrex/LFM2-8B-A1B-MXFP4_MOE-GGUF:...` | 4.9 GB | 108 (79.4%) | 125 (80.1%) | 156.2 | 56 | 6.92 | 0 | 0 | GPU | `20260911T071629Z` | 1 |
| `mistral-small3.2:24b` | 15 GB | 107 (78.7%) | 124 (79.5%) | **5.2** | 24 | 20.94 | 12 | 0 | **~70% CPU** | `20260911T171038Z` | 1 |
| `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` | 2.0 GB | 105 (77.2%) | 122 (78.2%) | 101.3 | 27 | 6.70 | 6 | 0 | GPU | `20260911T032105Z` | 2 |
| `gpt-oss:20b` | 13 GB | 96 (70.6%) | 113 (72.4%) | 29.7 | 100 | 20.80 | 0 | 4 | **~60% CPU** | `20260911T094107Z` | 1 |
| `hf.co/noctrex/OLMoE-1B-7B-0125-Instruct-MXFP4_MOE-GGUF:...` | 3.9 GB | 91 (66.9%) | 108 (69.2%) | 213.0 | 64 | 6.56 | 6 | 0 | GPU | `20260911T032105Z` | 2 |
| `hf.co/noctrex/Phi-mini-MoE-instruct-MXFP4_MOE-GGUF:...` | 4.9 GB | 63 (46.3%) | 80 (51.3%) | 95.3 | 118 | 9.24 | 0 | 0 | GPU | `20260911T071629Z` | 1 |

Retrieval-only was 17/20 in all five runs -- it does not vary by generator,
which is why it is reported once rather than folded into each row.

`Runs` is how many full runs the row's model has on this set. The four with
2 are tranche 1, repeated to measure the noise (next section); their row
keeps the first run's figures so the table stays one artifact per row, and
the second run sits below.

### Tranche 1 repeated

Same four generators, same declared conditions (`conditions` blocks equal
field for field except `git_commit`: the repeat ran on `bbf284c`, three
defect fixes later -- #237, #238, #239 -- none of them in the generation
or grading path). `20260911T032105Z` first, `20260911T230513Z` second, 105
min against 159.

| Model | Answered, run 1 | Answered, run 2 | Flips (+/-) | Budget hit | Infra |
|---|---|---|---|---|---|
| `Ling-3.0-tiny` | 116 (85.3%) | 117 (86.0%) | +4 / -3 | 6 -> 2 | 0 -> 0 |
| `Llama-3.2-3B` | 105 (77.2%) | 114 (83.8%) | +10 / -1 | 6 -> 1 | 0 -> 0 |
| `Granite-4.0-H-Tiny` | 112 (82.4%) | 114 (83.8%) | +6 / -2 | 6 -> 1 | 2 -> 6 |
| `OLMoE-1B-7B` | 91 (66.9%) | 90 (66.2%) | +5 / -6 | 6 -> 1 | 0 -> 0 |

`python tests/eval/compare_runs.py --exclude-infrastructure-errors` over the
pair: 25 flipped to PASS, 12 to FAIL, 521 unchanged, 6 pairs excluded (all
Granite's `GGML_ASSERT` on `att-study-*`, which now hits six of the nine).
Two effects sit in those 37 flips and they should not be read as one:

- **Sampling.** Dropping every pair that exhausted the budget in either run
  leaves 536 pairs and 28 flips (16 up, 12 down): 4 for Ling, 7 for Llama,
  6 for Granite, 11 for OLMoE, net -2 to +5 per model. That is the noise
  floor of one row. Decode rates and `s/answer` medians agree between the
  runs to within 0.4 s, so the two runs measured the same machine in the
  same state.
- **Budget.** 24 exhaustions became 5. Llama's +9 net is mostly this: 6 of
  its flips to PASS are cases that ran out of time in run 1 and answered in
  9-12 s in run 2. The 19 that vanished were two contiguous windows of run 1
  (records 225-240 and 248-255 in generation order, every model, `study`
  and factual cases alike), and the 54 minutes of difference in wall time
  is those 19 x 180 s. What made run 1 stall there is not in the artifact
  (issue #234).

The first-run figures stay in the table above because the repeat was made to
measure the noise, not to pick the better of two draws; replacing a row with
its best run is the selection bias the row count exists to expose.

**The Infra column is cases the model never got to answer.** Granite's two
are `GGML_ASSERT(buffer) failed` on `study_outline`; `gpt-oss:20b`'s four and
`qwen3-coder-30b`'s one and `qwen3:30b-a3b`'s two are `500 Server Error`
from Ollama mid-stream, the signature of a model that does not fit running
out of memory under offload.
A row with a non-zero Infra count rests on fewer than 136 attempts, and the
gate marks the whole run inconclusive by its own rule -- the figures are
still the best available, but they are not a clean 136.

### What this does and does not show

The spread from 91.9% to 46.3% is far wider than the noise floor plausibly
covers, so the ends of the table are real findings. `Phi-mini-MoE` is not
usable for this pipeline. `OLMoE` gives up 25 points for a decode rate nobody
waits on: its `s/answer` is the best in the table and it is the second worst
model in it, which is the clearest illustration available that speed and
latency are different questions.

The top is not a ranking. `qwen3-coder-30b` leads by five cases over a
three-way cluster at 119-120, and five cases is inside the measured noise of
a single run (4-11 flips per model, net up to +5); it is also the model that
led the 23-case table on 2026-09-01, which is weak evidence but not none.
`gemma4:e2b`, `qwen3:8b` and the shipped default `gemma4:e4b` are separated
by one case.

Placement is the finding that cuts across the table. The best measured model
runs half in system RAM, `qwen3:8b` ties for third while never reaching the
GPU, and every model above 10 GB ran at 60-72% CPU. On this card, quality and
fitting in VRAM are not the same axis, and a wait of 15-25 s against 7-8 s is
the price of the extra cases.

The four largest models answer the question the old "pending" rows left
open, and the answer is not uniform. `qwen3:30b-a3b` reaches 89% but does it
by reasoning inline -- a median of **335 tokens per answer** against 24-35
for everything else, which is the truncation risk the old file described and
here cost it six budget exhaustions and 25 s per answer. `granite4:small-h`
matches `Ling-3.0-tiny` on quality at a third of the speed. `mistral-small3.2`
at 5.2 tok/s is the slowest row in the table and exhausted the budget twelve
times; with 70% of a 15 GB model in system RAM it is not a configuration
anyone would run. The tranche took 278 minutes against 144-159 for the
others.

---

## Coverage

Every generator the previous table listed, including the five rows it carried
as "pending" or "not yet run", now has a row above measured on the same set.
The tranches: `20260911T032105Z` (4 models, 159 min), `20260911T071629Z`
(4, 144 min), `20260911T094107Z` (3, 144 min), `20260911T171038Z` (3, 278
min), plus the repeat of tranche 1, `20260911T230513Z` (4, 105 min). Total
GPU time for the table: about 14 hours.

---

## The fixed stack

Three of the five model roles are not configurable, and they decide more than
the generator does.

| Role | Model | Where it runs | Notes |
|---|---|---|---|
| PDF extraction | **MinerU 3.4.5** | `.venv-mineru`, GPU | Downloads its own models on first use. Its 2.x line produced a different output layout and different block types -- not interchangeable, see #118. |
| Text+image embedding | **jinaai/jina-clip-v2**, 512-dim Matryoshka | `.venv-mineru`, GPU, ~2.8 GiB | Fixed by design. CC BY-NC 4.0: local use is non-commercial. |
| Reranking | **BAAI/bge-reranker-v2-m3** | product venv, GPU, ~1.5 GiB | Cross-encoder. Not an Ollama model. |
| Query decomposition, RECOMP, contextual | Ollama, `AUX_MODEL` | Ollama | Pinned for a whole run so a `--models` sweep varies only the generator. Shares the card with it -- see placement, above. |
| Answer generation | Ollama, `rag` role | Ollama | The only role the table varies. |

Since issue #222 the installed versions of these are recorded in each
artifact's `conditions` block rather than being remembered here.

---

## Superseded measurements

Kept because they are the only record of what was observed at the time, and
separated because none of them can be compared with the table above or with
each other. The gold set changed twice, and the second change also altered
what a percentage means.

**On a 32-case search set, 23 answered (2026-09-01, `20260901T023915Z`):**

| Model | Answered passed |
|---|---|
| `qwen3-coder-30b:latest` | 21 / 23 |
| `gemma4:e2b` | 20 / 23 |
| `gemma4:e4b` | 19 / 23 |

Paired over identical fragments, 21 of those 23 cases gave the same verdict
under all three models; only `dpo-model-scale` and `planck-h0` discriminated,
and two more failed under all three, making them system failures rather than
model failures. The conclusion drawn at the time -- that the generator was not
where the remaining failures were -- was drawn from two discriminating cases
in one language. Full analysis in issue #134.

**On an 83-case set, 63 answered, six English papers plus the blind set
(2026-09-03 to 2026-09-08):**

| Model | Answered passed | Run |
|---|---|---|
| `Ling-3.0-tiny` | 61 / 63 (96.8%) | `20260908T130135Z` |
| `Ling-3.0-tiny` | 56 / 63 (88.9%) | `20260903T124819Z` |
| `Granite-4.0-H-Tiny` | 53 / 63 (84.1%) | `20260903T134407Z` |
| `LFM2-8B-A1B` | 49 / 63 (77.8%) | `20260903T161257Z` |
| `Llama-3.2-3B-Instruct` | 49 / 63 (77.8%) | `20260903T115934Z` |
| `OLMoE-1B-7B` | 40 / 63 (63.5%) | `20260903T143414Z` |
| `Phi-mini-MoE-instruct` | 5 / 63 (7.9%) | `20260903T180057Z` |

The es and ca corpora did not exist in this set: every figure above describes
English-only behaviour, and the file did not say so at the time.

**One row on the 156-case set before the budget existed**
(`20260908T200256Z`): `Ling-3.0-tiny`, 120/136 (88.2%). Not comparable with
the current table because nothing bounded a generation then; the same model
under the 180 s budget scored 116/136 with six cases exhausting the cap,
which is the difference the budget makes rather than a regression.

### The MoE argument, and what it actually rests on

Earlier versions of this file argued at length that compact MoE models
outperform dense models on 8 GB, citing bandwidth per token and zero
offloading. The mechanism is sound and the second half is now directly
observed -- `qwen3:8b` at 5.2 GB never reached the GPU and decoded at 35.8
tok/s, against 95-213 for models that fit.

The accuracy half is not supported by the current table. The two best
measured models are `gemma4:e2b` and `qwen3:8b`, and the worst two
(`Phi-mini-MoE`, `OLMoE`) are both compact MoE. Architecture is not what
separates the top of this table from the bottom.

---

## Stack drift, which moved more than any generator swap

Same generator (`gemma4:e4b`), same 32 search-set cases, different extractor
version:

| When | Stack | Search set |
|---|---|---|
| Reference (per #30's comment) | MinerU 2.x line | 27 / 32 |
| 2026-09-01 | MinerU 3.4.5 | 26 / 32 |

Three flips: one recovered (`dpo-pipeline-figure-es`), two lost
(`dpo-model-scale`, `planck-h0`). The measured noise floor of that gate was
zero flips, so three was not noise.

One honest qualification, added after the three-model comparison: both "lost"
cases are among the two that discriminate between generators, so they sit
near a decision boundary. The drift is real; attributing those two
specifically to MinerU's version is not supportable. See issue #107. Both
figures are on the retired 32-case set.

---

## Hardware these numbers came from

RTX 4060 Laptop, 8 GB (7.6 GiB usable), Ryzen 9 8945HS, 30 GB RAM. Recorded
per run in the artifact's `conditions` block since #222; stated here because
every row currently in the tables above came from this one machine, and the
placement column only means anything against a known VRAM budget.
