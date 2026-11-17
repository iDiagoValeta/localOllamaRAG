# Training: the MonkeyGrab 0.6B generator

This folder trains the answer model of the `rag-lite` profile: a
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) distilled from the
product's own generator, `qwen3:4b-instruct-2507`, on MonkeyGrab's exact
prompts. It is not product code: nothing under `rag/` or `src/` imports it.
It imports the product instead, so every question it trains on goes through
the same retrieval, prompt and grader the shipped app and the evaluation gate
use.

Published model: [nadiva1243/MonkeyGrab-Qwen3-0.6B-RAG-GGUF](https://huggingface.co/nadiva1243/MonkeyGrab-Qwen3-0.6B-RAG-GGUF)
(GGUF for Ollama) and [nadiva1243/MonkeyGrab-Qwen3-0.6B-RAG](https://huggingface.co/nadiva1243/MonkeyGrab-Qwen3-0.6B-RAG)
(merged safetensors weights; the trainer merges the LoRA and does not keep the
adapter on its own).

## Which build uses it

| Build | Answer model | Why |
|---|---|---|
| `develop` / `main` (8 GB GPU) | `qwen3:4b-instruct-2507-q4_K_M` | 135/136 answered on the gate; it fits an 8 GB card with the reranker and the embedder resident |
| `rag-lite` (4 GB GPU, `MONKEYGRAB_PROFILE=lite`) | this model in every Ollama role, with the reranker on | a 4 GB card cannot hold the 4B generator next to anything else; the 0.6B keeps most of its accuracy at a third of the memory. It was trained on reranked evidence: on the lite profile it answers 126/136 with the reranker (2676 MiB together) and 115 without, under the gate floor |

The 8 GB build does not switch: on that card the teacher fits and answers
eight more gate cases. The student exists for the machines where it does not.

## Result

Paired gate (`tests/eval/run_eval.py`, same retrieval, models compared in one
run, 136 answered cases of which 12 are study modes). VRAM is the peak
`size_vram` Ollama reports while answering, KV cache q8_0, 16k context (the gate's `rag_num_ctx`).

| Model | Answered /136 | Study /12 | Median s/case | VRAM |
|---|---|---|---|---|
| `qwen3:0.6b`, stock | 105 | 10 | | |
| **this model** (`sft-kd2`, Q4_K_M) | **126 to 127** (six runs) | **12** | 6.3 | **1.52 GB** |
| `qwen3:4b-instruct-2507` (teacher) | 135 | 12 | 9.6 | 3.90 GB |

The remaining failures are mostly a wrong number read from a table (an Adam
beta, an ImageNet top-1, a final black-hole mass). Every run, gate and finding
is in [EXPERIMENTS.md](EXPERIMENTS.md), with its date, wall time and memory.

## Recipe

- **Data** (3364 rows over 65 documents in English, Spanish and Catalan, none
  of them in the gate corpus, `scripts/heldout.json`):
  - extractive questions the teacher wrote about indexed chunks, answered by
    the teacher through the product's retrieval, and kept only when the gate's
    grader passes the answer;
  - Claude alignment: Haiku judged those answers against a rubric
    (`alignment/rubric.md`), Sonnet rewrote the fixable ones, Opus audited a
    sample of both;
  - Claude-written question and answer pairs aimed at tables and numbers
    (Sonnet wrote them, Haiku verified each one, Opus audited Haiku;
    `alignment/synth_prompt.md`);
  - the teacher's study artifacts (summary, outline, quiz) for every document
    and language, replayed so the short-answer data does not erase them.
- **Training**: LoRA r32 on every projection, loss on the answer tokens only,
  sequences up to 14336 tokens (the study prompts carry a whole document),
  2 epochs at lr 2e-4, plus logit distillation: the teacher's top-32
  log-probabilities at every answer position, loss
  `0.5 * CE + 0.5 * KL(teacher || student)`.
- **Export**: merged weights to GGUF with llama.cpp b11669, Q4_K_M, and the
  Modelfile of Ollama's own `qwen3:0.6b` (`Modelfile.qwen3-0.6b`), so the
  template, stop tokens and sampling match the stock model it was measured
  against.
- **Selection**: hyperparameters and data were chosen on a dev set (one
  held-out document per language, never trained on, `scripts/i_deveval.py`);
  only finalists went to the gate.

## What did not help

Each of these was measured; the numbers are in [EXPERIMENTS.md](EXPERIMENTS.md).

- **DPO** on graded on-policy samples: +13 factual on dev, a tie on the gate.
  The dev questions come from the same writers as the training questions, so
  dev rewards matching them; the gate's hand-written questions do not.
- **More teacher weight** (KL share 0.8): 10 gate cases lost, none gained. The
  graded, often rewritten target is better than the teacher's raw
  distribution.
- **A third epoch** (`sft-kd4`): +7 factual on dev, one case behind on the
  gate (126 against 127) for 50% more training time, and the held-out loss
  rose through the extra epoch. Same dev bias as DPO.
- **Q5_K_M or Q8_0 weights**: within one case of Q4_K_M, for more memory.
- **A q4_0 KV cache**: the answers degenerate on long contexts (57/136). q8_0
  is the floor for a model this small.

## Reproduce

Needs the product set up in this checkout (`tools/setup_environments.py`, the
corpora, Ollama with `qwen3:4b-instruct-2507-q4_K_M`, the embedder), a CUDA GPU
with 8 GB, [uv](https://docs.astral.sh/uv/) and a llama.cpp checkout in
`training/vendor/llama.cpp` for the GGUF conversion. Claude stages (E, F, M)
were run by Claude Code subagents over the batch files the scripts write;
they need no API key in the scripts.

```bash
cd training
uv sync                                         # training venv: torch cu129, transformers, peft
mkdir -p data && ../.venv/bin/python -c "import sys; sys.path[:0]=['..','../src']; import rag.chat_pdfs as r; open('data/system_prompt_rag.txt','w').write(r.SYSTEM_PROMPT_RAG)"
../.venv/bin/python scripts/fetch.py            # held-out corpus into training/docs/
cd scripts
../../.venv/bin/python a_index.py               # index it with the product's path
../../.venv/bin/python b_questions.py && ../../.venv/bin/python b2_filter.py
../../.venv/bin/python c_distill.py             # retrieval + teacher answers, graded
../../.venv/bin/python h_study.py               # teacher study artifacts
#   Claude stages: e_judge.py split|merge, f_rewrite.py, m_synth.py split|merge, n_synth_retrieve.py
../.venv/bin/python o_mix.py ../data/mix_c.jsonl
DATA=../data/mix_c.jsonl OUT=../data/kd_top32_c.pt ../.venv/bin/python j_teacher_logits.py
cd .. && scripts/run_exp.sh sft sft-kd2 DATA=$PWD/data/mix_c.jsonl KD=$PWD/data/kd_top32_c.pt MAX_LEN=14336 PROBES=6
```

`run_exp.sh` trains, exports to Ollama as `qwen06-rag:<run>` and runs the dev
evaluation. The gate then runs from the repository root:
`python tests/eval/run_eval.py --models qwen06-rag:sft-kd2`.

Every stage merges its facts (inputs, counts, wall time, VRAM, environment)
into `runs/manifest.json`; `runs/<run>/phases.json` holds the wall time of
each phase. `scripts/telar.py` serves a live view of a training run.

## Licence

Qwen3-0.6B and the teacher are Apache-2.0, and so is the published model. The
training data is not published: its contexts are verbatim passages of arXiv
papers, several of which do not allow redistribution, and of Wikipedia
articles. The scripts and `scripts/heldout.json` rebuild it from the sources.
