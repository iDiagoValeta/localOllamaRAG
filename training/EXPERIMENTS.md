# Qwen3-0.6B for MonkeyGrab RAG: experiment log

Goal (set 2026-11-15): the most effective 0.6B generator for MonkeyGrab that
fits in 8 GB. Done means, on the paired gate (`tests/eval/run_eval.py`, same
retrieval, models compared in one run):

- answered cases >= 120/136 (base0, the untouched model through the same
  export path, scores 108/136), and
- study modes (summary, outline, quiz) >= 10/12, so nothing the product ships
  regresses.

## Protocol

- **The gate is the exam, not the tuning set.** Hyperparameters and data
  choices are picked on the dev set (`scripts/i_deveval.py`): one held-out
  document per language (`lora.pdf`, `Volcán.pdf`, `Pirineus.pdf`) that no
  training set contains (`d_train.py` drops them). Only finalists go to the gate.
- Every model reaches Ollama through the same path (`scripts/g_export.sh`:
  merged HF to bf16 GGUF to Q4_K_M, Ollama's own qwen3:0.6b Modelfile), so a
  difference between rows is the training, not the conversion.
- Every stage writes its facts (inputs, counts, wall time, VRAM, environment)
  to `runs/manifest.json`; training streams per step to `runs/<run>/live.jsonl`
  (viewable with `scripts/telar.py`).
- Hardware: RTX 4060 Laptop 8 GB, the display holds about 1.4 GB of it.

## Data sources

| Source | Stage | What it gives |
|---|---|---|
| Held-out corpus, 26 docs (en 10, es 8, ca 8) | fetch, a_index | product-indexed chunks, none in the gate corpus |
| Teacher-written extractive questions | b_questions, b2_filter | 1072 questions, 863 after filtering |
| Teacher RAG answers (qwen3:4b-instruct-2507) | c_distill | 863 answers, 696 pass the gate grader |
| Claude alignment (Haiku judge, Sonnet rewrite, Opus audit) | e_judge, f_rewrite | 596 kept targets (55 good, 541 rewritten) |
| Teacher study artifacts | h_study | summary/outline/quiz per doc and language, graded by grade_study |
| Corpus extension, +40 docs (2026-11-15) | fetch | arXiv with tables, Wikipedia en/es/ca |
| Claude-written training pairs (Sonnet, 12 subagents) | m_synth | 415 items over 276 chunks of training docs, tables and numbers first; 0 spans not verbatim |
| Haiku verification (10 subagents) | m_synth | 389 kept, 25 dropped |
| Opus audit of Haiku (40 items) | m_synth | agrees 35/40; 0 wrong items kept, 5 valid items dropped (Haiku errs strict) |
| Claude pairs for the 39 new documents (Sonnet, 10 subagents; Haiku, 10) | m_synth --only-new | 468 chunks, 579 items written in about 2 min per batch; Haiku keeps 536 (drops 'según la tabla' references, hedged values like 'casi 200', cross-chunk answers) |
| Teacher data for the 36 new documents (2026-11-16) | b_questions, b2_filter, c_distill, h_study | 1588 questions written (340 dropped) in 32 min; 2169 in all after filtering; 1306 new distilled answers, 1026 pass (2 h 28 min); 351 new study artifacts, 343 pass (93 min, about 16 s each) |
| Claude-written hard test set (Opus, 3 subagents) | m_synth | 169 items on the dev documents, scored by i_deveval as `hard` |

## Runs

| Run | Data | Method | Train wall | Peak VRAM | Dev factual /124 | Dev hard /169 | Dev study /27 | Gate answered | Gate study | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:0.6b | | Ollama stock | | | | | | 104/136 | 10/12 | |
| base0 | | no training, same export | | | 59 | 83 | 24 | 108/136 | 10/12 | baseline; dev eval 487 s |
| sft-r32 | 596 Claude-aligned factual | LoRA r32, 2 ep, lr 2e-4, MAX_LEN 8192 | 29 min | 2.81 GiB | 86 *(saw dev docs)* | 108 *(saw dev docs)* | 6 | 108/136 | 4/12 | +6 factual, forgot study schemas (1 section, 1 question) |
| sft-a | 509 factual + 202 study (teacher), dev docs out | LoRA r32, 2 ep, lr 2e-4, MAX_LEN 14336 | 71 min (trainer), 143 min train phase | 4.81 GiB reserved | 77 | 109 | 27 | | | study replay cures the amnesia; +18 factual over base0 on unseen docs |
| sft-b | mix_a + 373 Claude pairs (Sonnet, Haiku-verified) | same as sft-a | 93 min | 4.81 GiB reserved | 77 | 119 | 27 | 120/136 | 9/12 | Claude pairs add +10 on hard, nothing on factual |
| sft-kd | mix_b, teacher top-32 log-probs | sft-b + logit KD (alpha 0.5) | 93 min + 44 min teacher pass | 4.81 GiB reserved | 80 | 120 | 27 | **125/136** | **12/12** | KD free at train time; +3/+1 over sft-b, inside noise (finding 11) |
| dpo-kd | 166 on-policy pairs from sft-kd | DPO, beta 0.1, RPO 0.2, lr 2e-5, LoRA r16, 1 ep (21 steps) | 10 min | 3.42 GiB | 80 | 123 | 25 | 120/136 | 11/12 | margins stay near 0.01: too few pairs and steps to move the policy |
| sft-kd2 | mix_c (65 docs, 2821 train rows after dev hold-out and length), teacher top-32 | sft-kd recipe | 4 h 37 min (706 steps) + 68 min teacher pass | 5.92 GB allocated peak | 82 | 131 | 27 | **126, 127/136** | **12/12** | +11 hard over sft-kd, factual +2 (noise), study intact |
| dpo-kd2 | 525 on-policy pairs from sft-kd2 over mix_c (453 on-policy chosen, 72 SFT target) | DPO, beta 0.1, RPO 0.2, lr 5e-5, LoRA r16, 2 ep (132 steps) | 49 min sampling + 65 min DPO | 3.99 GB | 95 | 148 | 27 | 126/136 | 11/12 | margins 1.4 to 2.9 by the end, win rate near 1 |
| sft-kd3 | mix_c, teacher top-32 | sft-kd2 with KD alpha 0.8 | 4 h 35 min | | 86 | 128 | 27 | 117/136 | 12/12 | 0 cases gained, 10 lost against sft-kd2 on the same run (finding 23) |
| sft-kd4 | mix_c, teacher top-32 | sft-kd2 with a third epoch (1059 steps) | 6 h 51 min | 5.92 GB allocated peak | 89 | 136 | 27 | 126/136 | 12/12 | dev +7 factual over sft-kd2, gate one case behind (finding 24); eval loss rose through the third epoch, 0.50 to 0.54 |
| qwen3:4b teacher | | | | | | | | 135/136 | 12/12 | ceiling |

## Findings

1. **Short-answer SFT erases structured output.** sft-r32 gains 6 factual cases
   (Catalan 17 to 25 of 28) but study drops 10/12 to 4/12: the artifacts come
   back with 1 section, 1 question or 5 titles. Replay of study artifacts is
   the first fix tried (sft-study).
2. **Study prompts are long.** The product feeds the whole document's budget:
   about 12.2k prompt tokens, so MAX_LEN has to go from 8192 to 14336.
3. Remaining factual failures of sft-r32 are mostly picking the wrong number
   from a table (BLEU 25.8 for 28.4, HellaSwag 76.2 for 84.2) and a few format
   misses (3:4:5 for 3-4-5, "ICLR in 2015").
4. **The dev set measures the same amnesia.** sft-r32 drops study from 24/27
   (base0) to 6/27 on documents the gate never sees, so the dev set can stand
   in for the gate's study cases when choosing between runs.
5. **Synthetic-data yield.** Of 389 Haiku-kept Sonnet items, 373 survive the
   product's retrieval (the value has to reach the context) and the grader on
   the ideal answer; 16 lose their value at retrieval. All 169 Opus dev items
   keep it. Retrieval of 1,000 items took 45 min.
6. **Corpus extension is the slow source.** MinerU took about an hour for 3 of
   the 40 new documents (long Wikipedia articles), so the other 36 are parked
   in `docs_pending/` and indexed in a separate pass, not inside an experiment.
7. **VRAM fragmentation.** Training with 300 to 14k-token sequences allocates
   at most 3.05 GiB but the caching allocator reserves 6.1 GB; runs after
   sft-a set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
8. **Study replay fixes the amnesia without costing factual.** sft-a (factual
   plus the teacher's study artifacts, dev documents held out) scores study
   27/27 on dev, above base0's 24, while factual goes 59 to 77 and hard 83 to
   109 on documents it never saw. Clean dev numbers, unlike sft-r32's.
9. **Claude's synthetic pairs help where they were aimed.** sft-b adds the 373
   Sonnet pairs (tables and numbers first) to sft-a's mix: hard dev 109 to 119,
   factual flat at 77, study still 27/27. mix_b becomes the base for the next
   methods (logit distillation, DPO).
10. **Teacher logits, first pass discarded.** With a boolean `logits_to_keep`
   the model already returns (kept positions, vocab); indexing `[0]` kept one
   position per row (912 KB for 1197 rows). d_train's position assert would
   have caught it; j_teacher_logits.py now asserts the count itself. The pass
   costs 44 min at a 3.68 GiB peak (teacher NF4, 162,586 answer tokens).
11. **Dev-eval noise is about 2 cases.** Re-running the dev eval on sft-b gives
   factual 79 and hard 117 against 77 and 119 the first time (study 27 both).
   Differences under 3 cases between runs are not evidence; sft-kd's lead over
   sft-b is inside that band.
12. **Empty context: every model invents an answer.** 116 of 863 training
   questions reach the generator with an empty `<context>` (the product's
   retrieval kept nothing). On one of them base0, sft-b and sft-kd all state
   a made-up year at temperature 0 and 0.8, and sft-kd at 0.8 can loop until
   Ollama aborts with `token repeat limit reached` (HTTP 500). The gate has no
   unanswerable cases, so it cannot see this; DPO sampling skips those rows
   because no answer can be right for the right reason.
13. **sft-kd is already right most of the time on its own training questions.**
   Four samples at temperature 0.8 on the 633 non-empty, non-dev questions
   pass 83% (1056/1272, 4 min). 427 questions pass all four, so DPO gets only
   166 pairs (152 on-policy chosen, 14 with the SFT target as chosen).

## Paired gate, finalists (run `20261116T072827Z`, 22.4 min)

Same retrieval for every model, 156 cases (136 answered by the generator: 124
factual plus 12 study; 20 retrieval-only).

| Model | Answered /136 | of which factual /124 | Study /12 | Median s/case |
|---|---|---|---|---|
| base0 | 104 | 93 | 11 | 6.46 |
| sft-b | 120 | 111 | 9 | 6.29 |
| **sft-kd** | **125** | **113** | **12** | 6.29 |
| dpo-kd | 120 | 109 | 11 | 6.29 |

sft-kd clears both bars (>= 120 answered, >= 10/12 study). base0 scored 108
in the earlier paired run and 104 here, so the gate itself moves about 4
cases between runs; a second, independent run confirms before calling it.
The dev set ranked the same way (sft-kd best, DPO no help), which supports
using it to choose runs.

## Confirmation run (`20261116T074659Z`, 18.2 min)

| Model | Answered /136 | of which factual /124 | Study /12 |
|---|---|---|---|
| qwen3:0.6b (Ollama stock) | 103 | 94 | 9 |
| **sft-kd** | **122** | **110** | **12** |

sft-kd clears the bar in two independent paired runs (125 and 122, mean
123.5; study 12/12 both times), +19 over stock qwen3:0.6b in the same run.
The margin over 120 is about the gate's own run-to-run spread, so the next
round (corpus extension to 65 documents) aims to widen it.
14. **MinerU is fast when it has the GPU.** The 36 parked documents indexed in
   16.5 min (en 19 docs 565 s, es 10 docs 256 s, ca 7 docs 164 s): 65
   documents, 6,422 chunks. The hour for 3 documents in finding 6 was GPU
   contention with a training run, not MinerU.
15. **Same file name in two corpora.** Leonardo_da_Vinci, Ruta_de_la_Seda and
   Vacuna exist in es and ca (Leonardo also in en) with identical chunk ids;
   m_synth.py now prefixes the language to those chunk ids.
16. **mix_c, the whole corpus.** sft.jsonl, 1020 new distilled answers (one
   empty-context pass dropped), 585 study rows and 896 Claude pairs (523 new
   survive retrieval out of 536, 46 min): 3364 rows, 2970 trainable outside
   the dev documents, 2.5 times mix_b. The teacher pass resumes from
   kd_top32.pt and adds 1886 rows in 68 min (398,315 answer tokens in all,
   peak 4.85 GiB).
17. **The embedder cannot share the GPU with training.** sft-kd2's first
   attempt ran out of memory at step 256 of 706 (1 h 39 min lost) asking for
   1 GiB while the embedder held 1.2 GB. run_exp.sh now stops it while
   training and starts it again for the dev evaluation.
18. **Side by side on CPU (9 Opus dev questions, temperature 0, llama-server
   with the GPU hidden while sft-kd2 trained).** Stock qwen3:0.6b 2/9, sft-kd
   7/9, teacher 8/9; mean 19.1 s, 17.6 s and 80.2 s per answer. sft-kd answers
   in one sentence, in the question's language and with the context's
   spelling ("24 000", "diez mil", "tres"); the stock model reformats numbers
   ("24,000", "10,000") and invents table values; the teacher is right but
   verbose (quotes, bold, one step-by-step reply in English to a Spanish
   question). sft-kd's two misses are the wrong cell of a LoRA paper table,
   the failure mode of finding 3.

## Paired gate, corpus extension (run `20261116T212506Z`, 29.6 min)

| Model | Answered /136 | of which factual /124 | Study /12 | Median s/case | Ollama VRAM |
|---|---|---|---|---|---|
| sft-kd | 119 | 107 | 12 | 6.53 | 1.52 GB |
| **sft-kd2** | **126** | **114** | **12** | 6.63 | **1.52 GB** |
| qwen3:4b-instruct-2507 (teacher) | 135 | 123 | 12 | 9.59 | 3.90 GB |

VRAM is the peak `size_vram` Ollama reports for each model while the gate
runs it (polled every 2 s), so it includes the KV cache at the product's
context size, not just the weights (0.4 GB against 2.5 GB). sft-kd2 against
sft-kd case by case: 10 cases only sft-kd2 passes, 3 only sft-kd. sft-kd's
three gate runs give 125, 122 and 119, so a 7-case gap in one run is above
the noise of about 4 but a second run confirms it.

## Confirmation run (`20261116T214411Z`, 18.8 min)

| Model | Answered /136 | of which factual /124 | Study /12 | Median s/case |
|---|---|---|---|---|
| qwen3:0.6b (Ollama stock) | 105 | 95 | 10 | 6.47 |
| **sft-kd2** | **127** | **115** | **12** | 6.30 |

sft-kd2 scores 126 and 127 in two independent paired runs (mean 126.5,
study 12/12 both times) against sft-kd's 125, 122 and 119 (mean 122): the
corpus extension is worth about 4.5 cases on the gate and 11 on the hard dev
set. It is the new best, 8.5 cases below the teacher at 39% of its VRAM.
19. **DPO works once it has enough pairs and steps.** Sampling sft-kd2 four
   times at temperature 0.8 on the 2549 non-dev factual questions of mix_c
   passes 88.5% (9024/10196, 49 min); 1984 questions pass all four, leaving
   525 pairs, three times dpo-kd's 166. With lr 5e-5 and 2 epochs (132 steps
   against 21) the margin grows from 0.01 to about 2 and dev moves factual 82
   to 95 and hard 131 to 148, study still 27/27. No dev document is among the
   sampled questions (checked by source).

## Paired gate, DPO round 2 (run `20261117T000623Z`, 19.4 min)

| Model | Answered /136 | of which factual /124 | Study /12 | Median s/case |
|---|---|---|---|---|
| **sft-kd2** | **127** | 115 | **12** | 6.29 |
| dpo-kd2 | 126 | 115 | 11 | 6.69 |

20. **The dev gain of DPO does not reach the gate.** dpo-kd2 is +13 factual
   and +17 hard on dev but ties sft-kd2 on the gate (126 against 127: 3 cases
   gained, 4 lost) and drops one study quiz to a single question, since DPO
   saw only factual pairs. The dev questions come from the same writers as the
   training questions (the teacher's b_questions and Opus/Sonnet), and DPO
   optimises against the same grader on that same style, so dev rewards it
   for matching the writers; the gate's hand-written questions do not. Dev
   still ranked the SFT runs the way the gate did, so it stays the filter for
   data and SFT changes, but a method that trains on graded samples needs the
   gate itself (or hand-written dev questions) to be judged.

## Inference variants: weight quantization and KV cache (runs `20261117T002914Z`, 20.3 min; `20261117T011404Z`, 44.8 min)

Same merged sft-kd2 weights, exported at three quantizations with
`scripts/g_export.sh sft-kd2 <QUANT>`. VRAM is the peak `size_vram` Ollama
reports for the model during the gate (system Ollama: flash attention, KV
cache q8_0, 16k context from the gate's `rag_num_ctx`). The KV q4_0 run used a private `ollama serve` on
port 11435 with `OLLAMA_KV_CACHE_TYPE=q4_0` and everything else equal.

| Variant | GGUF | Answered /136 | Study /12 | Median s/case | VRAM |
|---|---|---|---|---|---|
| **Q4_K_M, KV q8_0** | 397 MB | **126** | **12** | 6.3 | **1.52 GB** |
| Q5_K_M, KV q8_0 | 444 MB | 125 | 12 | 6.3 | 1.57 GB |
| Q8_0, KV q8_0 | 639 MB | 126 | 12 | 6.3 | 1.73 GB |
| Q4_K_M, KV q4_0 | 397 MB | 57 | 0 | 23.6 | 1.05 GB |

21. **Weight precision above Q4_K_M buys nothing.** Q5_K_M and Q8_0 land
   within one case of Q4_K_M, inside the gate's noise, for 50 to 210 MB more.
   The LoRA merge survives 4-bit quantization intact; Q4_K_M stays.
22. **A 4-bit KV cache breaks the 0.6B model.** With q4_0 keys and values the
   answers degenerate on long contexts (stray Arabic tokens, a leaked
   `</think>`, wrong numbers, one-question quizzes), the pass rate halves and
   generation is four times slower because the answers run on. It saves
   0.47 GB, which is not worth it; q8_0, the system default, is the floor.
   A model this small has little redundancy in its attention to absorb the
   cache error that a 4B model tolerates.

## Paired gate, teacher weight (run `20261117T061317Z`, 17.9 min)

| Model | Answered /136 | Study /12 | Median s/case |
|---|---|---|---|
| **sft-kd2** (alpha 0.5) | **127** | 12 | 6.29 |
| sft-kd3 (alpha 0.8) | 117 | 12 | 6.26 |

23. **More teacher weight hurts.** Raising the KL share from 0.5 to 0.8 ties
   on dev (factual +4, hard -3, inside the noise of finding 11) but loses 10
   gate cases and gains none: dpo-acronym, gw-site, planck-sigma8-es,
   llama-65b-hellaswag, adam-beta2, gaudi-montserrat, basquet-naixement,
   revindustrial-hobsbawm, catala-familia, immunitari-fagolisosoma, six of
   them Spanish or Catalan. The CE target is the graded, often
   Claude-rewritten answer, which is better than the teacher's raw
   distribution; at 0.8 the student copies the teacher's habits, including
   its errors, and the weaker 4B Catalan shows most. sft-kd2 scores 127 again,
   its third paired gate (126, 127, 127).

## Paired gate, third epoch (run `20261117T133043Z`, 18.4 min)

| Model | Answered /136 | Study /12 | Median s/case |
|---|---|---|---|
| **sft-kd2** (2 epochs) | **127** | 12 | 6.2 |
| sft-kd4 (3 epochs) | 126 | 12 | 6.3 |

24. **A third epoch moves dev, not the gate.** sft-kd4 beats sft-kd2 on dev
   (factual 89 against 82, hard 136 against 131) while its held-out eval
   loss climbs through the third epoch, from 0.50 at the end of epoch 2 to
   0.54. On the gate the two swap cases: sft-kd2 alone passes gw-site,
   planck-sigma8-es, bert-large-layers, llama-65b-hellaswag and
   basquet-naixement; sft-kd4 alone passes vit-imagenet-top1, adam-beta2,
   revindustrial-hobsbawm and dali-primera-exposicio. Net, one case behind
   for 50% more training time. Dev rewards the third epoch for the same reason
   it rewarded DPO (finding 20): its questions come from the same writers as
   the training questions. sft-kd2 scores 127 again: 126 or 127 on every one of its six
   paired gates, and it stays the model. Training stops here.

## Where it stands (2026-11-17)

**sft-kd2 is the model**: 126 or 127/136 on six paired gates
(study 12/12 every time), against 105 for stock qwen3:0.6b and 135 for the teacher, at 1.52 GB
of VRAM in Ollama against the teacher's 3.90 GB. The recipe: LoRA r32 SFT
with answer-only loss and the teacher's top-32 logits (alpha 0.5), over the
teacher's graded answers, Claude-written table and number pairs and replayed
study artifacts for 65 documents, dev documents held out.
