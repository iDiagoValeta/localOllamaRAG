"""Stage D: LoRA SFT of Qwen3-0.6B on the distilled pairs, in Ollama's exact prompt format.

The prompt is what Ollama's qwen3 template renders for /api/generate with a
system prompt and think=False, so the adapter is trained on the bytes the
product sends at inference.

VRAM: the loss only needs the answer's logits, so the LM head projects only
those positions (``logits_to_keep`` as a boolean mask). Projecting the whole
8k-token prompt onto the 152k vocabulary costs about 2.5 GB in bf16 plus its
fp32 upcast; the answer is a few hundred tokens.

Every optimizer step is streamed to ``runs/<RUN>/live.jsonl`` for the Telar
viewer (``scripts/telar.py``) and the run's facts go to ``runs/manifest.json``.

Env knobs: BASE, RUN, R, EPOCHS, LR, MAX_LEN, DATA, MAX_STEPS, N_ROWS, PROBES.
"""
import json, os, random, sys, time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import environment, file_facts, now, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("BASE", "Qwen/Qwen3-0.6B")
RUN = os.environ.get("RUN", "lora-r32")
OUT = os.path.join(ROOT, "models", RUN)
LIVE_DIR = os.path.join(ROOT, "runs", RUN)
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
R = int(os.environ.get("R", "32"))
DATA = os.environ.get("DATA", os.path.join(ROOT, "data", "distilled.jsonl"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
N_ROWS = int(os.environ.get("N_ROWS", "0"))
N_PROBES = int(os.environ.get("PROBES", "3"))
# Logit distillation: KD names a file from j_teacher_logits.py holding, per row
# id, the teacher's top-k log-probabilities at every answer position. The loss
# becomes (1 - KD_ALPHA) * CE + KD_ALPHA * KL(teacher top-k || student).
KD = torch.load(os.environ["KD"]) if os.environ.get("KD") else None
KD_ALPHA = float(os.environ.get("KD_ALPHA", "0.5"))
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SYSTEM = open(os.path.join(ROOT, "data", "system_prompt_rag.txt")).read()


def render(user, answer=None, system=None):
    prompt = (f"<|im_start|>system\n\n{system or SYSTEM}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return prompt if answer is None else (prompt, answer + "<|im_end|>")


def target(row):
    """The answer to learn: the rewrite when alignment produced one, else the teacher's."""
    return (row.get("target") or row["teacher_answer"]).strip()


def kept(row):
    return row["keep"] if "keep" in row else row["teacher_pass"]


tok = AutoTokenizer.from_pretrained(BASE)
rows = [json.loads(line) for line in open(DATA)]
rows = [r for r in rows if kept(r)]
DEV = set(json.loads(os.environ.get("DEV_DOCS_JSON", "null")) or ["lora.pdf", "Volcán.pdf", "Pirineus.pdf"])
held = sum((r.get("source") or r.get("doc")) in DEV for r in rows)
rows = [r for r in rows if (r.get("source") or r.get("doc")) not in DEV]
print(f"{held} rows of the dev documents held out", flush=True)
random.seed(626)
random.shuffle(rows)
if N_ROWS:
    rows = rows[:N_ROWS]


class SFT(Dataset):
    def __init__(self, rows):
        self.items, self.rows, self.dropped = [], [], 0
        for r in rows:
            p, a = render(r["user_message"], target(r), r.get("system"))
            pi, ai = tok(p, add_special_tokens=False)["input_ids"], tok(a, add_special_tokens=False)["input_ids"]
            if len(pi) + len(ai) > MAX_LEN:
                self.dropped += 1
                continue
            self.items.append((pi + ai, [-100] * len(pi) + ai))
            self.rows.append(r)
            if KD is not None and r["id"] in KD:
                assert KD[r["id"]][0].shape[0] == len(ai), f"teacher positions disagree for {r['id']}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, lab = self.items[i]
        item = {"input_ids": ids, "labels": lab}
        if KD is not None and self.rows[i]["id"] in KD:
            item["kd"] = KD[self.rows[i]["id"]]
        return item


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
            **({"kd_ids": torch.cat([b["kd"][0] for b in batch]).long(), "kd_logp": torch.cat([b["kd"][1] for b in batch]).float()}
               if all("kd" in b for b in batch) else {})}


def answer_logits(model, input_ids, attention_mask, labels):
    """Logits at the positions that predict an answer token, and those tokens."""
    keep = torch.zeros_like(labels, dtype=torch.bool)
    keep[:, :-1] = labels[:, 1:] != -100
    out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep, use_cache=False)
    return out.logits.reshape(-1, out.logits.shape[-1]), labels[:, 1:][keep[:, :-1]]


class AnswerOnlyTrainer(Trainer):
    tokens_seen = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits, targets = answer_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        AnswerOnlyTrainer.tokens_seen += int(inputs["attention_mask"].sum())
        loss = F.cross_entropy(logits.float(), targets)
        if "kd_ids" in inputs:
            s_logp = F.log_softmax(logits.float(), -1).gather(1, inputs["kd_ids"])
            t_logp = F.log_softmax(inputs["kd_logp"], -1)  # renormalised over the teacher's top-k
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            loss = (1 - KD_ALPHA) * loss + KD_ALPHA * kl
        return (loss, {"loss": loss}) if return_outputs else loss


n_eval = max(20, len(rows) // 20)
train, evalset = SFT(rows[n_eval:]), SFT(rows[:n_eval])
print(f"train {len(train)} eval {len(evalset)} (of {len(rows)} kept pairs, "
      f"{train.dropped + evalset.dropped} over {MAX_LEN} tokens)", flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=PROJ))
trainable, total = model.get_nb_trainable_parameters()
print(f"trainable {trainable:,} of {total:,}", flush=True)


def lora_modules(model):
    """LoRA layers in loom order: layer by layer, projections in PROJ order."""
    found = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            layer = int(name.split(".layers.")[1].split(".")[0])
            found[(layer, PROJ.index(name.rsplit(".", 1)[1]))] = mod
    return [found[k] for k in sorted(found)]


class Telar(TrainerCallback):
    """Streams the run to live.jsonl: steps, the loom of LoRA updates, and probe answers."""

    def __init__(self, model, probes):
        self.model, self.probes = model, probes
        self.mods = lora_modules(model)
        with torch.no_grad():
            self.base_norm = [m.base_layer.weight.float().norm().item() for m in self.mods]
        self.grads, self.last_t, self.last_tokens, self.pending = None, None, 0, {}
        os.makedirs(LIVE_DIR, exist_ok=True)
        self.path = os.path.join(LIVE_DIR, "live.jsonl")
        open(self.path, "w").close()

    def emit(self, **event):
        with open(self.path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kw):
        free, total_mem = torch.cuda.mem_get_info()
        self.emit(type="meta", run=RUN, base=BASE, teacher="qwen3:4b-instruct-2507-q4_K_M", r=R, alpha=2 * R,
                  lr=args.learning_rate, epochs=args.num_train_epochs, accum=args.gradient_accumulation_steps,
                  max_steps=state.max_steps, n_train=len(train), n_eval=len(evalset), trainable=trainable,
                  total_params=total, layers=max(1, len(self.mods) // len(PROJ)), proj=PROJ,
                  vram_total=total_mem, started=now(),
                  probes=[{k: p[k] for k in ("id", "lang", "question", "reference", "teacher", "tokens")}
                          for p in self.probes])
        self.last_t = time.time()
        self.probe(0, None)

    def on_pre_optimizer_step(self, args, state, control, **kw):
        g = []
        for m in self.mods:
            a, b = m.lora_A["default"].weight.grad, m.lora_B["default"].weight.grad
            g.append(((a.float().norm() ** 2 if a is not None else 0) +
                      (b.float().norm() ** 2 if b is not None else 0)) ** 0.5)
        self.grads = [float(x) for x in g]

    def on_step_end(self, args, state, control, **kw):
        with torch.no_grad():
            weave = []
            for m, wn in zip(self.mods, self.base_norm):
                a, b = m.lora_A["default"].weight.float(), m.lora_B["default"].weight.float()
                sq = torch.trace((b.T @ b) @ (a @ a.T)).clamp_min(0).sqrt() * m.scaling["default"]
                weave.append(float(sq) / wn)
        t = time.time()
        tokens = AnswerOnlyTrainer.tokens_seen - self.last_tokens
        free, total_mem = torch.cuda.mem_get_info()
        self.pending = dict(step=state.global_step, weave=[round(x, 7) for x in weave],
                            grads=[round(x, 6) for x in (self.grads or [])], tokens=tokens,
                            tok_s=round(tokens / max(t - self.last_t, 1e-6), 1), sec=round(t - self.last_t, 2),
                            vram_alloc=torch.cuda.memory_allocated(), vram_peak=torch.cuda.max_memory_allocated(),
                            vram_device=total_mem - free)
        torch.cuda.reset_peak_memory_stats()
        self.last_t, self.last_tokens = t, AnswerOnlyTrainer.tokens_seen

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        if "loss" in logs and self.pending:
            self.emit(type="step", epoch=round(state.epoch or 0, 4), loss=logs["loss"], lr=logs.get("learning_rate"),
                      grad_norm=logs.get("grad_norm"), t=round(time.time(), 1), **self.pending)
            self.pending = {}

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        self.probe(state.global_step, (metrics or {}).get("eval_loss"))

    def on_train_end(self, args, state, control, **kw):
        self.emit(type="done", step=state.global_step, finished=now())

    def probe(self, step, eval_loss):
        """Surprisal of each reference token, and what the model says greedily, at this step."""
        was_training = self.model.training
        self.model.eval()
        out = []
        with torch.no_grad():
            for p in self.probes:
                ids = torch.tensor([p["ids"]], device="cuda")
                logits, targets = answer_logits(self.model, ids, torch.ones_like(ids), torch.tensor([p["labels"]], device="cuda"))
                nll = F.cross_entropy(logits.float(), targets, reduction="none")
                prompt = torch.tensor([p["prompt_ids"]], device="cuda")
                gen = self.model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=96,
                                          do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
                text = tok.decode(gen[0, prompt.shape[1]:], skip_special_tokens=True).strip()
                out.append({"id": p["id"], "nll": [round(float(x), 3) for x in nll], "generated": text})
        if was_training:
            self.model.train()
        self.emit(type="eval", step=step, eval_loss=eval_loss, probes=out)


def pick_probes(ds, n):
    """One eval item per language and task kind first, shortest prompts, so probing stays cheap."""
    order = sorted(range(len(ds)), key=lambda i: len(ds.items[i][0]))
    chosen, seen = [], set()
    for i in order:
        key = ds.rows[i].get("kind", "factual")
        key = (key, ds.rows[i]["lang"]) if key == "factual" else (key,)
        if key not in seen:
            chosen.append(i); seen.add(key)
    chosen += [i for i in order if i not in chosen]
    probes = []
    for i in chosen[:n]:
        ids, lab = ds.items[i]
        cut = next(j for j, x in enumerate(lab) if x != -100)
        row = ds.rows[i]
        probes.append({"id": f"{row['lang']}:{i}", "lang": row["lang"], "question": row.get("question") or row["kind"],
                       "reference": row.get("answer", ""), "teacher": target(row), "ids": ids, "labels": lab,
                       "prompt_ids": ids[:cut], "tokens": [tok.decode([t]) for t in ids[cut:]]})
    return probes


args = TrainingArguments(output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                         num_train_epochs=float(os.environ.get("EPOCHS", "2")), max_steps=MAX_STEPS,
                         learning_rate=float(os.environ.get("LR", "2e-4")), lr_scheduler_type="cosine",
                         warmup_steps=0.05, bf16=True, logging_steps=1, eval_strategy="steps",
                         eval_steps=int(os.environ.get("EVAL_STEPS", "25")), save_strategy="no", report_to=[],
                         per_device_eval_batch_size=1, prediction_loss_only=True, dataloader_num_workers=0, seed=626)
trainer = AnswerOnlyTrainer(model=model, args=args, train_dataset=train, eval_dataset=evalset, data_collator=collate,
                            callbacks=[Telar(model, pick_probes(evalset, N_PROBES))])
trainer.loss_is_scaled_for_ga = False  # our loss is a per-micro-batch mean; let the Trainer divide by accumulation

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
result = trainer.train()
wall = time.time() - t0
merged = model.merge_and_unload()
merged.save_pretrained(os.path.join(OUT, "merged"), safe_serialization=True)
tok.save_pretrained(os.path.join(OUT, "merged"))
print("saved", os.path.join(OUT, "merged"), flush=True)

steps = [json.loads(line) for line in open(os.path.join(LIVE_DIR, "live.jsonl"))]
peaks = [e["vram_peak"] for e in steps if e["type"] == "step"]
record(f"sft:{RUN}", environment=environment(), base=BASE, data=file_facts(DATA), n_train=len(train),
       n_eval=len(evalset), dropped_over_max_len=train.dropped + evalset.dropped, max_len=MAX_LEN,
       lora={"r": R, "alpha": 2 * R, "dropout": 0.05, "targets": PROJ, "trainable": trainable, "total": total},
       optim={"lr": args.learning_rate, "schedule": "cosine", "warmup_ratio": 0.05, "epochs": args.num_train_epochs,
              "batch": 1, "accum": 8, "bf16": True, "grad_checkpointing": True, "loss": "answer tokens only",
              "seed": 626, "kd": os.environ.get("KD"), "kd_alpha": KD_ALPHA if KD is not None else None},
       dev_docs_held_out=held,
       wall_seconds=round(wall, 1), global_steps=result.global_step, train_loss=result.training_loss,
       tokens_seen=AnswerOnlyTrainer.tokens_seen, tokens_per_second=round(AnswerOnlyTrainer.tokens_seen / wall, 1),
       vram_peak_bytes=max(peaks) if peaks else None, log_history=trainer.state.log_history,
       live=file_facts(os.path.join(LIVE_DIR, "live.jsonl")), merged=os.path.relpath(os.path.join(OUT, "merged"), ROOT))
print(f"recorded sft:{RUN} wall {wall:.0f}s peak {max(peaks) / 2**30 if peaks else 0:.2f} GiB", flush=True)
