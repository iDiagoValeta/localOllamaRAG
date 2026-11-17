"""Stage L: DPO on verifiable preference pairs, LoRA over a merged SFT model.

    BASE=models/<sft>/merged SAMPLES=data/samples_<tag>.jsonl RUN=dpo-x python l_dpo.py

Pairs come from k_sample.py and need no judge: the gate's grader already said
which student answers carry the reference value. Per question with at least
one failing sample:

- chosen: a passing student sample (on-policy), or, when the student never
  got it right, the SFT target for that question (data/sft.jsonl);
- rejected: a failing student sample.

The reference policy is BASE itself: its log-probabilities are computed once,
before training, with the adapter disabled, so no second model sits in VRAM.
The loss is DPO (sigmoid, beta) plus RPO_ALPHA times the chosen answer's
per-token NLL, which keeps the chosen likelihood from drifting down.
After training the adapter is merged and saved to models/<RUN>/merged.
"""
import json, math, os, random, sys, time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import environment, file_facts, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ["BASE"]
SAMPLES = os.environ["SAMPLES"]
RUN = os.environ["RUN"]
BETA = float(os.environ.get("BETA", "0.1"))
RPO_ALPHA = float(os.environ.get("RPO_ALPHA", "0.2"))
LR = float(os.environ.get("LR", "2e-5"))
R = int(os.environ.get("R", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
ACCUM = 8
MAX_LEN = int(os.environ.get("MAX_LEN", "14336"))
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SYSTEM = open(os.path.join(ROOT, "data", "system_prompt_rag.txt")).read()
OUT = os.path.join(ROOT, "models", RUN)
os.makedirs(os.path.join(ROOT, "runs", RUN), exist_ok=True)
LOG = open(os.path.join(ROOT, "runs", RUN, "dpo.jsonl"), "w")

random.seed(626)
tok = AutoTokenizer.from_pretrained(BASE)
# The fallback chosen answer is the row's training target, from the same file the
# SFT run learned from (SFT_DATA), with d_train.py's own keep rule.
SFT_DATA = os.environ.get("SFT_DATA", os.path.join(ROOT, "data", "sft.jsonl"))
sft = {r["id"]: {"target": (r.get("target") or r["teacher_answer"]).strip()}
       for r in map(json.loads, open(SFT_DATA)) if (r["keep"] if "keep" in r else r["teacher_pass"])}
pairs, why = [], {"on_policy": 0, "sft_target": 0, "solved": 0, "no_target": 0}
for r in map(json.loads, open(SAMPLES)):
    good = [s["text"] for s in r["samples"] if s["pass"]]
    if "<context></context>" in r["user_message"]:
        continue
    bad = [s["text"] for s in r["samples"] if not s["pass"] and s["text"]]
    if not bad:
        why["solved"] += 1
        continue
    if good:
        chosen = random.choice(good); why["on_policy"] += 1
    elif r["id"] in sft:
        chosen = sft[r["id"]]["target"]; why["sft_target"] += 1
    else:
        why["no_target"] += 1
        continue
    pairs.append((r["user_message"], chosen.strip(), random.choice(bad).strip()))
print(f"{len(pairs)} pairs {why}", flush=True)


def encode(user, answer):
    prompt = (f"<|im_start|>system\n\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    pi = tok(prompt, add_special_tokens=False)["input_ids"]
    ai = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return pi, ai


data = []
for user, c, rj in pairs:
    pc, ac = encode(user, c)
    _, ar = encode(user, rj)
    if len(pc) + max(len(ac), len(ar)) <= MAX_LEN:
        data.append((pc, ac, ar))

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=PROJ))


def seq_logp(pi, ai):
    """Summed log-probability of the answer tokens, and their count."""
    ids = torch.tensor([pi + ai], device="cuda")
    keep = torch.zeros_like(ids, dtype=torch.bool)
    keep[0, len(pi) - 1:len(pi) + len(ai) - 1] = True
    # A boolean logits_to_keep already returns (kept positions, vocab).
    logits = model(input_ids=ids, logits_to_keep=keep, use_cache=False).logits
    logits = logits.reshape(-1, logits.shape[-1]).float()
    assert logits.shape[0] == len(ai)
    lp = F.log_softmax(logits, -1).gather(1, torch.tensor(ai, device="cuda")[:, None]).squeeze(1)
    return lp.sum(), len(ai)


t0 = time.time()
model.eval()
ref = []
with torch.no_grad(), model.disable_adapter():
    for pi, ac, ar in data:
        ref.append((seq_logp(pi, ac)[0].item(), seq_logp(pi, ar)[0].item()))
ref_wall = time.time() - t0
print(f"reference log-probs for {len(data)} pairs in {ref_wall:.0f}s", flush=True)

model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
total = math.ceil(len(data) * EPOCHS / ACCUM)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total))) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
torch.cuda.reset_peak_memory_stats()
order = [i for _ in range(EPOCHS) for i in random.sample(range(len(data)), len(data))]
step, acc, hist = 0, [], []
for k, i in enumerate(order):
    pi, ac, ar = data[i]
    pc_lp, nc = seq_logp(pi, ac)
    pr_lp, _ = seq_logp(pi, ar)
    margin = BETA * ((pc_lp - ref[i][0]) - (pr_lp - ref[i][1]))
    loss = -F.logsigmoid(margin) + RPO_ALPHA * (-pc_lp / nc)
    (loss / ACCUM).backward()
    acc.append({"loss": loss.item(), "margin": margin.item(), "win": float(margin.item() > 0)})
    if (k + 1) % ACCUM == 0 or k + 1 == len(order):
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        ev = {k2: round(sum(a[k2] for a in acc) / len(acc), 4) for k2 in acc[0]}
        ev.update(step=step, lr=sched.get_last_lr()[0], t=round(time.time() - t0, 1),
                  vram=torch.cuda.max_memory_allocated())
        LOG.write(json.dumps(ev) + "\n"); LOG.flush(); hist.append(ev); acc = []
        if step % 10 == 0:
            print(f"[dpo] step {step}/{total} loss {ev['loss']} margin {ev['margin']} win {ev['win']}", flush=True)
wall = time.time() - t0
model = model.merge_and_unload()
model.save_pretrained(os.path.join(OUT, "merged"))
tok.save_pretrained(os.path.join(OUT, "merged"))
record(f"dpo:{RUN}", environment=environment(), base=BASE, samples=file_facts(SAMPLES), pairs=len(data), pair_sources=why,
       beta=BETA, rpo_alpha=RPO_ALPHA, lr=LR, lora_r=R, epochs=EPOCHS, accum=ACCUM, steps=step,
       wall_s=round(wall), ref_wall_s=round(ref_wall), vram_peak_bytes=torch.cuda.max_memory_allocated(),
       final=hist[-5:] if hist else None)
print("done", round(wall), "s")
