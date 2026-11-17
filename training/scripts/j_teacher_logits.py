"""Stage J: the teacher's top-k log-probabilities at every answer position.

    DATA=data/train_mix.jsonl OUT=data/kd_top32.pt python j_teacher_logits.py

Qwen3-4B-Instruct-2507 and Qwen3-0.6B share one tokenizer, so the teacher's
distribution over the student's exact answer tokens is a dense target: per
token, not just the one token the text happened to use. The teacher runs in
4-bit NF4 (bitsandbytes) to fit next to nothing else in 8 GB, and only the
answer positions reach the LM head (``logits_to_keep``), as in d_train.py.

Each row is rendered and tokenised exactly as d_train.py does (prompt and
answer tokenised separately, then concatenated), so position i here is
position i there; d_train asserts the counts match.
"""
import json, os, sys, time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import file_facts, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER = os.environ.get("TEACHER_HF", "Qwen/Qwen3-4B-Instruct-2507")
DATA = os.environ["DATA"]
OUT = os.environ["OUT"]
K = int(os.environ.get("TOPK", "32"))
MAX_LEN = int(os.environ.get("MAX_LEN", "14336"))
SYSTEM = open(os.path.join(ROOT, "data", "system_prompt_rag.txt")).read()


def render(user, answer, system=None):
    prompt = (f"<|im_start|>system\n\n{system or SYSTEM}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return prompt, answer + "<|im_end|>"


def target(row):
    return (row.get("target") or row["teacher_answer"]).strip()


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
assert AutoTokenizer.from_pretrained(TEACHER).get_vocab() == tok.get_vocab(), "teacher and student vocabularies differ"
model = AutoModelForCausalLM.from_pretrained(
    TEACHER, attn_implementation="sdpa", dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16),
    device_map={"": 0})
model.eval()

rows = [json.loads(line) for line in open(DATA)]
rows = [r for r in rows if (r["keep"] if "keep" in r else r["teacher_pass"])]
store = torch.load(OUT) if os.path.exists(OUT) else {}
t0, skipped, tokens = time.time(), 0, 0
torch.cuda.reset_peak_memory_stats()
for n, r in enumerate(rows):
    if r["id"] in store:
        continue
    p, a = render(r["user_message"], target(r), r.get("system"))
    pi, ai = tok(p, add_special_tokens=False)["input_ids"], tok(a, add_special_tokens=False)["input_ids"]
    if len(pi) + len(ai) > MAX_LEN:
        skipped += 1
        continue
    ids = torch.tensor([pi + ai], device="cuda")
    keep = torch.zeros_like(ids, dtype=torch.bool)
    keep[0, len(pi) - 1:len(pi) + len(ai) - 1] = True
    with torch.no_grad():
        # A boolean logits_to_keep returns (kept positions, vocab), already flattened:
        # indexing [0] here once kept only the first answer position.
        logits = model(input_ids=ids, logits_to_keep=keep, use_cache=False).logits
        logits = logits.reshape(-1, logits.shape[-1]).float()
    assert logits.shape[0] == len(ai), f"{r['id']}: {logits.shape[0]} teacher positions for {len(ai)} answer tokens"
    logp, top = F.log_softmax(logits, -1).topk(K, -1)
    store[r["id"]] = (top.to(torch.int32).cpu(), logp.to(torch.float16).cpu())
    tokens += len(ai)
    if n % 100 == 0:
        torch.save(store, OUT)
        print(f"[kd] {n}/{len(rows)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
torch.save(store, OUT)
wall = time.time() - t0
print(f"done {len(store)} rows, {tokens} answer tokens, {skipped} over {MAX_LEN}, {wall:.0f}s", flush=True)
record(f"teacher_logits:{os.path.basename(OUT)}", teacher=TEACHER, quant="bnb nf4, bf16 compute", topk=K,
       data=file_facts(DATA), rows=len(store), answer_tokens=tokens, skipped_over_max_len=skipped,
       wall_s=round(wall), vram_peak_bytes=torch.cuda.max_memory_allocated())
