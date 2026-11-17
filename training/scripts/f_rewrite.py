"""Stage F: rewrite the fixable answers and build the SFT set.

    python f_rewrite.py split [--size 50]   # alignment/rewrite_batches/rewrite_NN.jsonl from data/judged.jsonl
    python f_rewrite.py merge               # alignment/rewrites/*.jsonl -> data/sft.jsonl (keep + target)

The rewriters are Claude Code subagents (Sonnet) following the rubric's rewrite
instructions. A rewrite that drops the reference value is not trusted: it is
checked with the gate's own grader, and a rewrite that fails it, or a REJECT,
discards the item instead of training on it.
"""
import collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "eval"))
from record import file_facts, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = os.path.join(ROOT, "alignment", "rewrite_batches")
REWRITES = os.path.join(ROOT, "alignment", "rewrites")
JUDGED = os.path.join(ROOT, "data", "judged.jsonl")
SFT = os.path.join(ROOT, "data", "sft.jsonl")


def judged():
    return [json.loads(line) for line in open(JUDGED)]


def split(size):
    os.makedirs(BATCHES, exist_ok=True)
    items = [{"id": r["id"], "lang": r["lang"], "question": r["question"], "reference": r["answer"],
              "passage": r["chunk"], "candidate": r["teacher_answer"].strip(), "failed": r["failed"]}
             for r in judged() if r.get("verdict") == "fixable"]
    for k in range(0, len(items), size):
        with open(os.path.join(BATCHES, f"rewrite_{k // size:02d}.jsonl"), "w") as f:
            for it in items[k:k + size]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} fixable items in {(len(items) + size - 1) // size} batches of {size}")


def merge():
    from grade import grade_answer
    rewrites = {}
    for path in sorted(glob.glob(os.path.join(REWRITES, "*.jsonl"))):
        for line in open(path):
            if line.strip().startswith("{"):
                v = json.loads(line)
                rewrites[v["id"]] = v["answer"].strip()
    counts = collections.Counter()
    with open(SFT, "w") as f:
        for r in judged():
            verdict, target = r.get("verdict"), None
            if verdict == "good":
                target, why = r["teacher_answer"].strip(), "good"
            elif verdict == "fixable":
                text = rewrites.get(r["id"])
                if text is None:
                    why = "fixable:missing"
                elif text == "REJECT":
                    why = "fixable:reject"
                elif not grade_answer(text, {"accepted_answers": [r["answer"]]})["pass"]:
                    why = "fixable:lost_value"
                else:
                    target, why = text, "fixable:rewritten"
            else:
                why = verdict or "not_judged"
            r["keep"], r["target"], r["why"] = target is not None, target, why
            counts[why] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))
    record("rewrite", outcome=dict(counts), kept=sum(n for k, n in counts.items() if k in ("good", "fixable:rewritten")),
           sft=file_facts(SFT))


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 50)
    else:
        merge()
