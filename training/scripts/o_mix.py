"""Build a training mix from every source, one row per id.

    python o_mix.py data/mix_c.jsonl

Factual rows: sft.jsonl (the 863 Claude-aligned teacher answers) plus every
distilled row it does not cover, with the teacher's own answer as target when
the gate grader passes it. A distilled row whose context came back empty is
dropped even when it passes: the answer cannot come from the context, so the
student would learn to answer from memory (EXPERIMENTS.md finding 12).
Then study.jsonl and synth_sft.jsonl as they are. d_train.py applies
keep/teacher_pass and holds out the dev documents.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def load(name):
    return [json.loads(l) for l in open(os.path.join(D, name))]


sft = load("sft.jsonl")
seen = {(r["source"], r["question"]) for r in sft}
new, empty = [], 0
for i, r in enumerate(load("distilled.jsonl")):
    if (r["source"], r["question"]) in seen or not r["teacher_pass"]:
        continue
    if "<context></context>" in r["user_message"]:
        empty += 1
        continue
    seen.add((r["source"], r["question"]))
    new.append({**r, "id": f"x{i:04d}"})
study, synth = load("study.jsonl"), load("synth_sft.jsonl")
rows = sft + new + study + synth
ids = [r["id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate ids"
with open(sys.argv[1], "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"sft {len(sft)} new distilled {len(new)} (empty context dropped {empty}) "
      f"study {len(study)} synth {len(synth)} total {len(rows)}")
