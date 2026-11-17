"""Stage E: hand the distilled pairs to the judges and fold their verdicts back.

    python e_judge.py split [--size 40]   # alignment/batches/judge_NN.jsonl, teacher-passing pairs only
    python e_judge.py merge               # alignment/verdicts/*.jsonl -> data/judged.jsonl + counts

The judges are Claude Code subagents (Haiku) applying alignment/rubric.md; each
reads one batch file and writes one verdict file with the same name, so no two
judges ever touch the same file.
"""
import collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import file_facts, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = os.path.join(ROOT, "alignment", "batches")
VERDICTS = os.path.join(ROOT, "alignment", "verdicts")
DISTILLED = os.path.join(ROOT, "data", "distilled.jsonl")
JUDGED = os.path.join(ROOT, "data", "judged.jsonl")


def rows():
    out = []
    for i, line in enumerate(open(DISTILLED)):
        r = json.loads(line)
        r["id"] = f"d{i:04d}"
        out.append(r)
    return out


def split(size):
    os.makedirs(BATCHES, exist_ok=True)
    items = [{"id": r["id"], "lang": r["lang"], "question": r["question"], "reference": r["answer"],
              "passage": r["chunk"], "candidate": r["teacher_answer"].strip()}
             for r in rows() if r["teacher_pass"]]
    for k in range(0, len(items), size):
        with open(os.path.join(BATCHES, f"judge_{k // size:02d}.jsonl"), "w") as f:
            for it in items[k:k + size]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items in {(len(items) + size - 1) // size} batches of {size}")


def merge():
    verdicts = {}
    for path in sorted(glob.glob(os.path.join(VERDICTS, "*.jsonl"))):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                v = json.loads(line)
                verdicts[v["id"]] = v
    all_rows = rows()
    expected = {r["id"] for r in all_rows if r["teacher_pass"]}
    missing = sorted(expected - verdicts.keys())
    counts = collections.Counter()
    with open(JUDGED, "w") as f:
        for r in all_rows:
            v = verdicts.get(r["id"])
            if v:
                r["verdict"], r["failed"], r["reason"] = v["verdict"], v.get("failed", []), v.get("reason", "")
                counts[(r["lang"], r["verdict"])] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {f"{lang}:{verdict}": n for (lang, verdict), n in sorted(counts.items())}
    print(json.dumps(by, indent=1), f"\nmissing {len(missing)}", missing[:10])
    record("judge", verdict_counts=by, missing=len(missing), judged=file_facts(JUDGED))


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 40)
    else:
        merge()
