"""Stage M: synthetic questions and answers written by Claude models.

    python m_synth.py split [--per-doc 12] [--size 25]   # alignment/synth_batches/train_NN.jsonl + dev_NN.jsonl
    python m_synth.py merge                              # verified items -> data/synth_questions.jsonl

Writers are Claude Code subagents: Sonnet writes training items over chunks of
training documents, Opus writes a harder test set over the dev documents, and
Haiku verifies every Sonnet item (alignment/synth_verdicts/). Chunks with
tables and numbers are drawn first, because picking the wrong number from a
table is the student's most common failure on the gate.

An item survives only if its answer span is verbatim in the chunk (checked
here, not trusted), its question does not echo the span, and, for training
items, Haiku kept it. Output rows have the shape of data/questions_clean.jsonl
plus ``ideal`` (the writer's model answer) and ``writer``, so the existing
retrieval stage can build the product's user message for them.
"""
import collections, glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEV_DOCS, ROOT
from record import file_facts, record

BATCHES = os.path.join(ROOT, "alignment", "synth_batches")
OUTS = os.path.join(ROOT, "alignment", "synth_items")
VERDICTS = os.path.join(ROOT, "alignment", "synth_verdicts")
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
DIGIT = re.compile(r"\d")


def chunks():
    rows = [json.loads(line) for line in open(os.path.join(ROOT, "data", "chunks.jsonl"))]
    # Leonardo_da_Vinci, Ruta_de_la_Seda and Vacuna exist in both the es and ca
    # corpus under one file name and the same chunk ids; their chunk_id carries
    # the language so the two never merge. Every other id keeps its old form, so
    # items written before the corpus extension still resolve.
    langs = collections.defaultdict(set)
    for c in rows:
        langs[c["metadata"]["source"]].add(c["lang"])
    out = []
    for c in rows:
        meta = c["metadata"]
        if meta.get("format") in ("markdown", "table") and len(c["doc"]) >= 400:
            prefix = f"{c['lang']}:" if len(langs[meta["source"]]) > 1 else ""
            out.append({"chunk_id": f"{prefix}{meta['source']}#{c['id']}", "doc": meta["source"], "doc_lang": c["lang"],
                        "format": meta.get("format"), "page": meta.get("page"), "text": c["doc"][:3000]})
    return out


def split(per_doc, size, only_new=False, prefix=""):
    os.makedirs(BATCHES, exist_ok=True)
    rng = random.Random(626)
    by_doc = collections.defaultdict(list)
    for c in chunks():
        by_doc[(c["doc_lang"], c["doc"])].append(c)
    # --only-new: a corpus extension writes batches for the documents no earlier
    # batch drew from, under its own file prefix, leaving the old batches alone.
    seen = {(c["doc_lang"], c["doc"]) for c in read_jsonl(os.path.join(BATCHES, "*.jsonl"))} if only_new else set()
    dev = set(DEV_DOCS.values())
    train, test = [], []
    for (_lang, doc), cs in sorted(by_doc.items()):
        if (_lang, doc) in seen:
            continue
        # Tables first, then passages with numbers, then the rest.
        rng.shuffle(cs)
        cs.sort(key=lambda c: (c["format"] != "table", not DIGIT.search(c["text"])))
        (test if doc in dev else train).extend(cs[:per_doc * (2 if doc in dev else 1)])
    for name, items in (("train", train), ("dev", test)):
        for k in range(0, len(items), size):
            with open(os.path.join(BATCHES, f"{prefix}{name}_{k // size:02d}.jsonl"), "w") as f:
                for c in items[k:k + size]:
                    # Questions on English documents are sometimes asked in
                    # Spanish or Catalan, as product users do.
                    qlang = c["doc_lang"] if c["doc_lang"] != "en" else rng.choices(["en", "es", "ca"], [0.6, 0.2, 0.2])[0]
                    f.write(json.dumps({**c, "question_lang": qlang}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} chunks in {(len(items) + size - 1) // size} batches")


def read_jsonl(pattern):
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def merge():
    src = {c["chunk_id"]: c for c in chunks()}
    verdicts = {v["item_id"]: v for v in read_jsonl(os.path.join(VERDICTS, "*.jsonl"))}
    why = collections.Counter()
    out = open(os.path.join(ROOT, "data", "synth_questions.jsonl"), "w")
    for it in read_jsonl(os.path.join(OUTS, "*.jsonl")):
        c = src.get(it.get("chunk_id"))
        span, q = (it.get("answer") or "").strip(), (it.get("question") or "").strip()
        split_name = "dev" if c and c["doc"] in DEV_DOCS.values() else "train"
        if c is None:
            why["unknown_chunk"] += 1; continue
        if not span or norm(span) not in norm(c["text"]):
            why["span_not_verbatim"] += 1; continue
        if norm(span) in norm(q):
            why["span_echoed"] += 1; continue
        if split_name == "train":
            v = verdicts.get(it["item_id"])
            if v is None:
                why["not_verified"] += 1; continue
            if v.get("verdict") != "keep":
                why["haiku_drop"] += 1; continue
        why[f"kept_{split_name}"] += 1
        out.write(json.dumps({"lang": it["question_lang"], "doc_lang": c["doc_lang"], "source": c["doc"],
                              "chunk_idx": c["chunk_id"].split("#", 1)[1], "page": c["page"], "question": q,
                              "answer": span, "chunk": c["text"], "ideal": (it.get("ideal") or "").strip(),
                              "kind": it.get("kind"), "writer": it.get("writer"), "split": split_name,
                              "item_id": it["item_id"]}, ensure_ascii=False) + "\n")
    out.close()
    print(json.dumps(why, indent=1))
    record("synth_merge", outcome=dict(why), out=file_facts(os.path.join(ROOT, "data", "synth_questions.jsonl")))


if __name__ == "__main__":
    if sys.argv[1] == "split":
        a = sys.argv
        split(int(a[a.index("--per-doc") + 1]) if "--per-doc" in a else 12, int(a[a.index("--size") + 1]) if "--size" in a else 25,
              only_new="--only-new" in a, prefix=a[a.index("--prefix") + 1] if "--prefix" in a else "")
    else:
        merge()
