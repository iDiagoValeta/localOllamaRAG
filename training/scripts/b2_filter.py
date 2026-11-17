"""Drop degenerate references before paying for retrieval, teacher and judges (Opus rubric finding)."""
import json, os, re, collections
from record import ROOT, record, file_facts
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
BIB = re.compile(r"\b(ISBN|doi|pp\.|vol\.|Retrieved|Consultado|Consultat|Archived|Arxivat|Archivado)\b", re.I)
src = os.path.join(ROOT, "data", "questions.jsonl"); dst = os.path.join(ROOT, "data", "questions_clean.jsonl")
kept, why = [], collections.Counter()
for l in open(src):
    q = json.loads(l)
    if norm(q["answer"]) in norm(q["question"]):
        why["answer_echoed_in_question"] += 1; continue
    if len(BIB.findall(q["chunk"])) >= 3:
        why["bibliography_chunk"] += 1; continue
    kept.append(q)
with open(dst, "w") as f:
    for q in kept:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
record("questions_filter", rules={"answer_echoed_in_question": "normalised reference is a substring of the question",
                                  "bibliography_chunk": ">= 3 citation markers (ISBN, doi, pp., vol., Retrieved, Consultado, Consultat, Archived...) in the chunk"},
       dropped=dict(why), kept=len(kept), by_question_lang=dict(collections.Counter(q["lang"] for q in kept)), output=file_facts(dst))
print(len(kept), dict(why))
