"""Stage I: fast dev evaluation on documents no experiment trains on.

    python i_deveval.py <ollama model> [<ollama model> ...]

The gate is the final exam: choosing hyperparameters on it would overfit it.
This scores a model on DEV_DOCS (one held-out document per language, excluded
from every training set by ``dev_docs()``) with the product's own code paths:

- factual: every clean question of those documents, the exact user message
  the product built for it (stored in data/messages.jsonl), the product's
  generator, the gate's ``grade_answer`` against the verbatim reference. The
  teacher's own failures count too: the reference is the truth, not the teacher.
- study: summary, outline and quiz over each whole document in all three
  languages, graded by the gate's ``grade_study`` with the gate's thresholds.

Results append to runs/deveval.jsonl, one line per (model, item), and a
summary line per model goes to the manifest.
"""
import json, os, sys, time
from common import DEV_DOCS, ROOT
import grade
import rag.chat_pdfs as rag
from rag.engine import wiring
from monkeygrab.application.study import Study, outline_to_dict, quiz_to_dict, summary_to_dict
from record import record

LABEL = {"en": "English", "es": "Castellano", "ca": "Valencià"}
OUT = os.path.join(ROOT, "runs", "deveval.jsonl")


def dev_docs():
    return set(DEV_DOCS.values())


def factual_items():
    return [m for m in map(json.loads, open(os.path.join(ROOT, "data", "messages.jsonl")))
            if m["source"] in dev_docs()]


def hard_items():
    """Opus-written test questions on the dev documents whose value reached the context."""
    path = os.path.join(ROOT, "data", "synth_dev.jsonl")
    if not os.path.exists(path):
        return []
    return [r for r in map(json.loads, open(path)) if r["value_in_context"]]


def study_items():
    chunks = {}
    for line in open(os.path.join(ROOT, "data", "chunks.jsonl")):
        c = json.loads(line)
        if c["metadata"]["source"] in dev_docs():
            chunks.setdefault(c["metadata"]["source"], []).append(c)
    return [(doc, kind, lang, chunks[doc]) for doc in sorted(chunks)
            for kind in ("study_summary", "study_outline", "study_quiz") for lang in LABEL]


def evaluate(model):
    rag.set_model_roles_runtime({"rag": model})
    config = wiring.app_config_from_runtime()
    res = {"factual": [0, 0], "hard": [0, 0], "study": [0, 0]}
    t0 = time.time()
    with open(OUT, "a") as out:
        for kind, m in [("factual", m) for m in factual_items()] + [("hard", m) for m in hard_items()]:
            t = time.time()
            answer = "".join(rag.generar_tokens_respuesta(m["user_message"]))
            case = {"case_type": "factual_number" if any(ch.isdigit() for ch in m["answer"]) else "factual_concept",
                    "accepted_answers": [m["answer"]]}
            ok = grade.grade_answer(answer, case)["pass"]
            res[kind][0] += ok; res[kind][1] += 1
            out.write(json.dumps({"model": model, "kind": kind, "lang": m["lang"], "doc": m["source"],
                                  "question": m["question"], "pass": ok, "answer": answer,
                                  "s": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
        for doc, kind, lang, chunks in study_items():
            t = time.time()
            frags = [wiring.fragment_from_dict(c) for c in chunks]
            study = Study(wiring.rag_chat_model(config))
            try:
                if kind == "study_summary":
                    art = summary_to_dict(study.summarize(frags, config, language=LABEL[lang]))
                elif kind == "study_outline":
                    art = outline_to_dict(study.outline(frags, config, language=LABEL[lang]))
                else:
                    art = quiz_to_dict(study.quiz(frags, config, language=LABEL[lang], question_count=3))
                v = grade.grade_study(art, {"case_type": kind, "min_sections": 3, "min_nodes": 6, "min_questions": 2})
            except Exception as exc:
                v = {"pass": False, "reason": f"{type(exc).__name__}"}
            res["study"][0] += v["pass"]; res["study"][1] += 1
            out.write(json.dumps({"model": model, "kind": kind, "lang": lang, "doc": doc, "pass": v["pass"],
                                  "reason": v["reason"], "s": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
    wall = round(time.time() - t0)
    print(f"{model}: factual {res['factual'][0]}/{res['factual'][1]}  hard {res['hard'][0]}/{res['hard'][1]}  study {res['study'][0]}/{res['study'][1]}  ({wall}s)",
          flush=True)
    record(f"deveval:{model}", factual=res["factual"], hard=res["hard"], study=res["study"], wall_s=wall)
    import urllib.request
    urllib.request.urlopen(urllib.request.Request(f"{rag.OLLAMA_BASE_URL}/api/generate",
                           data=json.dumps({"model": model, "keep_alive": 0}).encode()))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        evaluate(name)
