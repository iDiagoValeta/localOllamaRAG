"""Stage H: distill study-mode artifacts (summary, outline, quiz) from the teacher.

The first SFT (sft-r32) only saw short factual answers and forgot how to fill
the study schemas: 1 section, 1 question, 5 titles. This stage replays them.

Every artifact is built by the product's own ``Study`` use case over a whole
held-out document, exactly as the gate does, with the teacher as generator. A
thin wrapper records the (system, user, raw reply) of that one call, so the
student is trained on the bytes the product sends. The artifact is kept only
when the gate's ``grade_study`` accepts it, with the quiz held to the full
question count it was asked for.

Phase 1 opens each language's store (embedder up), phase 2 kills the embedder
and runs the teacher alone on the GPU.
"""
import json, os, random, signal, time
from common import LANGS, ROOT, TEACHER, docs_dir
import run_eval, grade
import rag.chat_pdfs as rag
from rag.engine import wiring
from monkeygrab.application.study import Study, outline_to_dict, quiz_to_dict, summary_to_dict
from record import file_facts, record

AUX = run_eval.AUX_MODEL
rag.set_model_roles_runtime({"rag": TEACHER, "chat": AUX, "contextual": AUX, "recomp": AUX})
LABEL = {"en": "English", "es": "Castellano", "ca": "Valencià"}
OUT = os.path.join(ROOT, "data", "study.jsonl")

docs = {}
for lang in LANGS:
    folder = docs_dir(lang)
    pdfs = sorted(p.name for p in folder.glob("*.pdf"))
    retrieve, _answer, stack = run_eval.ensure_indexed(rag, folder, pdfs, f"heldout-{lang}", path_label=folder.parent / f"heldout_{lang}")
    for pdf in pdfs:
        docs[(lang, pdf)] = run_eval.document_chunks(stack.vector_store, pdf)
    run_eval._release_gpu_models(retrieve)
    if hasattr(stack.embedder, "close"):
        stack.embedder.close()
pid_file = os.path.join(ROOT, "runs", "embedder.pid")
if os.path.exists(pid_file):
    try:
        os.kill(int(open(pid_file).read()), signal.SIGTERM)
    except ProcessLookupError:
        pass
print(f"[load] {len(docs)} documents, {sum(map(len, docs.values()))} chunks", flush=True)


class Recorder:
    """The generator role, remembering the one call Study makes through it."""

    def __init__(self, inner):
        self.inner, self.call = inner, None

    def generate(self, prompt, *, system=None, response_format=None, **kw):
        raw = self.inner.generate(prompt, system=system, response_format=response_format, **kw)
        self.call = {"system": system, "user_message": prompt, "raw": raw}
        return raw

    def __getattr__(self, name):
        return getattr(self.inner, name)


done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
rng = random.Random(626)
config = wiring.app_config_from_runtime()
counts = {"kept": 0, "total": 0}
t0 = time.time()
with open(OUT, "a") as out:
    for (doc_lang, pdf), chunks in docs.items():
        frags = [wiring.fragment_from_dict(c) for c in chunks]
        for kind in ("study_summary", "study_outline", "study_quiz"):
            for lang in LANGS:
                n_q = rng.choice((3, 4, 5))
                rid = f"{kind}:{doc_lang}:{pdf}:{lang}"
                if rid in done or not frags:
                    continue
                rec = Recorder(wiring.rag_chat_model(config))
                study = Study(rec)
                try:
                    if kind == "study_summary":
                        artifact = summary_to_dict(study.summarize(frags, config, language=LABEL[lang]))
                    elif kind == "study_outline":
                        artifact = outline_to_dict(study.outline(frags, config, language=LABEL[lang]))
                    else:
                        artifact = quiz_to_dict(study.quiz(frags, config, language=LABEL[lang], question_count=n_q))
                    verdict = grade.grade_study(artifact, {"case_type": kind, "min_sections": 3, "min_nodes": 6,
                                                           "min_questions": n_q})
                except Exception as exc:  # a malformed teacher reply is a dropped example, not a crash
                    verdict = {"pass": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
                call = rec.call or {}
                counts["total"] += 1
                counts["kept"] += verdict["pass"]
                out.write(json.dumps({"id": rid, "kind": kind, "doc": pdf, "doc_lang": doc_lang, "lang": lang,
                                      "question_count": n_q if kind == "study_quiz" else None,
                                      "system": call.get("system"), "user_message": call.get("user_message"),
                                      "teacher_answer": (call.get("raw") or "").strip(),
                                      "teacher_pass": verdict["pass"], "reason": verdict["reason"]},
                                     ensure_ascii=False) + "\n")
                out.flush()
                print(f"[teacher] {counts['total']} {rid} {'OK' if verdict['pass'] else 'DROP'} "
                      f"{verdict['reason']} ({time.time() - t0:.0f}s)", flush=True)
record("study_distill", teacher=TEACHER, outcome=counts, wall_s=round(time.time() - t0),
       study=file_facts(OUT))
print("done", counts)
