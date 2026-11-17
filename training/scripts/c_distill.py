"""Stage C: run each question through the product's retrieval, then the teacher.

Phase 1 retrieves every question (reranker resident, chat role on AUX_MODEL
for query decomposition, as the gate does); phase 2 releases the GPU and has
the teacher answer the exact user message the product builds. A pair is kept
only when the gate's grader accepts the teacher's answer against the span the
question was written from.
"""
import json, os
from collections import defaultdict
from common import ROOT, TEACHER, docs_dir
import run_eval, grade
import rag.chat_pdfs as rag

AUX = run_eval.AUX_MODEL
rag.set_model_roles_runtime({"rag": TEACHER, "chat": AUX, "contextual": AUX, "recomp": AUX})
qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "questions_clean.jsonl"))]
msg_path = os.path.join(ROOT, "data", "messages.jsonl")
have = {json.loads(l)["question"] for l in open(msg_path)} if os.path.exists(msg_path) else set()
by_lang = defaultdict(list)
for q in qs:
    if q["question"] not in have:
        by_lang[q["doc_lang"]].append(q)
with open(msg_path, "a") as out:
    for lang, items in by_lang.items():
        folder = docs_dir(lang)
        pdfs = sorted(p.name for p in folder.glob("*.pdf"))
        retrieve, evidence, stack = run_eval.ensure_indexed(rag, folder, pdfs, f"heldout-{lang}", path_label=folder.parent / f"heldout_{lang}")
        for i, q in enumerate(items):
            result = retrieve.run(q["question"])
            selected, _ = evidence.select_evidence(result.fragments)
            frags = [run_eval._fragment_to_dict(f) for f in selected]
            msg = rag._preparar_mensaje_usuario_rag(q["question"], frags)
            hit = any(f["metadata"]["source"] == q["source"] for f in frags)
            out.write(json.dumps({**q, "user_message": msg, "source_retrieved": hit}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"[retrieve] {lang} {i}/{len(items)}", flush=True)
        run_eval._release_gpu_models(retrieve)
        if hasattr(stack.embedder, "close"):
            stack.embedder.close()

msgs = [json.loads(l) for l in open(msg_path)]
ans_path = os.path.join(ROOT, "data", "distilled.jsonl")
done = {json.loads(l)["question"] for l in open(ans_path)} if os.path.exists(ans_path) else set()
kept = total = 0
with open(ans_path, "a") as out:
    for i, m in enumerate(msgs):
        if m["question"] in done:
            continue
        answer = "".join(rag.generar_tokens_respuesta(m["user_message"]))
        case = {"case_type": "factual_number" if any(ch.isdigit() for ch in m["answer"]) else "factual_concept",
                "accepted_answers": [m["answer"]]}
        verdict = grade.grade_answer(answer, case)
        total += 1; kept += verdict["pass"]
        out.write(json.dumps({**m, "teacher_answer": answer, "teacher_pass": verdict["pass"]}, ensure_ascii=False) + "\n")
        out.flush()
        if i % 50 == 0:
            print(f"[teacher] {i}/{len(msgs)} kept {kept}/{total}", flush=True)
print("done", kept, total)
