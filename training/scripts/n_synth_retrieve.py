"""Stage N: put the Claude-written questions through the product's retrieval.

    python n_synth_retrieve.py      # data/synth_questions.jsonl -> data/synth_sft.jsonl + data/synth_dev.jsonl

Same phase as c_distill.py's first half: the product's hybrid retrieval and
evidence selection build the exact user message the generator would receive.

- train items become SFT rows whose target is the writer's ``ideal`` answer,
  kept only when the reference value actually reached the context (otherwise
  the target would teach the student to state a value it cannot see) and the
  ideal answer passes the gate's grader against the reference;
- dev items become the hard dev set i_deveval.py scores as ``factual_hard``.
"""
import json, os, re, time
from collections import Counter, defaultdict
from common import ROOT, docs_dir
import run_eval, grade
import rag.chat_pdfs as rag
from record import file_facts, record

AUX = run_eval.AUX_MODEL
rag.set_model_roles_runtime({"chat": AUX, "contextual": AUX, "recomp": AUX})
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "synth_questions.jsonl"))]
SFT_PATH, DEV_PATH = (os.path.join(ROOT, "data", f) for f in ("synth_sft.jsonl", "synth_dev.jsonl"))
# Incremental: items already written keep their row (re-retrieving them would
# cost 45 min and change rows under a trained model, since query decomposition
# is not seeded). Items dropped earlier are retried.
done = {json.loads(l)["id"] for p in (SFT_PATH, DEV_PATH) if os.path.exists(p) for l in open(p)}
by_lang = defaultdict(list)
for it in items:
    if f"s{it['item_id']}" not in done:
        by_lang[it["doc_lang"]].append(it)
print("to retrieve", {k: len(v) for k, v in by_lang.items()}, "already done", len(done), flush=True)
t0, why = time.time(), Counter()
sft = open(SFT_PATH, "a")
dev = open(DEV_PATH, "a")
for lang, its in by_lang.items():
    folder = docs_dir(lang)
    pdfs = sorted(p.name for p in folder.glob("*.pdf"))
    retrieve, evidence, stack = run_eval.ensure_indexed(rag, folder, pdfs, f"heldout-{lang}", path_label=folder.parent / f"heldout_{lang}")
    for k, it in enumerate(its):
        result = retrieve.run(it["question"])
        selected, _ = evidence.select_evidence(result.fragments)
        frags = [run_eval._fragment_to_dict(f) for f in selected]
        msg = rag._preparar_mensaje_usuario_rag(it["question"], frags)
        reached = norm(it["answer"]) in norm(msg)
        case = {"case_type": "factual_number" if any(ch.isdigit() for ch in it["answer"]) else "factual_concept",
                "accepted_answers": [it["answer"]]}
        row = {**it, "id": f"s{it['item_id']}", "user_message": msg, "value_in_context": reached,
               "source_retrieved": any(f["metadata"]["source"] == it["source"] for f in frags)}
        if it["split"] == "dev":
            dev.write(json.dumps(row, ensure_ascii=False) + "\n"); why["dev"] += 1
        elif not reached:
            why["train_value_not_retrieved"] += 1
        elif not grade.grade_answer(it["ideal"], case)["pass"]:
            why["train_ideal_fails_grader"] += 1
        else:
            sft.write(json.dumps({**row, "keep": True, "target": it["ideal"], "why": f"synth:{it['writer']}"},
                                 ensure_ascii=False) + "\n")
            why["train_kept"] += 1
        if k % 50 == 0:
            print(f"[retrieve] {lang} {k}/{len(its)} {dict(why)}", flush=True)
    run_eval._release_gpu_models(retrieve)
    if hasattr(stack.embedder, "close"):
        stack.embedder.close()
sft.close(); dev.close()
print("done", dict(why), round(time.time() - t0), "s")
record("synth_retrieve", outcome=dict(why), wall_s=round(time.time() - t0),
       sft=file_facts(os.path.join(ROOT, "data", "synth_sft.jsonl")),
       dev=file_facts(os.path.join(ROOT, "data", "synth_dev.jsonl")))
