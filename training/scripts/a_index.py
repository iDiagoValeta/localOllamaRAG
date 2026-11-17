"""Stage A: index the held-out docs with the gate's own path; dump their chunks."""
import json, os
from common import ROOT, LANGS, docs_dir
import run_eval
import rag.chat_pdfs as rag
rag.set_model_roles_runtime({r: run_eval.AUX_MODEL for r in ("chat", "contextual", "recomp")})
out = open(os.path.join(ROOT, "data", "chunks.jsonl"), "w")
for lang in LANGS:
    folder = docs_dir(lang)
    pdfs = sorted(p.name for p in folder.glob("*.pdf"))
    retrieve, evidence, stack = run_eval.ensure_indexed(rag, folder, pdfs, f"heldout-{lang}", path_label=folder.parent / f"heldout_{lang}")
    for pdf in pdfs:
        for frag in stack.vector_store.get_page(None, 0):
            if frag.metadata.source != pdf:
                continue
            ch = run_eval._fragment_to_dict(frag)
            out.write(json.dumps({"lang": lang, **ch}, ensure_ascii=False) + "\n")
    run_eval._release_gpu_models(retrieve)
    if hasattr(stack.embedder, "close"):
        stack.embedder.close()
out.close()
print("done")
