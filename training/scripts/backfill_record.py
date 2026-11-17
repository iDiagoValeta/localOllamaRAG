"""Record the stages that ran before the recorder existed (corpus, index, questions)."""
import json, os, re, sys, collections, urllib.parse
from record import ROOT, record, environment, file_facts
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
from tools.fetch_corpus import ARXIV_PDF_URL, WIKI_PDF_URL, _current_revid, _wiki_filename

record("environment", **environment())
spec = json.load(open(os.path.join(ROOT, "scripts", "heldout.json")))
gold = {json.loads(l)["paper"] for l in open(os.path.join(REPO, "tests", "eval", "gold_cases.jsonl"))}
docs = []
for lang, s in spec.items():
    for aid, fn in s.get("arxiv", []):
        docs.append({"lang": lang, "kind": "arxiv", "id": aid, "file": fn, "url": ARXIV_PDF_URL.format(id=aid),
                     "license": "per-paper arXiv license, check before redistributing"})
    for t in s.get("wikipedia", []):
        fn = _wiki_filename(t)
        docs.append({"lang": lang, "kind": "wikipedia", "title": t, "file": fn, "revid_at_record": _current_revid(lang, t),
                     "url": WIKI_PDF_URL.format(lang=lang, title=urllib.parse.quote(t.replace(" ", "_"), safe="")),
                     "license": "CC BY-SA 4.0"})
for d in docs:
    p = os.path.join(ROOT, "docs", d["lang"], d["file"])
    d["bytes"] = os.path.getsize(p)
record("corpus", documents=docs, n_documents=len(docs),
       gold_overlap=sorted({d["file"][:-4] for d in docs} & gold),
       selection="Topics outside the 47 gold-set documents so the gate stays a held-out test. A first pick included bert, resnet and vit, which are gold papers; caught before any data was generated and replaced by word2vec (1301.3781), bahdanau-attention (1409.0473) and chinchilla (2203.15556).",
       fetcher="tools/fetch_corpus.py helpers (arXiv /pdf, Wikipedia REST page/pdf)")

log = open(os.path.join(ROOT, "runs", "a_index.log")).read()
chunks = {m[0]: int(m[1]) for m in re.findall(r"heldout-(\w+): stack=\S+ chunks=(\d+)", log)}
record("index", stack="mineru-eg2-faiss (MinerU 3.4.5 pipeline backend, embeddinggemma-2 Q8_0 512-dim via llama-server b11669 on CPU, FAISS)",
       path="run_eval.ensure_indexed, the gate's own indexing path, contextual retrieval off as in the gate",
       chunks_per_lang=chunks, chunks_dump=file_facts(os.path.join(ROOT, "data", "chunks.jsonl")),
       note="Embedder on CPU so MinerU kept the GPU; about 70 min for 26 documents.")

qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "questions.jsonl"))]
blog = open(os.path.join(ROOT, "runs", "b_questions.log")).read()
last = re.findall(r"kept (\d+) dropped (\d+)", blog)[-1]
record("questions", teacher="qwen3:4b-instruct-2507-q4_K_M (Ollama, structured JSON output, think=false)",
       sampling={"temperature": 0.7, "num_ctx": 4096, "num_predict": 300}, chunks_per_doc=30, questions_per_chunk=2,
       chunk_filter=">= 500 characters, markdown or table chunks",
       acceptance="answer is a verbatim span of the chunk (whitespace-normalised), <= 60 chars, question ends with '?'; translated questions (en doc, es/ca question) only when the answer has a digit",
       question_language_mix_for_en_docs={"en": 0.6, "es": 0.2, "ca": 0.2},
       kept=int(last[0]), dropped_by_filter=int(last[1]), teacher_json_errors=blog.count("ERR "),
       by_question_lang=dict(collections.Counter(q["lang"] for q in qs)),
       by_doc_lang=dict(collections.Counter(q["doc_lang"] for q in qs)),
       numeric_answers=sum(bool(re.search(r"\d", q["answer"])) for q in qs),
       output=file_facts(os.path.join(ROOT, "data", "questions.jsonl")))
record("agents", rubric={"model": "opus", "task": "alignment rubric (alignment/rubric.md)", "tokens": 107176, "tool_uses": 14, "seconds": 138,
                         "findings": "4.4% degenerate references (answer echoed in question), ~42 bibliography questions, 20 references behind a non-breaking space, ~209 chunks with digit-spaced numbers"})
print(json.dumps(json.load(open(os.path.join(ROOT, "runs", "manifest.json")))["questions"], indent=1)[:900])
