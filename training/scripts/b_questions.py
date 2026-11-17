"""Stage B: the teacher writes two extractive questions per sampled chunk.

Answers must be a verbatim span of the chunk, so stage C can check the
teacher's own RAG answer against them with the gate's grader.
"""
import json, os, random, re, urllib.request
from collections import defaultdict
from common import ROOT, TEACHER

random.seed(626)
PER_DOC = int(os.environ.get("PER_DOC", "30"))
LANG_NAME = {"en": "English", "es": "Spanish (castellano)", "ca": "Catalan (català)"}
SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "minItems": 2, "maxItems": 2,
          "items": {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                    "required": ["question", "answer"]}}}, "required": ["questions"]}
PROMPT = """You write questions for a reading-comprehension exam over one passage of the document "{doc}".

Write exactly two questions in {qlang}:
1. One whose answer is a number, quantity, percentage, date or year stated in the passage (if the passage has none, ask for a specific name or term instead).
2. One whose answer is a specific name, term or short phrase of at most six words from the passage.

Rules:
- Each answer must be copied verbatim from the passage, character for character, and be as short as possible (for a number include only the number and its unit).
- Each question must be self-contained: name the subject explicitly (the method, model, person, place or topic). Never say "the passage", "the text", "the document" or "the article".
- The question must have a single unambiguous answer in the passage.

Passage:
\"\"\"
{text}
\"\"\""""

def chat(prompt):
    body = {"model": TEACHER, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "think": False, "format": SCHEMA, "options": {"temperature": 0.7, "num_ctx": 4096, "num_predict": 300}, "keep_alive": 300}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"])

norm = lambda s: re.sub(r"\s+", " ", s).strip()
chunks = [json.loads(l) for l in open(os.path.join(ROOT, "data", "chunks.jsonl"))]
by_doc = defaultdict(list)
for c in chunks:
    meta = c["metadata"]
    c["text"], c["source"], c["page"] = c["doc"], meta["source"], meta.get("page")
    if len(c["text"]) >= 500 and meta.get("format", "markdown") in ("markdown", "table"):
        by_doc[(c["lang"], c["source"])].append(c)
out_path = os.path.join(ROOT, "data", "questions.jsonl")
done = {(json.loads(l)["source"], json.loads(l)["chunk_idx"]) for l in open(out_path)} if os.path.exists(out_path) else set()
out = open(out_path, "a")
kept = dropped = 0
# NEW_DOCS_ONLY: a corpus extension asks only about the documents it added,
# instead of drawing fresh chunks from the ones already questioned.
seen_docs = {s for s, _ in done}
for (lang, source), cs in sorted(by_doc.items()):
    if os.environ.get("NEW_DOCS_ONLY") and source in seen_docs:
        continue
    sample = random.sample(cs, min(PER_DOC, len(cs)))
    for i, c in enumerate(sample):
        key = (source, c["id"])
        if key in done:
            continue
        qlang = lang if lang != "en" else random.choices(["en", "es", "ca"], [0.6, 0.2, 0.2])[0]
        try:
            qs = chat(PROMPT.format(doc=source.rsplit(".", 1)[0].replace("_", " "), qlang=LANG_NAME[qlang], text=c["text"]))["questions"]
        except Exception as exc:
            print("ERR", source, exc, flush=True); continue
        for q in qs:
            ans, ques = q["answer"].strip(), q["question"].strip()
            ok = ans and len(ans) <= 60 and norm(ans) in norm(c["text"]) and ques.endswith("?")
            # A translated question cannot expect a verbatim English term back.
            if ok and qlang != lang and not re.search(r"\d", ans):
                ok = False
            kept += ok; dropped += not ok
            if ok:
                out.write(json.dumps({"lang": qlang, "doc_lang": lang, "source": source, "chunk_idx": key[1], "page": c.get("page"),
                                      "question": ques, "answer": ans, "chunk": c["text"]}, ensure_ascii=False) + "\n")
        out.flush()
    print(f"{lang} {source}: kept {kept} dropped {dropped}", flush=True)
print("done", kept, dropped)
