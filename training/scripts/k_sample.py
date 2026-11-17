"""Stage K: on-policy samples from the student, graded against the reference.

    python k_sample.py <ollama model> [N]     # N samples per question (default 4)

Every clean training question (dev documents excluded, teacher failures
included: the reference is the truth) gets N answers from the student at
temperature 0.8, through Ollama's chat endpoint with the product's RAG system
prompt and the exact user message the product built. Each is graded with the
gate's ``grade_answer``. Output: data/samples_<tag>.jsonl, one row per question
with every sample and its verdict, which l_dpo.py turns into preference pairs.
"""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from common import DEV_DOCS, ROOT
import grade
from record import file_facts, record

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SYSTEM = open(os.path.join(ROOT, "data", "system_prompt_rag.txt")).read()
OUT = os.path.join(ROOT, "data", f"samples_{MODEL.split(':')[-1]}.jsonl")


def chat(user, seed):
    body = {"model": MODEL, "stream": False, "think": False, "keep_alive": 300,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "options": {"temperature": 0.8, "top_p": 0.95, "seed": seed, "num_ctx": 16384, "num_predict": 400}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            # Ollama aborts a sample stuck repeating one token with a 500; it is a
            # failed answer with no text to learn from, so l_dpo skips it.
            if b"repeat limit" in body:
                return None
            print(f"[sample] HTTP {e.code} attempt {attempt}: {body!r}", flush=True)
    raise RuntimeError("Ollama failed three times on one prompt")


# DATA can be a training mix (o_mix.py); only its factual rows (those with a
# reference answer) are sampled, study rows have none to grade against.
rows = [json.loads(l) for l in open(os.environ.get("DATA", os.path.join(ROOT, "data", "messages.jsonl")))]
rows = [r for r in rows if r.get("answer") and r.get("user_message")]
ids = {json.loads(l)["question"]: f"d{i:04d}" for i, l in enumerate(open(os.path.join(ROOT, "data", "distilled.jsonl")))}
# An empty <context> (the product's retrieval kept nothing for 116 of 863
# questions) leaves the reference underivable: no sample can be right for the
# right reason, so there is no preference to learn.
rows = [r for r in rows if r["source"] not in set(DEV_DOCS.values()) and "<context></context>" not in r["user_message"]]
done = {json.loads(l)["question"] for l in open(OUT)} if os.path.exists(OUT) else set()
t0, n_pass, n_all = time.time(), 0, 0
with open(OUT, "a") as out:
    for i, r in enumerate(rows):
        if r["question"] in done:
            continue
        # Ollama serves one sequence at a time per loaded model by default, so
        # the pool only overlaps HTTP and grading, never two generations.
        with ThreadPoolExecutor(2) as pool:
            answers = list(pool.map(lambda s: chat(r["user_message"], s), range(N)))
        case = {"case_type": "factual_number" if any(ch.isdigit() for ch in r["answer"]) else "factual_concept",
                "accepted_answers": [r["answer"]]}
        verdicts = [a is not None and grade.grade_answer(a, case)["pass"] for a in answers]
        n_pass += sum(verdicts); n_all += N
        out.write(json.dumps({"id": r.get("id") or ids.get(r["question"]), "lang": r["lang"], "source": r["source"],
                              "question": r["question"], "answer": r["answer"], "user_message": r["user_message"],
                              "samples": [{"text": a, "pass": v} for a, v in zip(answers, verdicts)]},
                             ensure_ascii=False) + "\n")
        out.flush()
        if i % 50 == 0:
            print(f"[sample] {i}/{len(rows)} pass {n_pass}/{n_all} {time.time() - t0:.0f}s", flush=True)
wall = round(time.time() - t0)
print(f"done pass {n_pass}/{n_all} {wall}s")
record(f"samples:{MODEL}", n=N, temperature=0.8, pass_rate=[n_pass, n_all], wall_s=wall, out=file_facts(OUT))
