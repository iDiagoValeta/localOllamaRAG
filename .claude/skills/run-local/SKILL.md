---
name: run-local
description: Use when launching MonkeyGrab locally, verifying a running instance is actually healthy, or diagnosing a start that fails or runs oddly slowly. Covers Ollama not answering, missing model roles, a stale frontend build, a port already taken, and the two shapes of VRAM contention between the generator and the jina-clip worker: a CUDA out-of-memory, or a silent fallback to CPU.
---

# Run MonkeyGrab locally

## Overview

Running MonkeyGrab means two processes, not one: the **Ollama server**, which
holds LLM weights in VRAM, and the **Flask app**, which owns retrieval and
embeds queries through the isolated jina-clip worker. They are separate
processes competing for the same card, and that is where local runs break.

Reference material lives elsewhere and is not repeated here: install and
configuration in `README.md`, the operating contract in `AGENTS.md`, per-command
detail in `AGENTS.md` §9.

## Quick reference

| Goal | Command |
|---|---|
| Is this machine set up? | `python tools/setup_environments.py --check` |
| Launch the web app | `.venv/bin/python rag/web/app.py` |
| Is the app up? | `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/` |
| Is Ollama up? | `curl -s http://127.0.0.1:5000/api/ollama` |
| What holds VRAM right now? | `ollama ps` and `nvidia-smi` |
| Free the generator's VRAM | `curl -s http://localhost:11434/api/generate -d '{"model":"<tag>","keep_alive":0}'` |

## 1. Preflight

```bash
python tools/setup_environments.py --check   # both interpreters, CUDA, MinerU, jina-clip, model roles
ls rag/web/frontend/dist/index.html          # built frontend; app serves a bare page without it
```

No `dist/`: `cd rag/web/frontend && pnpm install && pnpm run build` (pnpm only,
`packageManager` is pinned).

`--check` reports each component separately and never pulls anything. Two of
its warnings are expected rather than blocking:

- **jina-clip "BUSY, not missing"** means the card has no free memory because
  another instance or a resident Ollama model holds it. The install is fine.
  Free the VRAM and re-check.
- **"missing model(s) for the configured roles"** can be false. It resolves
  roles from the environment and its own module default, not from
  `settings.json` (issue #215), so it names a model the app will not use.
  `curl -s http://127.0.0.1:5000/api/models` is what the app actually resolves;
  trust that against `ollama list`.

Ollama does not need starting by hand. `start_ollama_if_needed()` in
`rag/web/app.py` launches `ollama serve` on a daemon thread at boot when the
binary exists and nothing answers; the UI retries through `POST /api/ollama/start`.
Where Ollama runs under systemd, that path is a no-op and `systemctl status ollama`
is the truth.

## 2. Launch

```bash
.venv/bin/python rag/web/app.py    # http://127.0.0.1:5000
```

Use the venv interpreter explicitly. A bare `python` picks up whatever is on
PATH and fails later, at the first torch import, not at startup.

Binding is `127.0.0.1` by design: reaching it from another device on the LAN
needs an explicit host change, which also exposes an app with no authentication.

## 3. Smoke test: drive it, don't just curl the port

A 200 on `/` only proves Flask bound the socket. Walk outward until a real
query answers:

```bash
curl -s http://127.0.0.1:5000/api/ollama    # {"running": true, "version": ...}
curl -s http://127.0.0.1:5000/api/models    # the four roles, resolved
curl -s http://127.0.0.1:5000/api/stores    # three stores, active one flagged
curl -s http://127.0.0.1:5000/api/init      # loads the collection; slow on first call
```

Then the pipeline itself, which is the only check that exercises Ollama, the
FAISS index and the jina-clip worker together:

```bash
curl -s -N -X POST http://127.0.0.1:5000/api/rag \
  -H 'Content-Type: application/json' \
  -d '{"message":"<a question the active corpus answers>","stream":true}'
```

Healthy output is an SSE stream of `token` events ending in `done` with a
`sources` array. A JSON object with `"ok": false` instead means the pipeline
hard-failed, by policy (`AGENTS.md` rule 8), and its `message` names the real
cause. Read it: adapters in this repo never degrade silently, so the first error
is the true one. The Ollama server is not one of them and does degrade quietly,
which is what §4 covers.

## 4. Ollama, specifically

- **One endpoint, one reader.** `OLLAMA_BASE_URL` (falling back to Ollama's own
  `OLLAMA_HOST`) resolves once, in `monkeygrab.config.env`. Point it at another
  machine and every call follows, generation included: the way out when the
  local card cannot hold the generator you want.
- **The model roles must exist locally.** `ollama list` against the four roles
  in `/api/models`. A role naming an unpulled tag fails at first use, not at
  startup. Nothing here pulls models on your behalf.
- **`OLLAMA_KEEP_ALIVE` is the project's variable, not the server's.** Every
  call sends `keep_alive` explicitly (default 120s, `rag/chat_pdfs.py`), so it
  overrides whatever the systemd unit exports. Checking `systemctl show ollama
  -p Environment` tells you what other clients get, not what MonkeyGrab asks for.

### VRAM contention, in its two shapes

On a small card (8 GB) the generator and the jina-clip worker do not both fit.
Which symptom you get depends on **which process reached the GPU first**, and
Ollama decides GPU or CPU once, when it loads the weights.

**Ollama got there first.** It takes several GB, and the next query that has to
embed a question hard-fails, correctly (`AGENTS.md` rule 8):

```
jina-clip worker reported an error for op='text': CUDA out of memory.
Tried to allocate 490.00 MiB. GPU 0 has a total capacity of 7.60 GiB of
which 422.62 MiB is free.
```

**The worker got there first.** Nothing fails. Ollama quietly loads the model
into system RAM instead and generation gets much slower. The repo's no-silent-
fallback policy does not reach here: that decision belongs to the Ollama server,
not to an adapter in this codebase, so nothing in the app will tell you.

`ollama ps` distinguishes them, in the `PROCESSOR` column:

```
NAME                    SIZE     PROCESSOR         UNTIL
Ling-3.0-tiny...        5.1 GB   98%/2% CPU/GPU    About a minute from now
```

`98%/2% CPU/GPU` is the degraded case. `100% GPU` is the healthy one. Pair it
with `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`,
which names `llama-server` and the `.venv-mineru` worker with their sizes.

Recover from the OOM by unloading the generator (`keep_alive: 0` against
`/api/generate`) and repeating the query. Reduce how often either shape happens
by setting `OLLAMA_KEEP_ALIVE=0` in `.env`, which costs a cold model load per
query: 93-95% of a query's wall time (issue #25). The trade is latency against
headroom, and on a card this size you do not get both.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` | An instance is already running | `curl` it before assuming it is dead; kill the old one only if it is |
| Page loads bare, no UI | `rag/web/frontend/dist/` missing | `pnpm install && pnpm run build` |
| `/api/ollama` reports `running: false` | Server down or wrong endpoint | `systemctl status ollama`; check `OLLAMA_BASE_URL` |
| `ok: false` with `model not found` | Role names a tag that is not pulled | `ollama list`, then pull it or reassign the role in the UI |
| CUDA OOM on a RAG query | Ollama took the VRAM first | See VRAM contention above |
| Generation suddenly much slower | Ollama loaded into RAM, not VRAM | `ollama ps`: `PROCESSOR` says CPU |
| UI warns the stored index does not match the config | Index-time flags changed since indexing | Reindex from the UI, deliberately: a full pass can take an hour |
