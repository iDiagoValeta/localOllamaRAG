#!/usr/bin/env bash
# One experiment end to end: train (SFT or DPO), export to Ollama, dev evaluation.
#
#   scripts/run_exp.sh sft <run> [ENV=VALUE ...]     # d_train.py
#   scripts/run_exp.sh dpo <run> [ENV=VALUE ...]     # l_dpo.py
#
# Extra ENV=VALUE pairs reach the trainer (DATA, KD, R, LR, EPOCHS, BASE, ...).
# Wall time of each phase goes to runs/<run>/phases.json; the trainer and the
# dev evaluation record their own facts in runs/manifest.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIND="$1"; RUN="$2"; shift 2
PY="$ROOT/.venv/bin/python"
# Sequence lengths range from 300 to 14k tokens; without expandable segments the
# caching allocator fragments and reserves about twice the allocated peak.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The product venv of the checkout training/ lives in runs the dev evaluation.
REPO_PY="$(dirname "$ROOT")/.venv/bin/python"
mkdir -p "$ROOT/runs/$RUN"
# The embedder holds 1.2 GB; with it up, a long study row under KD ran sft-kd2 out
# of memory at step 256. It is only needed by the dev evaluation, so it is down
# while training and comes back just before.
EMB_PID="$ROOT/runs/embedder.pid"
[ -f "$EMB_PID" ] && kill "$(cat "$EMB_PID")" 2>/dev/null || true
t0=$(date +%s)
if [ "$KIND" = sft ]; then
  env RUN="$RUN" "$@" "$PY" "$ROOT/scripts/d_train.py" > "$ROOT/runs/$RUN.log" 2>&1
else
  env RUN="$RUN" "$@" "$PY" "$ROOT/scripts/l_dpo.py" > "$ROOT/runs/$RUN.log" 2>&1
fi
t1=$(date +%s)
"$ROOT/scripts/g_export.sh" "$RUN" >> "$ROOT/runs/$RUN.log" 2>&1
rm -f "$ROOT/models/$RUN/model-bf16.gguf"
t2=$(date +%s)
M="${MONKEYGRAB_MODELS_DIR:-$HOME/.local/share/monkeygrab/models}"
nohup "${LLAMA_SERVER:-llama-server}" -m "$M/embeddinggemma-2-Q8_0.gguf" --mmproj "$M/mmproj-embeddinggemma-2-Q8_0.gguf" \
  --embedding --host 127.0.0.1 --port 8091 -b 2048 -ub 2048 > "$ROOT/runs/embedder.log" 2>&1 &
echo $! > "$EMB_PID"
for _ in $(seq 60); do curl -sf localhost:8091/health > /dev/null && break; sleep 1; done
(cd "$ROOT/scripts" && "$REPO_PY" i_deveval.py "qwen06-rag:$RUN" 2>&1 | grep -v -i warn | tail -3) >> "$ROOT/runs/$RUN.log"
t3=$(date +%s)
printf '{"train_s": %d, "export_s": %d, "deveval_s": %d}\n' $((t1-t0)) $((t2-t1)) $((t3-t2)) > "$ROOT/runs/$RUN/phases.json"
tail -1 "$ROOT/runs/$RUN.log"
