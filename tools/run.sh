#!/usr/bin/env bash
# Start MonkeyGrab on Linux (issue #559): the embedder's llama-server, then the
# web app. Ctrl+C or any exit stops the llama-server this script started, and
# only that one.
#
#   tools/run.sh              # web UI at http://localhost:5000
#   tools/run.sh --check      # report whether it could start, start nothing
set -euo pipefail

# shellcheck source=tools/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: tools/run.sh [--check]
  (no option)  start the embedder's llama-server (unless one already answers) and the
               web UI at http://localhost:5000; Ctrl+C stops what this script started
  --check      report whether it could start, start nothing
USAGE
      exit 0 ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

PYTHON="$REPO_ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || die "$PYTHON not found. Run tools/install.sh first."
[ -f "$REPO_ROOT/rag/web/frontend/dist/index.html" ] \
  || die "the web interface is not built (/ would answer 503). Run tools/install.sh first."

embedder_url="$(config_value embedder-url)"
embedder_host="$(url_host "$embedder_url")"
embedder_port="$(url_port "$embedder_url")"
embedder_up() { curl -fsS --max-time 3 "$embedder_url/health" >/dev/null 2>&1; }

llama_pid=""
app_pid=""
log_file="$STATE_DIR/llama-server.log"

# Stops the app first (it holds the embedder connection) and then the
# llama-server, but only a process this script started: an embedder that was
# already answering belongs to someone else and keeps running.
cleanup() {
  trap - EXIT INT TERM
  if [ -n "$app_pid" ] && kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
  if [ -n "$llama_pid" ] && kill -0 "$llama_pid" 2>/dev/null; then
    say "Stopping llama-server (pid $llama_pid)"
    kill "$llama_pid" 2>/dev/null || true
    wait "$llama_pid" 2>/dev/null || true
  fi
}

if embedder_up; then
  say "Embedder already answering at $embedder_url; reusing it (not started by this script, so not stopped by it)"
else
  case "$embedder_host" in
    127.0.0.1|localhost|"[::1]") ;;
    *) die "embedder at $embedder_url is not reachable and is not a local address, so this script will not start one." ;;
  esac
  command -v "$LLAMA_SERVER" >/dev/null 2>&1 \
    || die "$LLAMA_SERVER not found on PATH (llama.cpp that loads gemma-embedding2, b$KNOWN_GOOD_LLAMA_BUILD does; or set LLAMA_SERVER_BIN)."
  for f in "$EMBEDDER_FILE" "$MMPROJ_FILE"; do
    [ -f "$MODELS_DIR/$f" ] || die "$MODELS_DIR/$f missing. Run tools/install.sh first."
  done
  if [ "$CHECK_ONLY" -eq 1 ]; then
    say "Would start: $LLAMA_SERVER -m $MODELS_DIR/$EMBEDDER_FILE --mmproj $MODELS_DIR/$MMPROJ_FILE --embedding --port $embedder_port -b 2048 -ub 2048"
  else
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir -p "$STATE_DIR"
    say "Starting llama-server on port $embedder_port (log: $log_file)"
    # The README's command. --mmproj is required for image embeddings and the
    # 2048 batch sizes for chunks longer than the 512-token default.
    "$LLAMA_SERVER" -m "$MODELS_DIR/$EMBEDDER_FILE" --mmproj "$MODELS_DIR/$MMPROJ_FILE" \
      --embedding --port "$embedder_port" -b 2048 -ub 2048 >"$log_file" 2>&1 &
    llama_pid=$!
    for _ in $(seq 1 120); do
      embedder_up && break
      kill -0 "$llama_pid" 2>/dev/null || die "llama-server exited during startup; see $log_file"
      sleep 1
    done
    embedder_up || die "llama-server did not answer /health within 120 s; see $log_file"
  fi
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "Would start: $PYTHON rag/web/app.py"
  exit 0
fi

ollama_url="$(config_value ollama-url)"
curl -fsS --max-time 5 "$ollama_url/api/tags" >/dev/null 2>&1 \
  || warn "Ollama is not reachable at $ollama_url; start it (ollama serve) or answers will fail."

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Background plus wait, not foreground: a non-interactive shell defers its
# traps until a foreground child exits, so a TERM aimed at this script alone
# would leave the app and the llama-server running.
cd "$REPO_ROOT"
"$PYTHON" rag/web/app.py &
app_pid=$!
status=0
wait "$app_pid" || status=$?
app_pid=""
exit "$status"
