# shellcheck shell=bash
# Shared by install.sh, run.sh and uninstall.sh (issue #559). Sourced, never run.
#
# Everything the three scripts must agree on lives here: where the embedder
# files and the install manifest go, which llama.cpp build is new enough, how
# the manifest is read and written. Two copies of any of it would drift apart,
# and uninstall would then leave behind what install created.

# -P: the manifest compares real paths, so a symlinked checkout must not differ from itself.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Both locations are overridable from the environment (MONKEYGRAB_MODELS_DIR,
# MONKEYGRAB_STATE_DIR). Shell-only knobs: no Python code reads them, which is
# why they are not in .env.example.
MODELS_DIR="${MONKEYGRAB_MODELS_DIR:-$HOME/.local/share/monkeygrab/models}"
STATE_DIR="${MONKEYGRAB_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/monkeygrab}"
MANIFEST="$STATE_DIR/install-manifest.tsv"

# The README's embedder files. The hashes are of the Q8_0 files in this repo
# on Hugging Face; a mismatch means a truncated download or a different file.
HF_REPO="ggml-org/embeddinggemma-2-GGUF"
EMBEDDER_FILE="embeddinggemma-2-Q8_0.gguf"
EMBEDDER_SHA256="2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
MMPROJ_FILE="mmproj-embeddinggemma-2-Q8_0.gguf"
MMPROJ_SHA256="c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f"

# llama.cpp builds known to load architecture gemma-embedding2 (b11669 does,
# older builds may not). Informational only: the number --version prints is the
# commit count of the clone, so a shallow `git clone --depth 1 --branch b11669`
# reports "build 1". Whether a binary works is decided by llama_probe.
KNOWN_GOOD_LLAMA_BUILD=11669
LLAMA_SERVER="${LLAMA_SERVER_BIN:-llama-server}"

say() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# The product's own answer to "what is configured", so the defaults are not
# written down a second time here. Standard library only, so system python3
# is enough before any venv exists.
config_value() {
  python3 "$REPO_ROOT/tools/setup_environments.py" --print "$1"
}

# Prints the llama.cpp build number of the server binary, or nothing when it
# cannot be determined. Newer binaries print "version: 0.3.0-dev (build N, ...)",
# older ones "version: N (hash)".
llama_build() {
  local out
  out="$("$LLAMA_SERVER" --version 2>&1)" || return 0
  printf '%s\n' "$out" | sed -n -E 's/.*build ([0-9]+).*/\1/p; s/^version: ([0-9]+) .*/\1/p' | head -n 1
}

# State of the probe in flight, global so the signal trap can reach it.
_PROBE_PID=""
_PROBE_LOG=""
_PROBE_PREVIOUS_TRAPS=""

# Stops the probe's llama-server (TERM, then KILL after a short grace) and
# removes its temp log. Idempotent: it runs from the INT/TERM handlers and
# again from EXIT.
_probe_cleanup() {
  if [ -n "$_PROBE_PID" ]; then
    if kill -0 "$_PROBE_PID" 2>/dev/null; then
      kill "$_PROBE_PID" 2>/dev/null || true
      local i
      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 "$_PROBE_PID" 2>/dev/null || break
        sleep 0.1
      done
      kill -9 "$_PROBE_PID" 2>/dev/null || true
    fi
    wait "$_PROBE_PID" 2>/dev/null || true
  fi
  if [ -n "$_PROBE_LOG" ]; then rm -f -- "$_PROBE_LOG"; fi
  _PROBE_PID=""
  _PROBE_LOG=""
}

# The single way out of the probe: stop the server, remove the log, give the
# caller its traps back, then leave. $1 is an exit status for the signal
# handlers (the exit then runs the caller's own EXIT trap), "exit" when the
# shell is already exiting (the caller's EXIT trap is run by hand, because a
# trap installed while an EXIT trap is running never fires), or empty on a
# normal return.
_probe_finish() {
  _probe_cleanup
  trap - EXIT INT TERM
  if [ -n "$_PROBE_PREVIOUS_TRAPS" ]; then eval "$_PROBE_PREVIOUS_TRAPS"; fi
  _PROBE_PREVIOUS_TRAPS=""
  case "${1-}" in
    "") ;;
    exit)
      eval "set -- $(trap -p EXIT)"
      if [ -n "${3-}" ]; then eval "$3"; fi ;;
    *) exit "$1" ;;
  esac
}

# Capability probe: start the llama-server on a free local port with the real
# model and projector, wait (bounded by wall clock) for /health, require
# /props to report vision, then stop it. This is what "can serve
# embeddinggemma-2" means in practice; the version number cannot say it.
# Returns 0 when it loads, 1 with the observed reason on stderr when it does
# not. Ctrl+C or TERM during the probe stops the server it started.
llama_probe() {
  local model="$1" mmproj="$2" timeout="${LLAMA_PROBE_TIMEOUT:-120}"
  local port pid reason="" props deadline
  # No leading zero: bash would read "08" as a bad octal number and abort.
  if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: LLAMA_PROBE_TIMEOUT must be a positive number of seconds, got "%s"\n' "$timeout" >&2
    return 1
  fi
  port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  _PROBE_PREVIOUS_TRAPS="$(trap -p EXIT INT TERM)"
  # Installed before the server starts. A non-interactive shell starts async
  # children with SIGINT ignored, which is why the handlers kill it by pid.
  _PROBE_LOG="$(mktemp)"
  trap '_probe_finish exit' EXIT
  trap '_probe_finish 130' INT
  trap '_probe_finish 143' TERM
  "$LLAMA_SERVER" -m "$model" --mmproj "$mmproj" --embedding --port "$port" -b 2048 -ub 2048 >"$_PROBE_LOG" 2>&1 &
  # Bash runs a trap between statements, so a signal landing between the fork
  # above and this assignment (one statement wide) could still orphan the server.
  _PROBE_PID=$!
  pid=$_PROBE_PID
  deadline=$(( $(date +%s) + timeout ))
  while :; do
    if ! kill -0 "$pid" 2>/dev/null; then reason="it exited during startup"; break; fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      # Our process must still be alive, or something else owns that port now.
      kill -0 "$pid" 2>/dev/null || reason="it exited while another listener answered /health on port $port"
      break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then reason="/health did not answer within ${timeout}s"; break; fi
    sleep 1
  done
  if [ -z "$reason" ]; then
    if ! props="$(curl -fsS --max-time 5 "http://127.0.0.1:$port/props" 2>/dev/null)"; then
      reason="/props did not answer"
    elif ! python3 -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("modalities", {}).get("vision") is True else 1)' <<<"$props"; then
      reason="/props reports no vision support, so the projector (--mmproj) did not load"
    fi
  fi
  if [ -n "$reason" ]; then
    {
      printf 'error: %s cannot serve embeddinggemma-2: %s.\n' "$LLAMA_SERVER" "$reason"
      # An upgrade is only advice when the log says the architecture is unknown;
      # otherwise it would send someone to rebuild a binary that is fine.
      if grep -qi 'unknown model architecture' "$_PROBE_LOG"; then
        printf 'This llama.cpp does not know architecture gemma-embedding2: use a newer build (b%s is known to work; older ones may not know it).\n' "$KNOWN_GOOD_LLAMA_BUILD"
      elif [[ "$reason" == */props\ reports* ]]; then
        printf 'Check that %s is the projector for the model and that this llama-server supports --mmproj.\n' "$(basename "$mmproj")"
      elif [[ "$reason" == /health* ]]; then
        printf 'It may just be slow to load (CPU, cold disk cache): raise LLAMA_PROBE_TIMEOUT (seconds) and re-run.\n'
      fi
      printf 'Last lines of its output:\n'; tail -n 15 "$_PROBE_LOG" | sed 's/^/  /'
    } >&2
  fi
  _probe_finish
  [ -z "$reason" ]
}

# Host and port of an endpoint URL. A URL without an explicit port gets the
# scheme default instead of echoing the host back as if it were a port.
url_host() {
  local a="${1#*://}"; a="${a%%/*}"
  if [[ "$a" =~ ^(.*):[0-9]+$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else printf '%s\n' "$a"; fi
}
url_port() {
  local a="${1#*://}"; a="${a%%/*}"
  if [[ "$a" =~ :([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "$1" == https://* ]]; then printf '443\n'
  else printf '80\n'; fi
}

# `ollama pull/show/rm` read OLLAMA_HOST while the product reads OLLAMA_BASE_URL
# first; exporting the product's resolved URL makes the CLI talk to the very
# server the reachability check asked.
export_ollama_host() {
  local url
  if url="$(config_value ollama-url 2>/dev/null)"; then OLLAMA_HOST="$url"; export OLLAMA_HOST; fi
}

# Lexical and symlink-resolved forms of a path, absolute and without "..",
# "//" or trailing slashes. Both must pass the allow-list in uninstall.sh.
norm_path() { realpath -m -s -- "$1"; }
real_path() { realpath -m -- "$1"; }

# Manifest: tab-separated "kind<TAB>value" lines. Kinds: repo, path, dir,
# ollama-model. Only what install.sh itself created is ever written, which is
# the whole basis of what uninstall.sh is allowed to delete.
manifest_has() { [ -f "$MANIFEST" ] && grep -qxF -- "$1"$'\t'"$2" "$MANIFEST"; }

manifest_add() {
  mkdir -p "$STATE_DIR"
  if ! manifest_has "$1" "$2"; then
    printf '%s\t%s\n' "$1" "$2" >> "$MANIFEST"
  fi
}
