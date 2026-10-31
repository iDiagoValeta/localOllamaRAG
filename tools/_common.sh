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
# b10751 does not). Informational only: the number --version prints is the
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

# Capability probe: start the llama-server on a free local port with the real
# model and projector, wait (bounded) for /health, require /props to report
# vision, then stop it. This is what "new enough for embeddinggemma-2" means in
# practice; the version number cannot say it. Returns 0 when it loads, 1 with
# an actionable message on stderr when it does not.
llama_probe() {
  local model="$1" mmproj="$2" timeout="${LLAMA_PROBE_TIMEOUT:-120}"
  local port log pid reason="" waited=0 props
  port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  log="$(mktemp)"
  "$LLAMA_SERVER" -m "$model" --mmproj "$mmproj" --embedding --port "$port" -b 2048 -ub 2048 >"$log" 2>&1 &
  pid=$!
  until curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then reason="it exited during startup"; break; fi
    waited=$((waited + 1))
    if [ "$waited" -ge "$timeout" ]; then reason="/health did not answer within ${timeout}s"; break; fi
    sleep 1
  done
  if [ -z "$reason" ]; then
    if ! props="$(curl -fsS --max-time 5 "http://127.0.0.1:$port/props" 2>/dev/null)"; then
      reason="/props did not answer"
    elif ! python3 -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("modalities", {}).get("vision") is True else 1)' <<<"$props"; then
      reason="/props reports no vision support, so the projector (--mmproj) did not load"
    fi
  fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if [ -n "$reason" ]; then
    {
      printf 'error: %s cannot serve embeddinggemma-2: %s.\n' "$LLAMA_SERVER" "$reason"
      printf 'It needs a llama.cpp build that knows architecture gemma-embedding2 (b%s does, b10751 does not).\n' "$KNOWN_GOOD_LLAMA_BUILD"
      printf 'Last lines of its output:\n'; tail -n 15 "$log" | sed 's/^/  /'
    } >&2
    rm -f -- "$log"
    return 1
  fi
  rm -f -- "$log"
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
