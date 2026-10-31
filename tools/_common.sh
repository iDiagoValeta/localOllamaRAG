# shellcheck shell=bash
# Shared by install.sh, run.sh and uninstall.sh (issue #559). Sourced, never run.
#
# Everything the three scripts must agree on lives here: where the embedder
# files and the install manifest go, which llama.cpp build is new enough, how
# the manifest is read and written. Two copies of any of it would drift apart,
# and uninstall would then leave behind what install created.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# The lowest llama.cpp build known to load architecture gemma-embedding2
# (b11669 does, b10751 does not). The real threshold lies somewhere between;
# this is the version that has been seen working, not a guess at the edge.
MIN_LLAMA_BUILD=11669
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

# host:port of the embedder endpoint, for the health probe and for --port.
url_host() { local u="${1#*://}"; u="${u%%/*}"; printf '%s\n' "${u%:*}"; }
url_port() { local u="${1#*://}"; u="${u%%/*}"; printf '%s\n' "${u##*:}"; }

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
