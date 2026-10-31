#!/usr/bin/env bash
# One-command MonkeyGrab install on Linux (issue #559).
#
#   tools/install.sh               # install everything below, idempotently
#   tools/install.sh --check       # report prerequisites and pending steps, change nothing
#   tools/install.sh --with-corpus # also fetch the three PDF corpora (slow, optional)
#
# Prerequisites that need root (Ollama, the NVIDIA driver, python3-venv, git,
# llama.cpp) are checked and reported, never installed. Everything this script
# creates is recorded in the install manifest so tools/uninstall.sh can remove
# exactly that and nothing the user had before.
set -euo pipefail

# shellcheck source=tools/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CHECK_ONLY=0
WITH_CORPUS=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --with-corpus) WITH_CORPUS=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: tools/install.sh [--check] [--with-corpus]
  (no option)    install everything, idempotently; records what it creates for uninstall.sh
  --check        report prerequisites and pending steps, change nothing
  --with-corpus  also fetch the three PDF corpora (slow, optional)
USAGE
      exit 0 ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

FRONTEND_DIR="$REPO_ROOT/rag/web/frontend"
MISSING=()
PNPM=()

missing() { MISSING+=("$1"); printf ' MISSING  %s\n' "$1"; }
found() { printf '   ok     %s\n' "$1"; }

check_prerequisites() {
  say "Prerequisites:"

  if command -v git >/dev/null 2>&1; then found "git"; else missing "git (sudo apt install git)"; fi
  if command -v curl >/dev/null 2>&1; then found "curl"; else missing "curl (sudo apt install curl)"; fi

  if command -v python3 >/dev/null 2>&1 \
     && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    found "python3 $(python3 -c 'import platform; print(platform.python_version())')"
    if python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
      found "python3 venv"
    else
      missing "python3 venv (sudo apt install python3-venv)"
    fi
  else
    missing "python3 >= 3.10"
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    found "NVIDIA driver ($(nvidia-smi -L | head -n 1 | sed "s/ (UUID.*//"))"
  else
    missing "NVIDIA driver (nvidia-smi must list a GPU; MinerU and the reranker need CUDA)"
  fi

  if command -v pnpm >/dev/null 2>&1; then
    PNPM=(pnpm); found "pnpm"
  elif command -v corepack >/dev/null 2>&1; then
    PNPM=(corepack pnpm); found "pnpm via corepack"
  else
    missing "pnpm or corepack (Node 22+ ships corepack: corepack enable)"
  fi

  if command -v ollama >/dev/null 2>&1; then found "ollama client"; else
    missing "ollama (https://ollama.com/download)"
  fi
  local ollama_url
  if ollama_url="$(config_value ollama-url 2>/dev/null)" \
     && curl -fsS --max-time 5 "$ollama_url/api/tags" >/dev/null 2>&1; then
    found "Ollama reachable at $ollama_url"
  else
    missing "Ollama server reachable (start it: ollama serve)"
  fi

  local build
  command -v "$LLAMA_SERVER" >/dev/null 2>&1 && build="$(llama_build)"
  # Present is all that is checked here; whether it can load embeddinggemma-2
  # is decided by llama_probe once the GGUFs are on disk.
  if command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
    found "llama-server (build ${build:-unknown}, informational; capability is probed with the real model)"
  else
    missing "llama-server on PATH (llama.cpp that loads gemma-embedding2, b$KNOWN_GOOD_LLAMA_BUILD does; or set LLAMA_SERVER_BIN)"
  fi
}

# The manifest is per install location: a second checkout reusing the same
# state dir would otherwise inherit entries it did not create, and its uninstall
# would delete the first checkout's venvs.
claim_manifest() {
  if [ -f "$MANIFEST" ]; then
    local owner
    owner="$(awk -F'\t' '$1 == "repo" { print $2; exit }' "$MANIFEST")"
    [ "$(real_path "$owner")" = "$REPO_ROOT" ] \
      || die "$MANIFEST belongs to another checkout ($owner). Run its tools/uninstall.sh first, or set MONKEYGRAB_STATE_DIR."
  fi
  manifest_add repo "$REPO_ROOT"
}

# Records a path before creating it, and only when it does not exist yet, so a
# half-finished install is still fully removable and a pre-existing path is
# never claimed.
# The manifest holds resolved paths, because uninstall.sh compares them with the
# resolved roots: with ~/.local symlinked, an unresolved entry would be refused
# and uninstall could never finish.
claim_path() {
  if [ ! -e "$1" ]; then manifest_add path "$(real_path "$1")"; fi
}

pull_models() {
  # A variable, not a process substitution: only this form stops the script
  # when the config lookup itself fails.
  local model models
  models="$(config_value models)"
  while IFS= read -r model; do
    [ -n "$model" ] || continue
    if ollama show "$model" >/dev/null 2>&1; then
      say "Ollama model $model: already present"
    else
      manifest_add ollama-model "$model"
      say "Ollama model $model: pulling"
      ollama pull "$model"
    fi
  done <<< "$models"
}

download_embedder_file() {
  local file="$1" sha="$2" dest="$MODELS_DIR/$1" rc
  if [ -f "$dest" ]; then
    local actual
    actual="$(sha256sum "$dest" | cut -d' ' -f1)"
    [ "$actual" = "$sha" ] \
      || die "$dest exists but its sha256 is $actual, expected $sha. Not overwriting a file this script did not verify; remove it and re-run."
    say "$file: already present, sha256 ok"
    return
  fi
  claim_path "$dest"
  claim_path "$dest.part"
  say "$file: downloading from huggingface.co/$HF_REPO"
  # .part + resume: a dropped connection resumes instead of restarting a
  # 600 MB download, and a half file never sits under the final name.
  # Exit 33 is "range not satisfiable": the .part file was already complete
  # (killed between download and rename), and the sha256 check below decides.
  rc=0
  curl -fL --retry 3 --retry-delay 5 -C - -o "$dest.part" \
    "https://huggingface.co/$HF_REPO/resolve/main/$file" || rc=$?
  [ "$rc" -eq 0 ] || [ "$rc" -eq 33 ] || die "$file: download failed (curl exit $rc)"
  local actual
  actual="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then
    rm -f -- "$dest.part"
    die "$file: sha256 $actual does not match the expected $sha (download discarded)"
  fi
  mv -- "$dest.part" "$dest"
}

# Paths the app writes once it runs (indices, extraction cache, debug dumps,
# chat history). Claimed here, only if absent now, so uninstall can clear what
# this install produced without touching an index someone built before it.
claim_runtime_state() {
  local data_dir="${MONKEYGRAB_DATA_DIR:-$REPO_ROOT/rag}"
  # Absolute even when the directory does not exist yet: a relative claim
  # would resolve against whatever directory uninstall.sh runs from.
  data_dir="$(real_path "$data_dir")"
  claim_path "$data_dir/vector_db"
  claim_path "$data_dir/debug_rag"
  claim_path "$data_dir/historial_chat.json"
  claim_path "$REPO_ROOT/.mineru_cache"
  claim_path "$STATE_DIR/llama-server.log"
}

export_ollama_host
check_prerequisites

if [ "$CHECK_ONLY" -eq 1 ]; then
  say ""
  say "Pending steps:"
  if command -v python3 >/dev/null 2>&1; then
    while IFS= read -r model; do
      command -v ollama >/dev/null 2>&1 && ollama show "$model" >/dev/null 2>&1 \
        || say "  pull Ollama model $model"
    done < <(config_value models 2>/dev/null)
  fi
  [ -f "$MODELS_DIR/$EMBEDDER_FILE" ] || say "  download $EMBEDDER_FILE into $MODELS_DIR"
  [ -f "$MODELS_DIR/$MMPROJ_FILE" ] || say "  download $MMPROJ_FILE into $MODELS_DIR"
  if [ -f "$MODELS_DIR/$EMBEDDER_FILE" ] && [ -f "$MODELS_DIR/$MMPROJ_FILE" ]; then
    if command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
      say "  probe llama-server with the downloaded model:"
      llama_probe "$MODELS_DIR/$EMBEDDER_FILE" "$MODELS_DIR/$MMPROJ_FILE" \
        && say "    ok: llama-server loads embeddinggemma-2 with vision" \
        || MISSING+=("llama-server cannot serve embeddinggemma-2 (see above)")
    fi
  else
    say "  probe llama-server with the downloaded model (pending: needs the GGUFs above)"
  fi
  [ -x "$REPO_ROOT/.venv/bin/python" ] || say "  build .venv"
  [ -x "$REPO_ROOT/.venv-mineru/bin/python" ] || say "  build .venv-mineru"
  [ -f "$FRONTEND_DIR/dist/index.html" ] || say "  build the frontend"
  [ "${#MISSING[@]}" -eq 0 ] || { say ""; say "${#MISSING[@]} prerequisite(s) missing."; exit 1; }
  exit 0
fi

if [ "${#MISSING[@]}" -ne 0 ]; then
  say ""
  die "${#MISSING[@]} prerequisite(s) missing (listed above); install them and re-run. Nothing was changed."
fi

say ""
claim_manifest
claim_runtime_state

# A directory created here is recorded so uninstall can remove it once empty.
if [ ! -d "$MODELS_DIR" ]; then manifest_add dir "$(real_path "$MODELS_DIR")"; fi
mkdir -p "$MODELS_DIR"
download_embedder_file "$EMBEDDER_FILE" "$EMBEDDER_SHA256"
download_embedder_file "$MMPROJ_FILE" "$MMPROJ_SHA256"

# Before the multi-GB steps: a llama-server too old for this model should stop
# the install here, not after torch and the generator are downloaded.
say "Probing llama-server with the downloaded model"
llama_probe "$MODELS_DIR/$EMBEDDER_FILE" "$MODELS_DIR/$MMPROJ_FILE" \
  || die "llama-server cannot serve embeddinggemma-2; install a newer llama.cpp (the downloaded files stay for the re-run)"

pull_models

claim_path "$REPO_ROOT/.venv"
claim_path "$REPO_ROOT/.venv-mineru"
say "Building .venv and .venv-mineru (several GB: torch and MinerU)"
python3 "$REPO_ROOT/tools/setup_environments.py"

claim_path "$FRONTEND_DIR/node_modules"
claim_path "$FRONTEND_DIR/dist"
say "Building the web interface"
(cd "$FRONTEND_DIR" && "${PNPM[@]}" install --frozen-lockfile && "${PNPM[@]}" run build)

if [ "$WITH_CORPUS" -eq 1 ]; then
  say "Fetching the corpora"
  # Only PDFs absent before this run are claimed: the ones git already tracks
  # are part of the checkout, not something uninstall may delete.
  before="$(mktemp)"; after="$(mktemp)"
  find "$REPO_ROOT/rag/docs" -name '*.pdf' | sort > "$before"
  fetch_status=0
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/tools/fetch_corpus.py" || fetch_status=$?
  find "$REPO_ROOT/rag/docs" -name '*.pdf' | sort > "$after"
  while IFS= read -r pdf; do manifest_add path "$pdf"; done < <(comm -13 "$before" "$after")
  rm -f -- "$before" "$after"
  # Recorded before failing, so a fetch that dies midway is still removable.
  [ "$fetch_status" -eq 0 ] || die "tools/fetch_corpus.py failed (exit $fetch_status)"
fi

say ""
say "Installed. Start it with tools/run.sh"
