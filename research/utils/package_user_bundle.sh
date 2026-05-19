#!/usr/bin/env bash
# Package a minimal "user" tarball: rag (includes rag/web/) + docs, excluding research/ and RagBench PDF trees.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ZIP="${1:-${REPO_ROOT}/monkeygrab-user-bundle.zip}"
STAGE="$(mktemp -d)"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

cp "$REPO_ROOT/README.md" "$REPO_ROOT/CLAUDE.md" "$REPO_ROOT/pytest.ini" "$STAGE/"

rsync -a --exclude 'en_ragbench_dev' --exclude 'en_ragbench_eval' --exclude 'en_ragbench_visual' \
  --exclude 'vector_db' --exclude 'debug_rag' --exclude '__pycache__' --exclude 'web/frontend/node_modules' \
  "$REPO_ROOT/rag/" "$STAGE/rag/"

rm -f "$OUT_ZIP"
( cd "$STAGE" && zip -r -q "$OUT_ZIP" . )
echo "Wrote $OUT_ZIP"
