#!/usr/bin/env bash
# Stage G: merged HF checkpoint -> GGUF bf16 -> Q4_K_M -> Ollama model "qwen06-rag:<tag>".
#
#   scripts/g_export.sh <run>            # models/<run>/merged
#   scripts/g_export.sh base0            # the untouched base, saved through the same path (fair baseline)
#   scripts/g_export.sh <run> Q8_0       # another quantisation, as "qwen06-rag:<run>-q8_0"
#
# Same llama.cpp build (b11669) as the embedder, and the Modelfile of Ollama's
# own qwen3:0.6b with only FROM swapped, so the template, stop tokens and
# parameters match the model this one is compared against.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"
QUANT="${2:-Q4_K_M}"
TAG="$RUN"
[ "$QUANT" = Q4_K_M ] || TAG="$RUN-$(echo "$QUANT" | tr 'A-Z' 'a-z')"
LLAMA="${LLAMA_CPP_DIR:-$HOME/.local/lib/llama-b11669/llama-b11669}"
DIR="$ROOT/models/$RUN"
"$ROOT/.venv/bin/python" "$ROOT/vendor/llama.cpp/convert_hf_to_gguf.py" "$DIR/merged" --outtype bf16 --outfile "$DIR/model-bf16.gguf"
LD_LIBRARY_PATH="$LLAMA" "$LLAMA/llama-quantize" "$DIR/model-bf16.gguf" "$DIR/model-$QUANT.gguf" "$QUANT" >/dev/null
Q="$DIR/model-$QUANT.gguf"
MF="$(mktemp)"
sed "s|^FROM .*|FROM $Q|" "$ROOT/Modelfile.qwen3-0.6b" > "$MF"
ollama create "qwen06-rag:$TAG" -f "$MF" >/dev/null
rm -f "$MF"
echo "qwen06-rag:$TAG <- $Q ($(du -h "$Q" | cut -f1))"
