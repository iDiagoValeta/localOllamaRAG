"""
Test suite for Ollama no-think mode suppression strategies.

Validates three different approaches to suppress the <think> reasoning
block in Qwen3-based models served through Ollama: (A) raw prompt with
a pre-filled empty think block, (B) raw prompt with /no_think user
directive, and (C) a derived Modelfile with the think parameter toggled.
Results are summarized showing which strategies successfully suppressed
reasoning output.

Usage:
    python scripts/tests/test_nothink.py
Dependencies:
    - requests
    - A running Ollama server with the Qwen3-FineTuned model loaded
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Constants and Qwen3 template
#
#  HELPERS
#  +-- 2. sep, stream_generate, assess
#
#  TEST CASES
#  +-- 3. Strategy A  raw prompt + pre-filled empty think block
#  +-- 4. Strategy B  raw prompt + /no_think directive
#  +-- 5. Strategy C  derived Modelfile with native Qwen3 template
#  +-- 6. Summary
#
# ─────────────────────────────────────────────

import json, subprocess, tempfile, time
from pathlib import Path
import requests

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS AND QWEN3 TEMPLATE
# ─────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "Qwen3-FineTuned:latest"
MODEL_TEST = "Qwen3-NoThink-Test:latest"
SYSTEM = "You are a helpful assistant."
QUESTION = "What is 2+2? Answer briefly."

QWEN3_TEMPLATE = (
    "{%- if .System %}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{%- end %}<|im_start|>user\n"
    "{{ .Prompt }}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{%- if .Thinking %}<think>\n"
    "{{ .Thinking }}</think>\n"
    "{%- else %}<think>\n"
    "</think>\n"
    "{%- end %}{{ .Response }}"
)


# ─────────────────────────────────────────────
# SECTION 2: HELPERS
# ─────────────────────────────────────────────

def sep(label):
    """Print a visual separator line with a centered label."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print("=" * 70)


def stream_generate(payload, timeout=30):
    """Send a streaming generate request to Ollama and return the full response text."""
    full = ""
    try:
        with requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                           json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    tok = chunk.get("response", "")
                    if tok:
                        print(tok, end="", flush=True)
                        full += tok
                    if chunk.get("done"):
                        break
    except requests.exceptions.Timeout:
        print("\n  [TIMEOUT]")
    print()
    return full


def assess(text):
    """Evaluate whether the output contains reasoning in a <think> block.

    Returns:
        A string verdict: 'OK - NO THINK' if no think block is present,
        'OK - empty think' if the block is empty, or 'FAIL - REASONS' if
        the model produced reasoning content.
    """
    if "<think>" not in text:
        return "OK - NO THINK"
    inner = text[text.find("<think>") + 7:]
    closing = inner.find("</think>")
    if closing != -1 and len(inner[:closing].strip()) == 0:
        return "OK - empty think"
    return "FAIL - REASONS"


# ─────────────────────────────────────────────
# SECTION 3: STRATEGY A -- raw prompt + pre-filled empty think block
# ─────────────────────────────────────────────

sep("A) raw:true + <think></think> pre-filled")
prompt_a = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n")
resp_a = stream_generate({"model": MODEL, "prompt": prompt_a, "raw": True,
    "stream": True, "options": {"temperature": 0.1, "num_ctx": 2048,
    "stop": ["<|im_end|>", "<|im_start|>"]}})
print(f"  -> {assess(resp_a)}")
time.sleep(1)

# ─────────────────────────────────────────────
# SECTION 4: STRATEGY B -- raw prompt + /no_think directive
# ─────────────────────────────────────────────

sep("B) raw:true + /no_think in user message")
prompt_b = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n/no_think\n{QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n")
resp_b = stream_generate({"model": MODEL, "prompt": prompt_b, "raw": True,
    "stream": True, "options": {"temperature": 0.1, "num_ctx": 2048,
    "stop": ["<|im_end|>", "<|im_start|>"]}})
print(f"  -> {assess(resp_b)}")
time.sleep(1)

# ─────────────────────────────────────────────
# SECTION 5: STRATEGY C -- derived Modelfile with native Qwen3 template
# ─────────────────────────────────────────────

sep("C) Derived Modelfile with native Qwen3 template")
with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile",
                                  delete=False, encoding="utf-8") as f:
    f.write("FROM Qwen3-FineTuned:latest\n\n")
    f.write('TEMPLATE  + QWEN3_TEMPLATE + \n\n')
    f.write(f'SYSTEM "{SYSTEM}"\n\n')
    f.write("PARAMETER num_ctx 8192\nPARAMETER repeat_penalty 1.15\n")
    f.write("PARAMETER stop <|im_start|>\nPARAMETER stop <|im_end|>\n")
    f.write("PARAMETER temperature 0.15\nPARAMETER top_p 0.9\n")
    mf_path = f.name

res = subprocess.run(["ollama", "create", MODEL_TEST, "-f", mf_path],
                     capture_output=True, text=True)
resp_c_no = resp_c_yes = ""
if res.returncode != 0:
    print(f"ERROR: {res.stderr}")
else:
    print("OK. Testing think:false ...")
    resp_c_no = stream_generate({"model": MODEL_TEST, "system": SYSTEM,
        "prompt": QUESTION, "stream": True, "think": False,
        "options": {"temperature": 0.1, "num_ctx": 2048}})
    print(f"  -> {assess(resp_c_no)}")
    time.sleep(1)
    print("Testing think:true ...")
    resp_c_yes = stream_generate({"model": MODEL_TEST, "system": SYSTEM,
        "prompt": QUESTION, "stream": True, "think": True,
        "options": {"temperature": 0.1, "num_ctx": 2048}})
    print(f"  -> {assess(resp_c_yes)}")
    subprocess.run(["ollama", "rm", MODEL_TEST], capture_output=True)
    print("Temporary model deleted.")
Path(mf_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────
# SECTION 6: SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for lbl, r in [("A raw+empty_think", resp_a), ("B raw+no_think", resp_b),
               ("C modelfile think:false", resp_c_no),
               ("C modelfile think:true", resp_c_yes)]:
    print(f"  {assess(r):25s}  {lbl}")
print()
