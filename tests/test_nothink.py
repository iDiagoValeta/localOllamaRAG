"""
Test suite for Ollama no-think mode suppression strategies.

Validates three different approaches to suppress the <think> reasoning
block in Qwen3-based models served through Ollama: (A) raw prompt with
a pre-filled empty think block, (B) raw prompt with /no_think user
directive, and (C) a derived Modelfile with the think parameter toggled.

The experiment is intentionally opt-in under pytest because it requires a
running Ollama server and local models. Direct script execution still runs
the full experiment.

Usage:
    python research/tests/core/test_nothink.py
    RUN_OLLAMA_INTEGRATION=1 python -m pytest research/tests/core/test_nothink.py

Dependencies:
    - requests
    - A running Ollama server with the Qwen3-FineTuned model loaded
"""

# ------------------------------------------------------------
# MODULE MAP -- Section index
# ------------------------------------------------------------
#
# CONFIGURATION
# +-- 1. Imports and constants
#
# HELPERS
# +-- 2. sep, stream_generate, assess, ollama_available
#
# EXPERIMENT
# +-- 3. run_nothink_experiment
# +-- 4. print_summary
#
# ENTRY
# +-- 5. pytest integration test and main()
#
# ------------------------------------------------------------

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

# ------------------------------------------------------------
# SECTION 1: IMPORTS AND CONSTANTS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# SECTION 2: HELPERS
# ------------------------------------------------------------

def sep(label: str) -> None:
    """Print a visual separator line with a centered label."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print("=" * 70)


def stream_generate(payload: dict, timeout: int = 30) -> str:
    """Send a streaming generate request to Ollama and return response text."""
    full = ""
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
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


def assess(text: str) -> str:
    """Evaluate whether the output contains reasoning in a <think> block.

    Args:
        text: Model response text.

    Returns:
        Verdict string: OK when reasoning is absent or the think block is
        empty, FAIL otherwise.
    """
    if "<think>" not in text:
        return "OK - NO THINK"
    inner = text[text.find("<think>") + 7:]
    closing = inner.find("</think>")
    if closing != -1 and len(inner[:closing].strip()) == 0:
        return "OK - empty think"
    return "FAIL - REASONS"


def ollama_available(timeout: float = 2.0) -> bool:
    """Return whether a local Ollama server responds quickly."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True


# ------------------------------------------------------------
# SECTION 3: NOTHINK EXPERIMENT
# ------------------------------------------------------------

def run_nothink_experiment() -> dict[str, str]:
    """Run all no-think suppression strategies against a live Ollama server.

    Returns:
        Mapping from strategy label to full model response text.
    """
    sep("A) raw:true + <think></think> pre-filled")
    prompt_a = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n"
    )
    resp_a = stream_generate({
        "model": MODEL,
        "prompt": prompt_a,
        "raw": True,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
            "stop": ["<|im_end|>", "<|im_start|>"],
        },
    })
    print(f"  -> {assess(resp_a)}")
    time.sleep(1)

    sep("B) raw:true + /no_think in user message")
    prompt_b = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n/no_think\n{QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    resp_b = stream_generate({
        "model": MODEL,
        "prompt": prompt_b,
        "raw": True,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
            "stop": ["<|im_end|>", "<|im_start|>"],
        },
    })
    print(f"  -> {assess(resp_b)}")
    time.sleep(1)

    sep("C) Derived Modelfile with native Qwen3 template")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".Modelfile",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write("FROM Qwen3-FineTuned:latest\n\n")
        f.write('TEMPLATE """\n')
        f.write(QWEN3_TEMPLATE)
        f.write('\n"""\n\n')
        f.write(f'SYSTEM "{SYSTEM}"\n\n')
        f.write("PARAMETER num_ctx 8192\nPARAMETER repeat_penalty 1.15\n")
        f.write("PARAMETER stop <|im_start|>\nPARAMETER stop <|im_end|>\n")
        f.write("PARAMETER temperature 0.15\nPARAMETER top_p 0.9\n")
        mf_path = f.name

    res = subprocess.run(
        ["ollama", "create", MODEL_TEST, "-f", mf_path],
        capture_output=True,
        text=True,
    )
    resp_c_no = ""
    resp_c_yes = ""
    try:
        if res.returncode != 0:
            print(f"ERROR: {res.stderr}")
        else:
            print("OK. Testing think:false ...")
            resp_c_no = stream_generate({
                "model": MODEL_TEST,
                "system": SYSTEM,
                "prompt": QUESTION,
                "stream": True,
                "think": False,
                "options": {"temperature": 0.1, "num_ctx": 2048},
            })
            print(f"  -> {assess(resp_c_no)}")
            time.sleep(1)
            print("Testing think:true ...")
            resp_c_yes = stream_generate({
                "model": MODEL_TEST,
                "system": SYSTEM,
                "prompt": QUESTION,
                "stream": True,
                "think": True,
                "options": {"temperature": 0.1, "num_ctx": 2048},
            })
            print(f"  -> {assess(resp_c_yes)}")
    finally:
        subprocess.run(["ollama", "rm", MODEL_TEST], capture_output=True)
        Path(mf_path).unlink(missing_ok=True)

    results = {
        "A raw+empty_think": resp_a,
        "B raw+no_think": resp_b,
        "C modelfile think:false": resp_c_no,
        "C modelfile think:true": resp_c_yes,
    }
    print_summary(results)
    return results


# ------------------------------------------------------------
# SECTION 4: SUMMARY
# ------------------------------------------------------------

def print_summary(results: dict[str, str]) -> None:
    """Print a compact verdict table for experiment responses."""
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for label, response in results.items():
        print(f"  {assess(response):25s}  {label}")
    print()


# ------------------------------------------------------------
# SECTION 5: PYTEST AND ENTRY POINT
# ------------------------------------------------------------

def test_nothink_strategies() -> None:
    """Pytest integration entry point, skipped unless explicitly enabled."""
    import pytest

    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama integration test.")
    if not ollama_available():
        pytest.skip("Ollama server is not available at localhost:11434.")

    results = run_nothink_experiment()
    assert any(assess(response).startswith("OK") for response in results.values())


def main() -> int:
    """Run the manual no-think experiment from the command line."""
    if not ollama_available():
        print(f"ERROR: Ollama server is not available at {OLLAMA_BASE_URL}.")
        return 1
    run_nothink_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
