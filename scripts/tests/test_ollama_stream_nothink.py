#!/usr/bin/env python3
"""
Streaming test for Ollama models with reasoning suppression.

Tests one or more Ollama-hosted models using the streaming generate API,
attempting to suppress <think> reasoning blocks. Supports configurable
prompts, system messages, think mode toggling, and optional post-processing
to strip any think blocks from the output. Reports whether each model
successfully avoided producing reasoning content.

Usage:
    python scripts/tests/test_ollama_stream_nothink.py
    python scripts/tests/test_ollama_stream_nothink.py --model qwen3:14b --prompt "Hello"
    python scripts/tests/test_ollama_stream_nothink.py -m qwen3-base-direct -m qwen3:4b-instruct
Dependencies:
    - requests
    - A running Ollama server with the target models loaded
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODELS = ["qwen3:14b", "qwen3.5:9b"]
DEFAULT_PROMPT = "What is 2+2? Answer in one brief sentence."
DEFAULT_SYSTEM = "You are a helpful assistant. Answer concisely."


def stream_generate(payload: dict, timeout: int = 60):
    """Send a streaming generate request and return the full response text."""
    full = ""
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
    print()
    return full


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from the output text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_model(
    model: str,
    prompt: str,
    system: str,
    think: bool,
    strip_think: bool,
    suffix_prompt: str = "",
) -> tuple[str, str]:
    """Run a model and return (raw_output, processed_output).

    Args:
        model: Ollama model name to query.
        prompt: User prompt text.
        system: System prompt text.
        think: Whether to enable think mode in the request.
        strip_think: Whether to strip think blocks from the output.
        suffix_prompt: Optional suffix appended to the prompt.

    Returns:
        A tuple of (raw_response, processed_response).
    """
    user_prompt = prompt + suffix_prompt
    payload = {
        "model": model,
        "system": system,
        "prompt": user_prompt,
        "stream": True,
        "think": think,
        "options": {"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 8192},
    }
    raw = stream_generate(payload)
    processed = strip_think_blocks(raw) if strip_think else raw
    return raw, processed


def assess(text: str) -> str:
    """Evaluate whether the output contains reasoning content.

    Returns:
        A string verdict: 'OK - no think' if no think block is present,
        'OK - empty think' if the block is empty, or 'REASONS' if the
        model produced reasoning content.
    """
    if "<think>" not in text:
        return "OK - no think"
    inner = text[text.find("<think>") + 7 :]
    closing = inner.find("</think>")
    if closing != -1 and len(inner[:closing].strip()) == 0:
        return "OK - empty think"
    return "REASONS"


def main():
    parser = argparse.ArgumentParser(description="Test Ollama models with streaming and reasoning suppression")
    parser.add_argument("-m", "--model", action="append", dest="models", help="Model(s) to test")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Test prompt")
    parser.add_argument("-s", "--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--think", action="store_true", help="Force think=true (default is false)")
    parser.add_argument("--no-strip", action="store_true", help="Do not filter think blocks from output")
    parser.add_argument("--suffix", default="", help="Suffix to append to prompt (e.g. ' /nothink')")
    parser.add_argument("--raw-only", action="store_true", help="Show raw output only, skip processing")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS

    for i, model in enumerate(models):
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Model: {model}")
        print(sep)
        try:
            raw, processed = run_model(
                model=model,
                prompt=args.prompt,
                system=args.system,
                think=args.think,
                strip_think=not args.no_strip,
                suffix_prompt=args.suffix,
            )
            print(f"\n  [Raw] {assess(raw)}")
            if not args.raw_only and processed != raw:
                print(f"  [After strip] {assess(processed)}")
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise

    print()


if __name__ == "__main__":
    main()
