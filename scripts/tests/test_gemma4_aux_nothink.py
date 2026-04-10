#!/usr/bin/env python3
"""
test_gemma4_aux_nothink.py -- Probe reasoning suppression for Gemma 4 (Ollama).

MonkeyGrab uses ``ollama.generate`` for the auxiliary chat model (sub-queries in
``generar_queries_con_llm``). Gemma 4 is reasoning-capable; this script checks
whether ``think: false`` at the API level (and optionally a Modelfile default)
keeps traces out of the final text and out of the ``thinking`` JSON field.

Strategies exercised:
  (1) ``/api/generate`` with ``think: false`` vs ``true`` — mirrors production path.
  (2) ``/api/chat`` with ``think: false`` vs ``true`` — useful if you switch aux to chat.
  (3) Printed Modelfile snippet: ``PARAMETER think false`` for a derived model.

Usage:
    python scripts/tests/test_gemma4_aux_nothink.py
    python scripts/tests/test_gemma4_aux_nothink.py --model gemma4:e4b
    set OLLAMA_GEMMA4_TEST_MODEL=gemma4:e4b && python scripts/tests/test_gemma4_aux_nothink.py

Dependencies:
    - requests
    - Ollama running locally with the target model pulled

See also:
    https://docs.ollama.com/capabilities/thinking
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Constants (base URL, default model, auxiliary-style prompt)
#
# HELPERS
# +-- 2. HTTP generate/chat, thinking heuristics, printing
#
# TEST RUNNER
# +-- 3. run_scenarios(), modelfile_snippet()
#
# ENTRY
# +-- 4. main()
#
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import requests

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS
# ─────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_GEMMA4_TEST_MODEL", "gemma4:e4b")

# Same shape as generar_queries_con_llm in rag/chat_pdfs.py (shortened question for tests).
AUXILIARY_STYLE_PROMPT = (
    "Generate exactly 3 search queries to retrieve relevant content "
    "from academic documents about the question below.\n\n"
    "Requirements:\n"
    "- Each query must target a DIFFERENT semantic aspect of the question\n"
    "- Write every query in the EXACT SAME LANGUAGE as the question\n"
    "- Output ONLY the 3 queries, one per line\n"
    "- No numbering, no bullets, no labels, no explanations\n\n"
    "Question: What is the transformer architecture?\n\n"
    "Queries:\n"
)

SIMPLE_PROMPT = "What is 2+2? Answer with one number only."

GENERATE_OPTIONS = {
    "temperature": 0.5,
    "num_predict": 400,
    "stop": ["\n\n\n"],
}

# Tags sometimes leaked into text even when ``thinking`` is split server-side.
_TEXT_THINK_PATTERNS = (
    r"<redacted_thinking>.*?</redacted_thinking>",
    r"<thinking>.*?</thinking>",
    r"<reasoning>.*?</reasoning>",
)


# ─────────────────────────────────────────────
# SECTION 2: HELPERS
# ─────────────────────────────────────────────


def _post_json(path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """POST JSON to Ollama and return parsed object."""
    url = f"{OLLAMA_BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_generate(
    model: str,
    prompt: str,
    think: Optional[bool],
    stream: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call ``/api/generate``.

    Returns:
        Tuple of (response_text, thinking_text, raw_top_level_keys_sample).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": dict(GENERATE_OPTIONS),
    }
    if think is not None:
        payload["think"] = think

    if not stream:
        data = _post_json("/api/generate", payload)
        text = (data.get("response") or "").strip()
        think_txt = data.get("thinking")
        if think_txt is None:
            think_txt = ""
        else:
            think_txt = str(think_txt).strip()
        return text, think_txt, {k: type(v).__name__ for k, v in data.items()}

    # Streaming: accumulate response and thinking tokens from line-delimited JSON.
    full = ""
    think_parts: list[str] = []
    url = f"{OLLAMA_BASE_URL}/api/generate"
    with requests.post(url, json=payload, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                full += chunk["response"]
            t = chunk.get("thinking")
            if t:
                think_parts.append(t if isinstance(t, str) else str(t))
            if chunk.get("done"):
                break
    return full.strip(), "".join(think_parts).strip(), {"stream": True}


def api_chat(
    model: str,
    user_text: str,
    think: Optional[bool],
    stream: bool,
    system: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call ``/api/chat`` with a single user turn."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if think is not None:
        payload["think"] = think

    if not stream:
        data = _post_json("/api/chat", payload)
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        think_txt = msg.get("thinking")
        if think_txt is None:
            think_txt = ""
        else:
            think_txt = str(think_txt).strip()
        return text, think_txt, {k: type(v).__name__ for k, v in data.items()}

    full = ""
    think_parts: list[str] = []
    url = f"{OLLAMA_BASE_URL}/api/chat"
    with requests.post(url, json=payload, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message") or {}
            if msg.get("content"):
                full += msg["content"]
            t = msg.get("thinking")
            if t:
                think_parts.append(t if isinstance(t, str) else str(t))
            if chunk.get("done"):
                break
    return full.strip(), "".join(think_parts).strip(), {"stream": True}


def thinking_verdict(thinking_field: str, response_text: str) -> str:
    """Classify whether reasoning leaked (API field or inline tags)."""
    if len(thinking_field) > 20:
        return "FAIL — non-empty API ``thinking`` field"
    for pat in _TEXT_THINK_PATTERNS:
        if re.search(pat, response_text, flags=re.DOTALL | re.IGNORECASE):
            return "FAIL — thinking-like tags inside response text"
    if thinking_field.strip():
        return "WARN — short non-empty ``thinking`` (inspect manually)"
    return "OK — no substantive reasoning trace detected"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


# ─────────────────────────────────────────────
# SECTION 3: TEST RUNNER
# ─────────────────────────────────────────────


def run_scenarios(
    model: str,
    stream: bool,
    skip_chat: bool,
) -> None:
    """Run generate/chat matrix and print verdicts."""

    banner(f"Model: {model!r}  |  stream={stream}")

    # --- /api/generate (production-like for aux model) ---
    for label, think_flag in (
        ("generate  think=False", False),
        ("generate  think=True", True),
        ("generate  think omitted", None),
    ):
        banner(label)
        try:
            text, think_txt, keys = api_generate(model, SIMPLE_PROMPT, think_flag, stream)
            print(f"  response keys (sample): {keys}")
            print(f"  ``thinking`` length: {len(think_txt)} chars")
            if think_txt:
                preview = think_txt[:200].replace("\n", " ")
                print(f"  ``thinking`` preview: {preview!r}...")
            print(f"  response preview: {text[:280]!r}...")
            print(f"  --> {thinking_verdict(think_txt, text)}")
        except requests.RequestException as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    banner("generate  think=False  (auxiliary sub-query prompt)")
    try:
        text, think_txt, keys = api_generate(
            model, AUXILIARY_STYLE_PROMPT, False, stream
        )
        print(f"  response keys (sample): {keys}")
        print(f"  ``thinking`` length: {len(think_txt)} chars")
        print(f"  response:\n{text[:1200]}")
        print(f"  --> {thinking_verdict(think_txt, text)}")
    except requests.RequestException as e:
        print(f"  ERROR: {e}", file=sys.stderr)

    if skip_chat:
        print("\n  (--skip-chat: no /api/chat runs)\n")
        return

    # --- /api/chat (if you migrate aux to chat + think flag) ---
    for label, think_flag in (
        ("chat  think=False", False),
        ("chat  think=True", True),
    ):
        banner(label)
        try:
            text, think_txt, keys = api_chat(
                model,
                SIMPLE_PROMPT,
                think_flag,
                stream,
                system="You are a helpful assistant. Be concise.",
            )
            print(f"  top-level keys (sample): {keys}")
            print(f"  ``thinking`` length: {len(think_txt)} chars")
            print(f"  response preview: {text[:280]!r}...")
            print(f"  --> {thinking_verdict(think_txt, text)}")
        except requests.RequestException as e:
            print(f"  ERROR: {e}", file=sys.stderr)


def print_modelfile_snippet(model: str, derived_name: str = "gemma4-e4b-nothink") -> None:
    """Print a Modelfile users can ``ollama create`` for default think off."""
    banner("Optional: derived model with PARAMETER think false")
    print(
        f"""
Create a small wrapper (request-level ``think`` still overrides this):

  ollama pull {model}

Save as ``Modelfile``:

```
FROM {model}
PARAMETER think false
```

Then:

  ollama create {derived_name} -f Modelfile

Point ``OLLAMA_CHAT_MODEL={derived_name}`` if this works better than passing
``think`` on every call (Python client: check that ``ollama.generate`` forwards
``think=False`` for your installed ``ollama`` package version).
""".strip()
    )


# ─────────────────────────────────────────────
# SECTION 4: ENTRY
# ─────────────────────────────────────────────


def main() -> None:
    global OLLAMA_BASE_URL

    parser = argparse.ArgumentParser(
        description="Test Gemma 4 reasoning suppression for auxiliary-style Ollama calls."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL!r} or env OLLAMA_GEMMA4_TEST_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        default=OLLAMA_BASE_URL,
        help="Ollama base URL (default: OLLAMA_BASE_URL or http://localhost:11434).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode (default: non-stream for easier JSON inspection).",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip /api/chat scenarios (only test /api/generate).",
    )
    parser.add_argument(
        "--no-modelfile-hint",
        action="store_true",
        help="Do not print Modelfile snippet at the end.",
    )
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.base_url.rstrip("/")

    print(
        "\nGemma 4 aux reasoning probe — compares think=False / True / omitted on "
        "/api/generate (same family as generar_queries_con_llm).\n"
        "Docs: https://docs.ollama.com/capabilities/thinking\n"
    )

    run_scenarios(args.model, stream=args.stream, skip_chat=args.skip_chat)

    if not args.no_modelfile_hint:
        print_modelfile_snippet(args.model)

    print(
        "\nMonkeyGrab: ``generar_queries_con_llm`` already calls "
        "``ollama.generate(..., think=False)`` for ``OLLAMA_CHAT_MODEL``. "
        "Re-run this script after Ollama upgrades if behaviour regresses."
    )


if __name__ == "__main__":
    main()
