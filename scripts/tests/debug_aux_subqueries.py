#!/usr/bin/env python3
"""
debug_aux_subqueries.py -- Standalone replica of generar_queries_con_llm for inspection.

Runs the same prompt and ``ollama.generate`` options as ``rag/chat_pdfs.py`` and prints
the **raw** model output to the terminal, plus any separate ``thinking`` field returned
by Ollama (reasoning trace). Use this to verify whether an auxiliary model (e.g.
``gemma4:e4b``) emits visible reasoning in the body or only in ``thinking``.

**Gemma 4 with ``think=True``:** reasoning goes into the API ``thinking`` field; if
``num_predict`` is too small (production uses 400), the visible ``response`` can be
**empty** because the budget is spent on thinking. Use ``--num-predict 2048`` (or
higher) only when debugging ``think=True``. Production keeps ``think=False`` (see
``generar_queries_con_llm``).

Models without thinking (e.g. ``gemma3:4b``) return HTTP 400 if ``think=True``;
``--both`` skips the second run with a short message when that happens.

Usage:
    python scripts/tests/debug_aux_subqueries.py
    python scripts/tests/debug_aux_subqueries.py "¿Cómo funciona un transformador?"
    python scripts/tests/debug_aux_subqueries.py --think
    python scripts/tests/debug_aux_subqueries.py --stream
    set OLLAMA_CHAT_MODEL=gemma4:e4b && python scripts/tests/debug_aux_subqueries.py

Dependencies:
    - ollama (Python package)
    - Ollama server running with the chosen model pulled
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Prompt builder (mirrors chat_pdfs.generar_queries_con_llm)
#
# HELPERS
# +-- 2. Response normalization, parse queries
#
# ENTRY
# +-- 3. main()
#
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Tuple

import ollama
from ollama import ResponseError

# ─────────────────────────────────────────────
# SECTION 1: PROMPT (same as chat_pdfs.generar_queries_con_llm)
# ─────────────────────────────────────────────


def build_subquery_prompt(pregunta: str) -> str:
    """Build the exact auxiliary prompt used in production."""
    return (
        "Generate exactly 3 search queries to retrieve relevant content "
        "from academic documents about the question below.\n\n"
        "Requirements:\n"
        "- Each query must target a DIFFERENT semantic aspect of the question\n"
        "- Write every query in the EXACT SAME LANGUAGE as the question\n"
        "- Output ONLY the 3 queries, one per line\n"
        "- No numbering, no bullets, no labels, no explanations\n\n"
        f"Question: {pregunta}\n\n"
        "Queries:\n"
    )


GENERATE_OPTIONS = {
    "temperature": 0.5,
    "num_predict": 400,
    "stop": ["\n\n\n"],
}


# ─────────────────────────────────────────────
# SECTION 2: HELPERS
# ─────────────────────────────────────────────


def split_response_thinking(result: Any) -> Tuple[str, Optional[str]]:
    """Return (response_text, thinking_text) from generate result (object or dict)."""
    if isinstance(result, dict):
        text = (result.get("response") or "").strip()
        think = result.get("thinking")
        if think is not None:
            think = str(think).strip() or None
        return text, think
    text = (getattr(result, "response", None) or "").strip()
    raw_think = getattr(result, "thinking", None)
    if raw_think is None:
        return text, None
    think = str(raw_think).strip() or None
    return text, think


def parse_queries_like_production(text: str) -> List[str]:
    """Same filtering as generar_queries_con_llm."""
    return [
        q.strip().lstrip("0123456789.-) ")
        for q in text.strip().split("\n")
        if q.strip() and len(q.strip()) > 20
    ][:3]


def run_once(
    model: str,
    prompt: str,
    think: bool,
    stream: bool,
    options: dict,
) -> bool:
    """Execute one generate call and print everything relevant.

    Returns:
        True if the call succeeded, False if the server rejected it (e.g. no thinking support).
    """
    print(f"\n{'=' * 72}")
    print(
        f"  model={model!r}  think={think!r}  stream={stream!r}  "
        f"num_predict={options.get('num_predict')!r}"
    )
    print("=" * 72)

    if not stream:
        try:
            result = ollama.generate(
                model=model,
                prompt=prompt,
                think=think,
                options=dict(options),
            )
        except ResponseError as e:
            print(f"\n  [SKIP/ERROR] Ollama: {e}")
            return False

        text, thinking = split_response_thinking(result)

        print("\n--- Field: thinking (API; empty means no separate trace) ---")
        if thinking:
            print(thinking)
        else:
            print("(empty / None)")

        print("\n--- Field: response (raw text shown to the pipeline) ---")
        print(text if text else "(empty)")
        print("\n--- repr(response) first 500 chars ---")
        print(repr(text[:500]) + ("..." if len(text) > 500 else ""))

        if thinking and not text:
            print(
                "\n*** NOTE: Non-empty ``thinking`` but empty ``response``. With "
                "``think=True``, Ollama often allocates ``num_predict`` to the trace first; "
                "400 tokens (production) may be insufficient for the final answer. "
                "Retry with e.g. ``--num-predict 2048`` to see the queries after the trace."
            )
        elif thinking and text and think:
            print(
                "\n*** NOTE: Both ``thinking`` and ``response`` are non-empty; compare "
                "lengths — short ``response`` may still be truncated if ``num_predict`` is low."
            )

        print("\n--- Parsed queries (production filter: len>20) ---")
        qs = parse_queries_like_production(text)
        if qs:
            for i, q in enumerate(qs, 1):
                print(f"  {i}. {q}")
        else:
            print("  (none passed filter)")

        # Inline tags that look like reasoning
        markers = ("<thinking>", "</thinking>", "<redacted_thinking>", "reasoning")
        lower = text.lower()
        found = [m for m in markers if m.lower() in lower or m in text]
        if found:
            print(f"\n--- WARNING: possible inline markers in response: {found}")
        else:
            print("\n--- No common <thinking> / redacted_thinking tags in response text.")
        return True

    # Streaming: show chunks as they arrive
    print("\n--- Stream (prefix T= thinking token, R= response token) ---\n")
    try:
        stream_iter = ollama.generate(
            model=model,
            prompt=prompt,
            think=think,
            options=dict(options),
            stream=True,
        )
    except ResponseError as e:
        print(f"\n  [SKIP/ERROR] Ollama: {e}")
        return False
    acc_response = ""
    acc_thinking = ""
    for chunk in stream_iter:
        cr, ct = split_response_thinking(chunk)
        if ct:
            acc_thinking += ct
            print(f"T:{ct!r}", end="", flush=True)
        if cr:
            acc_response += cr
            print(f"R:{cr!r}", end="", flush=True)
    print("\n")
    print("\n--- Accumulated thinking ---\n", acc_thinking or "(empty)")
    print("\n--- Accumulated response ---\n", acc_response or "(empty)")
    if acc_thinking and not acc_response.strip():
        print(
            "\n*** NOTE: Thinking stream consumed budget; try ``--num-predict 2048`` "
            "with ``think=True``."
        )
    return True


# ─────────────────────────────────────────────
# SECTION 3: ENTRY
# ─────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug auxiliary sub-query generation (same as generar_queries_con_llm).",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is the transformer architecture?",
        help="User question (default: English transformer question).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b"),
        help="Ollama model (default: env OLLAMA_CHAT_MODEL or gemma3:4b).",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Use think=True (reasoning enabled when the model supports it).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run twice: think=False then think=True for comparison.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens; prefix T= thinking, R= response per chunk.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=GENERATE_OPTIONS["num_predict"],
        metavar="N",
        help=(
            "Max new tokens (default: 400, same as production). "
            "Increase (e.g. 2048) when using --think on Gemma 4 so ``response`` is not empty."
        ),
    )
    args = parser.parse_args()

    prompt = build_subquery_prompt(args.question)

    gen_options = {**GENERATE_OPTIONS, "num_predict": args.num_predict}

    print(
        "\nDebug: auxiliary sub-queries (isolated from chat_pdfs.generar_queries_con_llm)\n"
        f"Question: {args.question!r}\n"
    )
    print("--- Full prompt sent to the model ---\n")
    print(prompt)
    print("--- end prompt ---")

    if args.both:
        run_once(args.model, prompt, think=False, stream=args.stream, options=gen_options)
        run_once(args.model, prompt, think=True, stream=args.stream, options=gen_options)
    else:
        run_once(
            args.model,
            prompt,
            think=args.think,
            stream=args.stream,
            options=gen_options,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
