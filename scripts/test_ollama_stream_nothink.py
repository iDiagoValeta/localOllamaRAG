#!/usr/bin/env python3
"""
Script para probar modelos auxiliares de Ollama con streaming,
intentando que no razonen (sin bloques <think>).

Uso:
    python scripts/test_ollama_stream_nothink.py
    python scripts/test_ollama_stream_nothink.py --model qwen3:14b --prompt "Hola"
    python scripts/test_ollama_stream_nothink.py -m qwen3-base-direct -m qwen3:4b-instruct
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
    """Genera con streaming y devuelve el texto completo."""
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
    """Elimina bloques <think>...</think> de la salida."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_model(
    model: str,
    prompt: str,
    system: str,
    think: bool,
    strip_think: bool,
    suffix_prompt: str = "",
) -> tuple[str, str]:
    """Ejecuta un modelo y devuelve (raw, processed)."""
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
    """Evalúa si la salida contiene razonamiento."""
    if "<think>" not in text:
        return "OK - sin think"
    inner = text[text.find("<think>") + 7 :]
    closing = inner.find("</think>")
    if closing != -1 and len(inner[:closing].strip()) == 0:
        return "OK - think vacío"
    return "RAZONA"


def main():
    parser = argparse.ArgumentParser(description="Probar modelos Ollama con streaming sin razonamiento")
    parser.add_argument("-m", "--model", action="append", dest="models", help="Modelo(s) a probar")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Prompt de prueba")
    parser.add_argument("-s", "--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--think", action="store_true", help="Forzar think=true (por defecto false)")
    parser.add_argument("--no-strip", action="store_true", help="No filtrar bloques think en salida")
    parser.add_argument("--suffix", default="", help="Sufijo al prompt (ej: ' /nothink')")
    parser.add_argument("--raw-only", action="store_true", help="Solo mostrar raw, sin procesar")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS

    for i, model in enumerate(models):
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Modelo: {model}")
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
                print(f"  [Tras strip] {assess(processed)}")
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise

    print()


if __name__ == "__main__":
    main()
