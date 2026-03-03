import json, subprocess, tempfile, time
from pathlib import Path
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "Qwen3-FineTuned:latest"
MODEL_TEST = "Qwen3-NoThink-Test:latest"
SYSTEM = "You are a helpful assistant."
QUESTION = "What is 2+2? Answer briefly."

# Qwen3 chat template that respects the 'think' flag from Ollama
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


def sep(label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print("=" * 70)


def stream_generate(payload, timeout=30):
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
    if "<think>" not in text:
        return "OK - SIN THINK"
    inner = text[text.find("<think>") + 7:]
    closing = inner.find("</think>")
    if closing != -1 and len(inner[:closing].strip()) == 0:
        return "OK - think vacio"
    return "FALLO - RAZONA"


# A: raw + think block vacio pre-rellenado
sep("A) raw:true + <think></think> pre-rellenado")
prompt_a = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n")
resp_a = stream_generate({"model": MODEL, "prompt": prompt_a, "raw": True,
    "stream": True, "options": {"temperature": 0.1, "num_ctx": 2048,
    "stop": ["<|im_end|>", "<|im_start|>"]}})
print(f"  -> {assess(resp_a)}")
time.sleep(1)

# B: raw + /no_think en usuario
sep("B) raw:true + /no_think en usuario")
prompt_b = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n/no_think\n{QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n")
resp_b = stream_generate({"model": MODEL, "prompt": prompt_b, "raw": True,
    "stream": True, "options": {"temperature": 0.1, "num_ctx": 2048,
    "stop": ["<|im_end|>", "<|im_start|>"]}})
print(f"  -> {assess(resp_b)}")
time.sleep(1)

# C: Modelfile derivado con plantilla Qwen3 nativa
sep("C) Modelfile derivado con plantilla Qwen3 nativa")
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
    print("OK. Probando think:false ...")
    resp_c_no = stream_generate({"model": MODEL_TEST, "system": SYSTEM,
        "prompt": QUESTION, "stream": True, "think": False,
        "options": {"temperature": 0.1, "num_ctx": 2048}})
    print(f"  -> {assess(resp_c_no)}")
    time.sleep(1)
    print("Probando think:true ...")
    resp_c_yes = stream_generate({"model": MODEL_TEST, "system": SYSTEM,
        "prompt": QUESTION, "stream": True, "think": True,
        "options": {"temperature": 0.1, "num_ctx": 2048}})
    print(f"  -> {assess(resp_c_yes)}")
    subprocess.run(["ollama", "rm", MODEL_TEST], capture_output=True)
    print("Modelo temporal eliminado.")
Path(mf_path).unlink(missing_ok=True)

print("\n" + "=" * 70)
print("  RESUMEN")
print("=" * 70)
for lbl, r in [("A raw+think_vacio", resp_a), ("B raw+no_think", resp_b),
               ("C modelfile think:false", resp_c_no),
               ("C modelfile think:true", resp_c_yes)]:
    print(f"  {assess(r):25s}  {lbl}")
print()
