"""Experiment record: one manifest.json that every stage merges its facts into.

Kept for publishing (model card, write-up): what ran, on what, with which
data, how long it took and what it measured. Data files are identified by
sha256, not copied.
"""
import datetime, hashlib, json, os, platform, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", os.path.expanduser("~/.local/lib/llama-b11669/llama-b11669"))
MANIFEST = os.path.join(ROOT, "runs", "manifest.json")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_facts(path):
    lines = sum(1 for _ in open(path, "rb")) if path.endswith(".jsonl") else None
    return {"path": os.path.relpath(path, ROOT), "bytes": os.path.getsize(path), "sha256": sha256(path), "lines": lines}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment():
    return {
        "recorded_at": now(),
        "host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "cpu_threads": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "ollama": sh("ollama -v"),
        "llama_cpp_build": sh(f"LD_LIBRARY_PATH={LLAMA_CPP_DIR} {LLAMA_CPP_DIR}/llama-server --version 2>&1 | head -1"),
        "monkeygrab_commit": sh(f"git -C {os.path.dirname(ROOT)} rev-parse --short HEAD"),
        "train_venv": sh(f"{ROOT}/.venv/bin/python -c \"import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)\""),
    }


def record(stage, **facts):
    """Merge ``facts`` under ``stage`` (dict-merge, lists replaced), stamped."""
    data = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    entry = data.setdefault(stage, {})
    entry.update(facts)
    entry["updated_at"] = now()
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    tmp = MANIFEST + ".tmp"
    json.dump(data, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST)
    return entry
