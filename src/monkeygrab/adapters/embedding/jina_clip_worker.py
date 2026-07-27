"""jina_clip_worker.py -- persistent out-of-process host for jina-clip-v2.

Runs ONLY under the isolated `.venv-mineru` interpreter (torch 2.6.0+cu124,
transformers 4.57.6, sentence-transformers 5.6.1) -- jina-clip-v2's remote
code is written against transformers 4.x and fails to load under the
product's own transformers 5.1 environment (`Tensor.item() cannot be called
on meta tensors`, reproduced with lazy loading disabled and CPU forced).
Never imported by product code: `JinaClipEmbedder`
(`monkeygrab.adapters.embedding.jina_clip_embedder`) launches this file as a
subprocess with `.venv-mineru`'s python and talks to it over stdin/stdout,
exactly like MinerU is invoked as an external CLI -- same reasoning, the
dependency graphs collide and the product environment is not degraded to
accommodate one adapter.

Wire protocol, one JSON object per line, UTF-8, always flushed immediately
(the parent blocks on readline otherwise):

  stdin  -> {"id": <int>, "op": "text",  "text": <str>}
          | {"id": <int>, "op": "image", "path": <str>}
  stdout <- {"event": "ready", "dim": <int>}                    -- once, after model load
          | {"event": "fatal", "error": <str>}                  -- once, then exit(1)
          | {"id": <int>, "ok": true,  "vector": [<float>, ...]}
          | {"id": <int>, "ok": false, "error": <str>}

A per-request failure (bad op, missing file, backend error) replies with
`ok: false` and keeps the process alive -- one bad request must not take
down a worker that took ~29s to load. Only a startup failure (no CUDA, the
model itself refuses to load) is fatal to the whole process; the caller
must restart it explicitly rather than have it silently come back in a
degraded state.
"""

import json
import sys

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS
# ─────────────────────────────────────────────

_MODEL_NAME = "jinaai/jina-clip-v2"
# Matryoshka truncation: jina-clip-v2's native dimension is 1024; 512 is the
# value verified in the spike (loads, produces correctly-discriminating
# text<->image similarity) and is fixed here, not exposed as a knob --
# changing it means re-verifying discrimination quality, not a config edit.
_TRUNCATE_DIM = 512


def _emit(message: dict) -> None:
    """Write one line-JSON message to stdout and flush immediately.

    Without an explicit flush, stdout stays block-buffered when it's a pipe
    (not a TTY), and the parent process's readline blocks forever waiting
    for a message that is sitting in an unflushed buffer.
    """
    print(json.dumps(message), flush=True)


def _fatal(error: str) -> None:
    """Report a startup failure and terminate. Never returns."""
    _emit({"event": "fatal", "error": error})
    sys.exit(1)


# STARTUP


def _load_model():
    """Hard-check CUDA, then load jina-clip-v2. Exits the process on any failure.

    CUDA is required, not preferred: the spike measured ~100s/document on
    CPU, which makes indexing a real corpus infeasible. Degrading to CPU
    silently would trade a clear startup error for a pipeline that "works"
    but is unusable in practice -- the hard-fail policy this project uses
    everywhere applies here too.
    """
    try:
        import torch
    except ImportError as exc:
        _fatal(f"torch is not installed in this interpreter: {exc}")

    if not torch.cuda.is_available():
        _fatal(
            "CUDA is required to run jina-clip-v2 (the CPU path was measured at "
            "~100s/document, infeasible for indexing) but torch.cuda.is_available() "
            "is False in this worker's interpreter."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        _fatal(f"sentence-transformers is not installed in this interpreter: {exc}")

    try:
        model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True, device="cuda")
    except Exception as exc:
        _fatal(f"failed to load {_MODEL_NAME!r}: {exc}")

    _patch_declared_modalities(model)
    return model


def _patch_declared_modalities(model) -> None:
    """Work around a version-skew gap between jina-clip-v2's remote code and
    sentence-transformers 5.x's modality validation.

    sentence-transformers 5.x added a strict pre-flight check in `.encode()`
    that rejects any input whose type isn't in the module's declared
    `modalities` list. `InputModule.modalities` defaults to `["text"]` and
    jina-clip-v2's remote `custom_st.Transformer` (written before this
    feature existed) never overrides it -- so `.encode()` refuses images with
    "Modality 'image' is not supported", even though that same class's own
    `tokenize()`/`forward()` implementation (verified by reading its source)
    fully handles `PIL.Image.Image` inputs via `get_image_features`. This
    patches only the DECLARATION read by the pre-flight check, on the class
    object loaded into this process -- it does not modify the cached remote
    code file, and it does not change what the model computes.
    """
    inner_module = model[0]
    inner_module.__class__.modalities = property(lambda self: ["text", "image"])


# REQUEST HANDLING


def _handle_request(line: str, model) -> None:
    """Parse and answer one request line. Always replies; never raises."""
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        # No id could be recovered from unparsable input; the adapter treats
        # a response with id=None as a protocol desync and fails loudly.
        _emit({"id": None, "ok": False, "error": f"invalid JSON request: {exc}"})
        return

    request_id = request.get("id")
    op = request.get("op")

    try:
        if op == "text":
            text = request["text"]
            vector = model.encode(text, truncate_dim=_TRUNCATE_DIM, normalize_embeddings=True)
        elif op == "image":
            from PIL import Image

            with Image.open(request["path"]) as image:
                image = image.convert("RGB")
                vector = model.encode(
                    image, truncate_dim=_TRUNCATE_DIM, normalize_embeddings=True
                )
        else:
            raise ValueError(f"unknown op {op!r}")
        _emit({"id": request_id, "ok": True, "vector": vector.tolist()})
    except Exception as exc:
        _emit({"id": request_id, "ok": False, "error": str(exc)})


# MAIN LOOP


def main() -> None:
    model = _load_model()
    _emit({"event": "ready", "dim": _TRUNCATE_DIM})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        _handle_request(line, model)


if __name__ == "__main__":
    main()
