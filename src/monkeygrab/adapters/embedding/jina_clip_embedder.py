"""JinaClipEmbedder -- Embedder/ImageEmbedder adapter over an out-of-process jina-clip-v2 worker.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. CONSTANTS         truncation dim, default timeouts, worker script path
#  +-- 2. WORKER LIFECYCLE  lazy start, one persistent subprocess, hard-fail on death
#  +-- 3. ADAPTER           JinaClipEmbedder: Embedder.embed + ImageEmbedder.embed_image
#  +-- 4. VECTOR MATH       pure, unit-testable L2-normalize / normalized-sum combination
#
# ─────────────────────────────────────────────

License note (jina-clip-v2, CC BY-NC): this adapter loads a non-commercial-
licensed model. That is an accepted, documented decision for this project's
personal/portfolio use
(docs/design/2026-07-26-monkeygrab-v2.md, section 3) -- not something to
re-litigate in code; a deployment under a different license needs a
different embedding model, not a change to this file.

Why out-of-process at all: jina-clip-v2's remote code targets transformers
4.x and fails to import under the product's transformers 5.1 (`Tensor.item()
cannot be called on meta tensors`, reproduced with lazy loading disabled and
CPU forced). It DOES load in the isolated `.venv-mineru` environment (torch
2.6.0+cu124, transformers 4.57.6, sentence-transformers 5.6.1). Rather than
downgrade the product's pinned stack to accommodate one model, this adapter
runs the model in that isolated interpreter as a persistent subprocess
(`jina_clip_worker.py`, same directory) and speaks line-JSON over its
stdin/stdout -- the same design MinerU uses as an external CLI, for the same
reason: colliding dependency pins, isolated by process boundary instead of
by degrading either environment. This file itself imports nothing but the
standard library, so it (and its unit tests) run in the product environment
without torch, CUDA, or the model ever being touched.
"""

import itertools
import json
import math
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

# Not subclassed from monkeygrab.ports.embedder.Embedder /
# monkeygrab.ports.image_embedder.ImageEmbedder: Protocol conformance here is
# structural (duck typing), the same contract every other adapter in this
# package satisfies without inheriting its port.

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS
# ─────────────────────────────────────────────

_TRUNCATE_DIM = 512  # must match jina_clip_worker.py's _TRUNCATE_DIM
# Cold model load was measured at ~29s in the spike; generous margin for a
# slower disk/first-run cache warm-up without masking a genuinely hung worker.
_DEFAULT_STARTUP_TIMEOUT_S = 180.0
_DEFAULT_REQUEST_TIMEOUT_S = 60.0
_WORKER_SCRIPT = str(Path(__file__).with_name("jina_clip_worker.py"))

# Sentinel distinct from any real message, used by the stdout-reading thread
# to signal "the pipe closed" (worker died) without waiting for a timeout.
_EOF = object()


# ─────────────────────────────────────────────
# SECTION 2: WORKER LIFECYCLE
# ─────────────────────────────────────────────


class JinaClipEmbedder:
    """Embeds text and images via a persistent jina-clip-v2 worker subprocess.

    The worker is started lazily on the first ``embed``/``embed_image`` call
    (never at construction time -- building this adapter, e.g. in a test
    with ``subprocess.Popen`` doubled, never launches a real process) and
    reused across every subsequent call: loading the model costs ~29s, so a
    process per call would make indexing a real corpus infeasible. Call
    ``close()`` when done to terminate the worker; it is also attempted
    best-effort on garbage collection, but an explicit ``close()`` is the
    caller's responsibility, matching every other resource in this project.

    Truncation is fixed at 512 dimensions (Matryoshka), the value verified
    in the spike; not a constructor parameter.

    Tables are never routed through ``embed_image``: they are indexed as
    their HTML text via ``embed`` instead (docs/design/2026-07-26-
    monkeygrab-v2.md, section "F3") -- that routing decision belongs to the
    caller (the multimodal indexing use case), not this adapter.

    Failure policy: hard-fail, with one deliberate exception to "restart
    transparently" -- once the worker process is observed dead (either
    mid-request, via a closed stdout pipe, or lazily on the next call), this
    adapter never respawns it silently. A crash loop (e.g. repeated CUDA
    OOM) must surface as a repeated, visible failure, not disappear behind
    what looks like uninterrupted operation. A caller that wants a fresh
    worker constructs a new ``JinaClipEmbedder``.
    """

    def __init__(
        self,
        python_executable: str,
        *,
        worker_script: str = _WORKER_SCRIPT,
        startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
    ):
        """Args:
            python_executable: Path to the isolated environment's python
                (e.g. ``.venv-mineru/Scripts/python.exe``) that has jina-
                clip-v2's dependencies installed. Required, not defaulted --
                that path is machine-specific and this project never
                captures machine-specific config in a default argument.
            worker_script: Path to ``jina_clip_worker.py``. Defaults to the
                copy shipped next to this file; overridable for tests.
            startup_timeout_s: Seconds to wait for the worker's "ready"
                message before giving up on the cold model load.
            request_timeout_s: Seconds to wait for a response to any single
                embed request before treating the worker as hung.
        """
        self._python_executable = python_executable
        self._worker_script = worker_script
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s

        self._process: Optional[subprocess.Popen] = None
        self._responses: "queue.Queue" = queue.Queue()
        self._stderr_tail: List[str] = []
        self._next_id = itertools.count(1)
        self._dead_reason: Optional[str] = None

    # --- lifecycle ------------------------------------------------------

    def _ensure_worker(self) -> subprocess.Popen:
        if self._process is not None:
            if self._process.poll() is None:
                return self._process
            # The process object exists but has already exited -- record it
            # as dead instead of silently spawning a replacement (see class
            # docstring: a crash loop must stay visible).
            exit_code = self._process.poll()
            self._process = None
            self._dead_reason = (
                f"jina-clip worker exited (code {exit_code}) between calls; "
                f"stderr: {self._format_stderr()}"
            )

        if self._dead_reason is not None:
            raise RuntimeError(f"jina-clip worker is no longer usable: {self._dead_reason}")

        self._start_worker()
        return self._process

    def _start_worker(self) -> None:
        try:
            process = subprocess.Popen(
                [self._python_executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(
                f"could not start the jina-clip worker with interpreter "
                f"{self._python_executable!r}: {exc}"
            ) from exc

        self._process = process
        self._responses = queue.Queue()
        self._stderr_tail = []
        threading.Thread(target=self._pump_stdout, args=(process,), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(process,), daemon=True).start()

        ready = self._await_message(self._startup_timeout_s, phase="starting up")

        if ready.get("event") == "fatal":
            self._fail_dead(f"jina-clip worker failed to start: {ready.get('error')}")
        if ready.get("event") != "ready":
            self._fail_dead(f"jina-clip worker sent an unexpected startup message: {ready!r}")
        if ready.get("dim") != _TRUNCATE_DIM:
            self._fail_dead(
                f"jina-clip worker reported dim={ready.get('dim')!r}, expected "
                f"{_TRUNCATE_DIM} (truncate_dim mismatch between adapter and worker)"
            )

    def _pump_stdout(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    message = {"event": "fatal", "error": f"worker sent a non-JSON line: {line!r}"}
                self._responses.put(message)
        finally:
            # The pipe closed (worker exited). Unblock whoever is waiting
            # immediately instead of making them sit through the full
            # timeout to discover the same thing.
            self._responses.put(_EOF)

    def _pump_stderr(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stderr:
                self._stderr_tail.append(line.rstrip())
                del self._stderr_tail[:-20]  # keep only the tail for the eventual error message
        except Exception:
            pass

    def _format_stderr(self) -> str:
        return " | ".join(self._stderr_tail) if self._stderr_tail else "(no stderr captured)"

    def _await_message(self, timeout_s: float, *, phase: str) -> dict:
        try:
            item = self._responses.get(timeout=timeout_s)
        except queue.Empty:
            self._fail_dead(
                f"jina-clip worker did not respond within {timeout_s}s while {phase}"
            )
        if item is _EOF:
            exit_code = self._process.poll() if self._process is not None else None
            self._fail_dead(
                f"jina-clip worker exited unexpectedly while {phase} "
                f"(exit code {exit_code}); stderr: {self._format_stderr()}"
            )
        return item

    def _fail_dead(self, reason: str) -> None:
        """Mark the worker permanently unusable and raise. Never returns."""
        self._dead_reason = reason
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
        raise RuntimeError(reason)

    def close(self) -> None:
        """Terminate the worker process, if one is running. Safe to call more than once."""
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.close()
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def __del__(self):
        # Best-effort only: an orphaned GPU process is worse than a slightly
        # late cleanup, but callers should still call close() explicitly.
        try:
            self.close()
        except Exception:
            pass

    # --- request/response -------------------------------------------------

    def _request(self, op: str, payload: Dict[str, str]) -> List[float]:
        self._ensure_worker()
        request_id = next(self._next_id)
        message = {"id": request_id, "op": op, **payload}

        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail_dead(f"jina-clip worker's stdin pipe is broken: {exc}")

        response = self._await_message(self._request_timeout_s, phase=f"handling op={op!r}")

        if "event" in response:
            # A startup-shaped message ("fatal", or a non-JSON line the reader
            # thread couldn't parse) arriving mid-request is a protocol
            # violation, not a normal id-keyed response -- report it as such
            # instead of falling through to a confusing id-mismatch error.
            self._fail_dead(
                f"jina-clip worker protocol violation while handling op={op!r}: "
                f"{response.get('error', response)}"
            )
        if response.get("id") != request_id:
            self._fail_dead(
                f"jina-clip worker response id {response.get('id')!r} does not match "
                f"request id {request_id!r} -- protocol desynchronized"
            )
        if not response.get("ok"):
            raise RuntimeError(
                f"jina-clip worker reported an error for op={op!r}: {response.get('error')}"
            )

        vector = response.get("vector")
        if not isinstance(vector, list) or len(vector) != _TRUNCATE_DIM:
            got = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise RuntimeError(
                f"jina-clip worker returned a vector of unexpected shape for op={op!r}: "
                f"expected {_TRUNCATE_DIM} floats, got {got}"
            )
        return vector

    # ─────────────────────────────────────────────
    # SECTION 3: ADAPTER
    # ─────────────────────────────────────────────

    def embed(self, text: str, *, keep_alive: Optional[int] = None) -> List[float]:
        """Embed ``text`` via the out-of-process jina-clip-v2 worker.

        Satisfies the ``Embedder`` port. ``keep_alive`` is accepted for
        signature compatibility with that port but has no effect here: VRAM
        residency is governed by the worker process's own lifetime (see
        ``close``), not a per-call parameter like Ollama's -- the worker
        keeps the model loaded across calls by design.

        Args:
            text: Text to embed.
            keep_alive: Unused; present only to satisfy the ``Embedder``
                port signature.

        Returns:
            The 512-dimensional embedding vector.

        Raises:
            RuntimeError: On any worker startup, protocol, or embedding failure.
        """
        return self._request("text", {"text": text})

    def embed_image(self, image_path: str, *, caption: str = "") -> List[float]:
        """Embed the image at ``image_path``, combined with ``caption`` if real.

        Satisfies the ``ImageEmbedder`` port. When ``caption`` is non-empty
        (after stripping whitespace -- a caption that is empty or only
        whitespace is treated exactly like "no caption", never sent to the
        model), the image and caption vectors are each L2-normalized, summed
        element-wise, and the SUM is re-normalized to unit length -- not
        averaged. Mathematically, normalizing a sum and normalizing an
        average point in the same direction (normalization removes the
        scalar factor), so the real risk an arithmetic mean introduces is
        someone returning `(v1 + v2) / 2` WITHOUT the final re-normalization
        step: that vector has sub-unit norm, and every image+caption chunk
        would then silently under-rank against the unit-normalized
        text-only vectors sharing the same FAISS inner-product index. The
        combination here always ends in a fresh L2 normalization specifically
        to rule that out.

        Args:
            image_path: Path to the image file on disk.
            caption: Caption text found near the image, if any.

        Returns:
            The 512-dimensional combined embedding vector.

        Raises:
            RuntimeError: If ``image_path`` does not exist, or on any
                worker startup, protocol, or embedding failure.
        """
        if not os.path.isfile(image_path):
            raise RuntimeError(f"jina-clip: image file not found: {image_path!r}")

        image_vector = _l2_normalize(self._request("image", {"path": image_path}))

        caption = caption.strip()
        if not caption:
            return image_vector

        # _request already enforces both vectors are exactly _TRUNCATE_DIM long,
        # so no separate dimension-mismatch check is needed here.
        caption_vector = _l2_normalize(self._request("text", {"text": caption}))
        combined = [a + b for a, b in zip(image_vector, caption_vector)]
        return _l2_normalize(combined)


# ─────────────────────────────────────────────
# SECTION 4: VECTOR MATH
# ─────────────────────────────────────────────


def _l2_normalize(vector: List[float]) -> List[float]:
    """Return ``vector`` scaled to unit L2 norm. Pure function, no I/O.

    Kept free of any worker/subprocess dependency so the image+caption
    combination rule is directly unit-testable without doubling anything.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise RuntimeError("jina-clip: cannot L2-normalize a zero vector")
    return [component / norm for component in vector]
