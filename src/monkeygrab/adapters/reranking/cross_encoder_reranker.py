"""CrossEncoderReranker -- Reranker adapter over sentence_transformers.CrossEncoder."""

import contextlib
import dataclasses
import io
import os
from typing import List, Optional, Sequence, Tuple

from sentence_transformers import CrossEncoder

from monkeygrab.config.models import ModelsConfig
from monkeygrab.domain.fragment import Fragment

# Not subclassed from monkeygrab.ports.reranker.Reranker: Protocol conformance
# here is structural (duck typing), the same contract every other adapter in
# this package satisfies without inheriting its port.

_MODEL_NAMES = {
    "quality": "BAAI/bge-reranker-v2-m3",
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def resolve_reranker_device(override: Optional[str] = None) -> Tuple[str, bool]:
    """Decide which device the Cross-Encoder should load on.

    An explicit argument wins, then the ``RERANKER_DEVICE`` environment
    variable, then CUDA availability. The second return value reports whether
    the choice was pinned by one of the first two: a pinned device must fail
    loudly if it cannot be used, while an auto-detected GPU may fall back to
    CPU.

    Args:
        override: Force ``"cpu"`` or ``"cuda"``, bypassing the environment
            and auto-detection.

    Returns:
        Tuple of (device, whether the choice was forced).
    """
    if override is not None:
        return override, True

    env_override = os.getenv("RERANKER_DEVICE", "").strip().lower()
    if env_override in ("cpu", "cuda"):
        return env_override, True

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", False
    except ImportError:
        pass
    return "cpu", False


class CrossEncoderReranker:
    """Re-scores retrieval candidates with a Cross-Encoder loaded once, lazily.

    Model variant comes from ``ModelsConfig.reranker_quality`` (``"quality"``
    -> BAAI/bge-reranker-v2-m3, ``"fast"`` -> cross-encoder/ms-marco-MiniLM-
    L-6-v2). The model is loaded on first ``rerank()``
    call rather than at construction time, so building this adapter (e.g. in
    a test with a stubbed ``sentence_transformers.CrossEncoder``) never
    touches a GPU or downloads weights until it is actually asked to score
    something.

    Device selection honours the ``RERANKER_DEVICE`` environment variable;
    the optional ``device`` constructor argument takes precedence over it,
    for callers and tests that want to force a device without touching the
    environment.

    Failure policy: hard-fail, with the explicit two-step device fallback the
    ``Reranker`` port docstring allows -- "if GPU-with-CPU-fallback is still
    wanted operationally, it is the adapter's own explicit two-step logic".
    That fallback only applies when the device was auto-detected (GPU
    available, no override): if ``RERANKER_DEVICE``/``device`` pins a
    specific device, that device is used as-is and a load failure raises
    immediately. Either way, a load failure that survives the allowed
    fallback raises rather than disabling reranking behind the caller's
    back. A scoring-time failure -- the model loads but ``.rank()`` raises
    -- also raises, instead of returning the input untouched.
    """

    def __init__(self, models: ModelsConfig, device: Optional[str] = None):
        """Args:
            models: Model-role config; only ``models.reranker_quality`` is read.
            device: Force ``"cpu"`` or ``"cuda"``, bypassing ``RERANKER_DEVICE``
                and CUDA auto-detection. ``None`` (default) preserves today's
                behavior.
        """
        self._model_name = _MODEL_NAMES[models.reranker_quality]
        self._device_override = device
        self._model: Optional[CrossEncoder] = None
        self._loaded_device: Optional[str] = None

    def _resolve_device(self) -> Tuple[str, bool]:
        """Return ``(device, forced)``. ``forced`` disables the CUDA->CPU fallback."""
        return resolve_reranker_device(self._device_override)

    def _load(self, device: str) -> CrossEncoder:
        model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
        with contextlib.redirect_stderr(io.StringIO()):
            return CrossEncoder(self._model_name, device=device, model_kwargs=model_kwargs)

    def _ensure_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model

        device, forced = self._resolve_device()
        try:
            self._model = self._load(device)
            self._loaded_device = device
            return self._model
        except Exception as exc:
            if forced or device != "cuda":
                raise RuntimeError(
                    f"Cross-Encoder {self._model_name!r} failed to load on "
                    f"{device!r}: {exc}"
                ) from exc
            try:
                self._model = self._load("cpu")
                self._loaded_device = "cpu"
                return self._model
            except Exception as exc2:
                raise RuntimeError(
                    f"Cross-Encoder {self._model_name!r} failed to load on GPU "
                    f"({exc}) and the CPU fallback also failed: {exc2}"
                ) from exc2

    def release(self) -> None:
        """Drop the loaded model and return its GPU memory.

        A reranker that has scored its candidates is done, but the weights stay
        resident and on an 8 GB card that is enough to stop a generator from
        loading at all: Ollama does not fail fast when the memory is not there,
        it blocks until the request times out. Lazy loading means a later
        ``rerank`` simply loads again.

        Safe to call when nothing is loaded, and never raises: freeing memory
        must not be able to break the caller that was being helpful.
        """
        if self._model is None:
            return
        self._model = None
        self._loaded_device = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    def rerank(self, query: str, fragments: Sequence[Fragment], top_k: int) -> List[Fragment]:
        """Re-score ``fragments`` against ``query`` and keep the best ``top_k``.

        Args:
            query: User query text.
            fragments: Candidate fragments from prior retrieval stages.
            top_k: Maximum number of fragments to return.

        Returns:
            Up to ``top_k`` fragments, best-first, each with
            ``score_reranker``/``score_final`` set to the model's score.

        Raises:
            RuntimeError: If the model cannot be loaded or scoring fails.
        """
        if not fragments:
            return []

        model = self._ensure_model()

        # Contextual retrieval prepends a situational summary to each chunk,
        # separated by a literal `\n\n` sentinel. Rerank against the body
        # alone: the summary was written to help retrieval find the chunk,
        # and scoring it again double-counts that hint.
        texts = []
        for frag in fragments:
            text = frag.doc
            if "\\n\\n" in text:
                text = text.split("\\n\\n", 1)[-1]
            texts.append(text)

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ranks = model.rank(
                    query, texts, top_k=min(top_k, len(texts)), return_documents=False
                )
        except Exception as exc:
            raise RuntimeError(
                f"Cross-Encoder reranking failed for model {self._model_name!r} "
                f"on device {self._loaded_device!r}: {exc}"
            ) from exc

        return [
            dataclasses.replace(
                fragments[rank_info["corpus_id"]],
                score_reranker=float(rank_info["score"]),
                score_final=float(rank_info["score"]),
            )
            for rank_info in ranks
        ]
