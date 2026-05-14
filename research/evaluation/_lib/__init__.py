"""Shared helpers for the MonkeyGrab evaluation scripts.

This package is imported by ``index.py``, ``infer.py`` and ``evaluate.py``.
"""

from . import (
    aggregation,
    checkpoints,
    datasets,
    inference,
    pipeline_flags,
    ragas_runner,
    ragbench,
)

from .datasets import (
    cargar_dataset,
    normalizar_columnas,
    resolver_ruta_dataset,
    SUPPORTED_CORPORA,
    EVAL_DIR,
    DATASETS_DIR,
    LOCAL_DATASETS_DIR,
    RAGBENCH_DATASETS_DIR,
    RAGBENCH_PREPARED_DIR,
    RUNS_DIR,
    RAGAS_RUNS_DIR,
)
from .pipeline_flags import (
    BASELINE_PIPELINE_FLAGS,
    ABLATION_VARIANTS,
    VARIANT_SUITES,
    RAGBENCH_FINAL_PIPELINE_FLAGS,
    seleccionar_variantes,
    listar_variantes,
)
from .ragas_runner import (
    METRIC_NAMES,
    METRIC_DISPLAY_NAMES,
    METRIC_DESCRIPTIONS,
    evaluar_respuestas_con_ragas,
)
from .checkpoints import (
    generation_from_checkpoint,
)

__all__ = [
    "aggregation",
    "datasets",
    "pipeline_flags",
    "checkpoints",
    "inference",
    "ragas_runner",
    "ragbench",
    "cargar_dataset",
    "normalizar_columnas",
    "resolver_ruta_dataset",
    "SUPPORTED_CORPORA",
    "EVAL_DIR",
    "DATASETS_DIR",
    "LOCAL_DATASETS_DIR",
    "RAGBENCH_DATASETS_DIR",
    "RAGBENCH_PREPARED_DIR",
    "RUNS_DIR",
    "RAGAS_RUNS_DIR",
    "BASELINE_PIPELINE_FLAGS",
    "ABLATION_VARIANTS",
    "VARIANT_SUITES",
    "RAGBENCH_FINAL_PIPELINE_FLAGS",
    "seleccionar_variantes",
    "listar_variantes",
    "METRIC_NAMES",
    "METRIC_DISPLAY_NAMES",
    "METRIC_DESCRIPTIONS",
    "evaluar_respuestas_con_ragas",
    "generation_from_checkpoint",
]
