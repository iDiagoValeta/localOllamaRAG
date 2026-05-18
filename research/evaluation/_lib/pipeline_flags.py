"""Pipeline-flag presets and variant selection for ablation comparisons.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Baseline and ablation variants
#  2. RagBench-specific presets
#  3. Selection helpers
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────
# SECTION 1: BASELINE AND ABLATION VARIANTS
# ─────────────────────────────────────────────

BASELINE_PIPELINE_FLAGS = {
    "USAR_LLM_QUERY_DECOMPOSITION": True,
    "USAR_BUSQUEDA_HIBRIDA": True,
    "USAR_BUSQUEDA_EXHAUSTIVA": True,
    "USAR_RERANKER": True,
    "EXPANDIR_CONTEXTO": True,
    "USAR_OPTIMIZACION_CONTEXTO": True,
    "USAR_RECOMP_SYNTHESIS": True,
}

ABLATION_VARIANTS = [
    {
        "name": "baseline_all_on",
        "description": "All inference-time optional stages enabled.",
        "flags": BASELINE_PIPELINE_FLAGS,
    },
    {
        "name": "no_query_decomposition",
        "description": "Disable LLM query decomposition.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_LLM_QUERY_DECOMPOSITION": False},
    },
    {
        "name": "no_lexical_search",
        "description": "Disable keyword/lexical Chroma search.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_BUSQUEDA_HIBRIDA": False},
    },
    {
        "name": "no_exhaustive_search",
        "description": "Disable exhaustive full-collection text scan.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_BUSQUEDA_EXHAUSTIVA": False},
    },
    {
        "name": "no_reranker",
        "description": "Disable Cross-Encoder reranking.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_RERANKER": False},
    },
    {
        "name": "no_context_expansion",
        "description": "Disable adjacent-chunk context expansion.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "EXPANDIR_CONTEXTO": False},
    },
    {
        "name": "no_context_optimization",
        "description": "Disable PDF artifact cleanup before generation.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_OPTIMIZACION_CONTEXTO": False},
    },
    {
        "name": "no_recomp_synthesis",
        "description": "Disable RECOMP/LLM context synthesis.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_RECOMP_SYNTHESIS": False},
    },
    {
        "name": "all_off",
        "description": "All optional inference-time stages disabled (semantic-only floor).",
        "flags": {k: False for k in BASELINE_PIPELINE_FLAGS},
    },
]

FINAL_COMPARISON_VARIANTS = [
    ABLATION_VARIANTS[0],
    ABLATION_VARIANTS[-1],
]

DEFAULT_VARIANT_SUITE = "final"

VARIANT_SUITES = {
    "final": FINAL_COMPARISON_VARIANTS,
    "ablation": ABLATION_VARIANTS,
}


# ─────────────────────────────────────────────
# SECTION 2: RAGBENCH-SPECIFIC PRESETS
# ─────────────────────────────────────────────

RAGBENCH_FINAL_PIPELINE_FLAGS = dict(BASELINE_PIPELINE_FLAGS)

# All flags on by default. Opt-out must be done explicitly (e.g. via a custom
# variant or by overriding flags in the caller). Previously this preset
# silently disabled USAR_LLM_QUERY_DECOMPOSITION and USAR_RERANKER.
RAGBENCH_VISUAL_PIPELINE_FLAGS = dict(BASELINE_PIPELINE_FLAGS)


# ─────────────────────────────────────────────
# SECTION 3: SELECTION HELPERS
# ─────────────────────────────────────────────

def seleccionar_variantes(suite: str, variant_names: str | None = None) -> list[dict[str, Any]]:
    """Resolve requested variant names into concrete pipeline-flag specs."""
    available = {variant["name"]: variant for variant in VARIANT_SUITES[suite]}
    aliases = {
        "all_on": "baseline_all_on",
        "baseline_all_off": "all_off",
    }
    if not variant_names:
        return list(available.values())

    selected = []
    unknown = []
    for raw_name in variant_names.split(","):
        name = aliases.get(raw_name.strip(), raw_name.strip())
        if not name:
            continue
        if name not in available:
            unknown.append(name)
        else:
            selected.append(available[name])

    if unknown:
        valid = ", ".join(available)
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}. Valid variants: {valid}")
    if not selected:
        raise ValueError("No variants selected.")
    return selected


def listar_variantes(suite: str = "ablation") -> None:
    """Print available variants for a suite."""
    print(f"Available variants for suite '{suite}':")
    for variant in VARIANT_SUITES[suite]:
        print(f"  {variant['name']}: {variant['description']}")


def normalizar_pipeline_flags(flags: dict[str, Any] | None) -> dict[str, bool]:
    """Return a stable boolean pipeline-flags dict for checkpoints/manifests."""
    if not flags:
        return {}
    return {str(k): bool(v) for k, v in sorted(flags.items())}
