import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import repeticion_run_eval as repeticion


def test_ablation_suite_has_all_off_variant():
    variants = repeticion.seleccionar_variantes("ablation")
    names = [variant["name"] for variant in variants]

    assert len(variants) == 9
    assert names[0] == "baseline_all_on"
    assert names[-1] == "baseline_all_off"
    assert all(value is False for value in variants[-1]["flags"].values())


def test_repetition_all_preindexes_all_requested_corpora_and_defers_heldout():
    plan = repeticion._planificar_corpus_repeticion("all")

    assert [spec["corpus_key"] for spec in plan] == [
        "es",
        "ca",
        "ragbench_dev",
        "ragbench_eval",
        "ragbench_visual",
    ]
    assert [spec["should_infer"] for spec in plan] == [True, True, True, False, False]
    assert {spec["deferred_reason"] for spec in plan[-2:]} == {"waiting_for_final_variant"}


def test_repetition_final_variant_enables_heldout_single_variant():
    plan = repeticion._planificar_corpus_repeticion(
        "ragbench_eval",
        final_variant="no_query_decomposition",
    )

    assert len(plan) == 1
    assert plan[0]["should_infer"] is True
    assert [variant["name"] for variant in plan[0]["variants"]] == ["no_query_decomposition"]
