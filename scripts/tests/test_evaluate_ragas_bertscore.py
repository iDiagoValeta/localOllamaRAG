import csv
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.evaluate_ragas_bertscore as bert_eval


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def install_fake_bertscore(monkeypatch, calls):
    module = types.ModuleType("bert_score")

    def fake_score(predictions, references, **kwargs):
        calls.append(
            {
                "predictions": list(predictions),
                "references": list(references),
                "kwargs": kwargs,
            }
        )
        return (
            FakeTensor([0.11 + i / 100 for i in range(len(predictions))]),
            FakeTensor([0.21 + i / 100 for i in range(len(predictions))]),
            FakeTensor([0.31 + i / 100 for i in range(len(predictions))]),
        )

    module.score = fake_score
    monkeypatch.setitem(sys.modules, "bert_score", module)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_single_csv_generates_bertscore_artifacts(monkeypatch, tmp_path):
    calls = []
    install_fake_bertscore(monkeypatch, calls)
    input_csv = tmp_path / "ragas_scores_es.csv"
    original_rows = [
        {
            "user_input": "Q1",
            "response": "Respuesta generada",
            "reference": "Respuesta de referencia",
            "answer_correctness": "0.5",
        },
        {
            "user_input": "Q2",
            "response": "",
            "reference": "Referencia vacia",
            "answer_correctness": "0.0",
        },
    ]
    write_csv(input_csv, original_rows)

    summary = bert_eval.evaluate_inputs(
        input_csv=input_csv,
        output_root=tmp_path / "out",
        label="mi_eval",
        config=bert_eval.BertScoreConfig(batch_size=2),
    )

    assert calls[0]["predictions"] == ["Respuesta generada"]
    assert calls[0]["references"] == ["Respuesta de referencia"]
    assert calls[0]["kwargs"]["model_type"] == "microsoft/deberta-xlarge-mnli"
    assert calls[0]["kwargs"]["lang"] == "en"
    assert calls[0]["kwargs"]["rescale_with_baseline"] is True
    assert calls[0]["kwargs"]["batch_size"] == 2

    output_csv = Path(summary["runs"][0]["output_csv"])
    rows = read_csv(output_csv)
    assert rows[0]["bertscore_precision"] == "0.110000"
    assert rows[0]["bertscore_recall"] == "0.210000"
    assert rows[0]["bertscore_f1"] == "0.310000"
    assert rows[0]["bertscore_model"] == "microsoft/deberta-xlarge-mnli"
    assert rows[0]["bertscore_rescale_with_baseline"] == "True"
    assert rows[1]["bertscore_f1"] == ""

    assert read_csv(input_csv) == original_rows
    payload = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
    assert payload["runs"][0]["rows_total"] == 2
    assert payload["runs"][0]["rows_scored"] == 1
    assert payload["runs"][0]["rows_skipped"] == 1


def test_comparison_dir_generates_summary_for_variants(monkeypatch, tmp_path):
    calls = []
    install_fake_bertscore(monkeypatch, calls)
    comparison_dir = tmp_path / "comparison"
    write_csv(
        comparison_dir / "baseline_all_on.csv",
        [{"response": "A", "reference": "B", "faithfulness": "1.0"}],
    )
    write_csv(
        comparison_dir / "no_reranker.csv",
        [{"response": "C", "reference": "D", "faithfulness": "0.5"}],
    )

    summary = bert_eval.evaluate_inputs(
        comparison_dir=comparison_dir,
        output_root=tmp_path / "out",
        config=bert_eval.BertScoreConfig(),
    )

    assert len(summary["runs"]) == 2
    assert len(calls) == 2
    output_dir = Path(summary["output_dir"])
    assert (output_dir / "baseline_all_on_bertscore.csv").exists()
    assert (output_dir / "no_reranker_bertscore.csv").exists()
    assert (output_dir / "bertscore_summary.json").exists()
    assert (output_dir / "bertscore_summary.csv").exists()


def test_all_completed_discovers_single_and_comparison_runs(monkeypatch, tmp_path):
    calls = []
    install_fake_bertscore(monkeypatch, calls)
    scores_root = tmp_path / "scores"
    write_csv(
        scores_root / "single" / "dataset_eval_es_es" / "scores.csv",
        [{"response": "A", "reference": "B", "answer_correctness": "0.8"}],
    )
    write_csv(
        scores_root / "comparisons" / "ablacion_es" / "scores" / "baseline_all_on.csv",
        [{"response": "C", "reference": "D", "faithfulness": "1.0"}],
    )
    write_csv(
        scores_root / "comparisons" / "ablacion_es" / "scores" / "resumen_por_conjunto.csv",
        [{"variante": "baseline_all_on", "conjunto": "wiki", "n": "1"}],
    )

    discovered = bert_eval.discover_completed_experiments(scores_root)
    assert [item["label"] for item in discovered] == ["single_dataset_eval_es_es", "ablacion_es"]

    summary = bert_eval.evaluate_all_completed(
        scores_root=scores_root,
        output_root=tmp_path / "out",
        config=bert_eval.BertScoreConfig(),
    )

    assert summary["experiments_count"] == 2
    assert len(calls) == 2
    assert (Path(summary["output_root"]) / "single_dataset_eval_es_es" / "scores_bertscore.csv").exists()
    assert (Path(summary["output_root"]) / "ablacion_es" / "baseline_all_on_bertscore.csv").exists()
    assert Path(summary["summary_json"]).exists()
    assert Path(summary["summary_csv"]).exists()
