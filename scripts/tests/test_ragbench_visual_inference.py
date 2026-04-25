import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.run_ragbench_visual_inference as visual


def test_parse_sources_rejects_non_visual_sources():
    assert visual.parse_sources("text-image,text-table") == ["text-image", "text-table"]

    with pytest.raises(ValueError):
        visual.parse_sources("text")


def test_prepare_visual_dataset_filters_sources_and_excludes_dev(monkeypatch, tmp_path):
    excluded_path = tmp_path / "excluded.json"
    excluded_path.write_text(json.dumps(["paper_b"]), encoding="utf-8")
    docs_dir = tmp_path / "docs"
    debug_dir = tmp_path / "debug"

    queries = {
        "q1": {"query": "Table question", "source": "text-table"},
        "q2": {"query": "Image question excluded by dev", "source": "text-image"},
        "q3": {"query": "Plain text question", "source": "text"},
        "q4": {"query": "Image question", "source": "text-image"},
    }
    qrels = {
        "q1": {"doc_id": "paper_a"},
        "q2": {"doc_id": "paper_b"},
        "q3": {"doc_id": "paper_c"},
        "q4": {"doc_id": "paper_d"},
    }
    answers = {"q1": "A1", "q2": "A2", "q3": "A3", "q4": "A4"}
    pdf_urls = {
        "paper_a": "http://a",
        "paper_b": "http://b",
        "paper_c": "http://c",
        "paper_d": "http://d",
    }

    monkeypatch.setattr(visual.run_eval, "descargar_metadatos", lambda: (queries, qrels, answers, pdf_urls))
    monkeypatch.setattr(
        visual.run_eval,
        "descargar_pdfs",
        lambda selected_papers, pdf_urls, pdfs_dir: list(selected_papers),
    )

    manifest = visual.preparar_ragbench_visual(
        sources=["text-image", "text-table"],
        n_papers=10,
        max_q=2,
        skip_download=False,
        docs_dir=docs_dir,
        debug_dir=debug_dir,
        excluded_doc_ids_path=excluded_path,
    )

    assert manifest["selected_papers"] == ["paper_a", "paper_d"]
    assert manifest["indexed_files"] == ["paper_a.pdf", "paper_d.pdf"]
    assert Path(manifest["docs_dir"]) == docs_dir.resolve()

    rows = json.loads(Path(manifest["dataset_path"]).read_text(encoding="utf-8"))
    assert [row["question"] for row in rows] == ["Table question", "Image question"]
    assert [row["source_type"] for row in rows] == ["text-table", "text-image"]
    assert {row["paper_id"] for row in rows} == {"paper_a", "paper_d"}


def test_prepare_visual_only_doc_rejects_excluded_dev_doc(tmp_path):
    with pytest.raises(SystemExit):
        visual.seleccionar_papers_visuales(
            queries={"q1": {"query": "Q", "source": "text-image"}},
            qrels={"q1": {"doc_id": "paper_a"}},
            pdf_urls={"paper_a": "http://a"},
            sources=["text-image"],
            n_papers=1,
            excluded_doc_ids=["paper_a"],
            only_doc="paper_a",
        )


def test_visual_inference_exports_results_without_ragas(monkeypatch, tmp_path):
    dataset_path = tmp_path / "debug" / "dataset_ragbench_visual_image_table_1p_1q.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_rows = [
        {
            "question": "What does the table show?",
            "ground_truth": "Reference",
            "paper_id": "paper_a",
            "source_type": "text-table",
        }
    ]
    dataset_path.write_text(json.dumps(dataset_rows), encoding="utf-8")
    docs_dir = tmp_path / "docs"
    result_dir = tmp_path / "results"
    debug_dir = tmp_path / "debug"
    manifest = {
        "sources": ["text-table"],
        "dataset_path": str(dataset_path),
        "docs_dir": str(docs_dir),
        "indexed_files": ["paper_a.pdf"],
    }
    calls = {}

    def fake_generar_respuestas_rag(**kwargs):
        calls.update(kwargs)
        return {
            "dataset_path": str(dataset_path.resolve()),
            "output_path": kwargs["output_path"],
            "debug_path": kwargs["debug_path"],
            "checkpoint_path": kwargs["checkpoint_path"],
            "questions": ["What does the table show?"],
            "ground_truths": ["Reference"],
            "answers": ["Generated answer"],
            "contexts_list": [["Context chunk"]],
            "question_statuses": [{"status": "ok", "reason": None}],
            "questions_count": 1,
            "indexed_fragments": 3,
            "recomp_enabled": True,
            "pipeline_flags": kwargs["pipeline_flags"],
            "eval_corpus": "ragbench",
            "docs_dir": kwargs["docs_dir"],
            "pipeline_seconds": 1.0,
            "tiene_ground_truth": True,
        }

    monkeypatch.setattr(visual.run_eval, "generar_respuestas_rag", fake_generar_respuestas_rag)
    monkeypatch.setattr(
        visual.run_eval,
        "evaluar_respuestas_con_ragas",
        lambda *args, **kwargs: pytest.fail("RAGAS must not be called"),
    )

    result = visual.ejecutar_inferencia_visual(
        manifest=manifest,
        result_dir=result_dir,
        debug_dir=debug_dir,
        verbose=True,
        force_reindex=True,
    )

    assert calls["eval_corpus"] == "ragbench"
    assert calls["docs_dir"] == str(docs_dir.resolve())
    assert calls["solo_archivos"] == ["paper_a.pdf"]
    assert calls["add_missing_from_filter"] is True
    assert calls["force_reindex"] is True
    assert calls["pipeline_flags"]["USAR_LLM_QUERY_DECOMPOSITION"] is False
    assert "en_ragbench_visual" not in calls["docs_dir"]

    csv_path = Path(result["output_csv"])
    json_path = Path(result["output_json"])
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["paper_id"] == "paper_a"
    assert rows[0]["source_type"] == "text-table"
    assert rows[0]["answer"] == "Generated answer"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["contexts"] == json.dumps(["Context chunk"], ensure_ascii=False)
