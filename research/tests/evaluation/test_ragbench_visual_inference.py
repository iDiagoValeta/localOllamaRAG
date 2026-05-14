"""Tests for the RagBench visual preparation logic in _lib/ragbench.py.

After the refactor, ``run_ragbench_visual_inference.py`` is replaced by the
``infer.py visual`` subcommand which delegates to ``_lib.ragbench`` for paper
selection, dataset writing, and CSV/JSON export.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.evaluation._lib import ragbench as ragbench_lib


def test_parse_visual_sources_rejects_non_visual_sources():
    assert ragbench_lib.parse_visual_sources("text-image,text-table") == ["text-image", "text-table"]

    with pytest.raises(ValueError):
        ragbench_lib.parse_visual_sources("text")


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

    monkeypatch.setattr(ragbench_lib, "descargar_metadatos", lambda: (queries, qrels, answers, pdf_urls))
    monkeypatch.setattr(
        ragbench_lib,
        "descargar_pdfs",
        lambda selected_papers, pdf_urls, pdfs_dir: list(selected_papers),
    )

    manifest = ragbench_lib.preparar_ragbench_visual(
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


def test_visual_only_doc_rejects_excluded_dev_doc():
    with pytest.raises(SystemExit):
        ragbench_lib.seleccionar_papers_visuales(
            queries={"q1": {"query": "Q", "source": "text-image"}},
            qrels={"q1": {"doc_id": "paper_a"}},
            pdf_urls={"paper_a": "http://a"},
            sources=["text-image"],
            n_papers=1,
            excluded_doc_ids=["paper_a"],
            only_doc="paper_a",
        )


def test_exportar_resultados_inferencia_writes_csv_and_json(tmp_path):
    dataset_path = tmp_path / "dataset_ragbench_visual.json"
    dataset_rows = [
        {
            "question": "What does the table show?",
            "ground_truth": "Reference",
            "paper_id": "paper_a",
            "source_type": "text-table",
        }
    ]
    dataset_path.write_text(json.dumps(dataset_rows), encoding="utf-8")

    generation = {
        "dataset_path": str(dataset_path),
        "questions": ["What does the table show?"],
        "ground_truths": ["Reference"],
        "answers": ["Generated answer"],
        "contexts_list": [["Context chunk"]],
        "question_statuses": [{"status": "ok", "reason": None}],
    }
    manifest = {
        "sources": ["text-table"],
        "dataset_path": str(dataset_path),
        "docs_dir": str(tmp_path / "docs"),
        "indexed_files": ["paper_a.pdf"],
    }
    result_csv = tmp_path / "out" / "results.csv"
    result_json = tmp_path / "out" / "results.json"

    ragbench_lib.exportar_resultados_inferencia(generation, manifest, result_csv, result_json)

    assert result_csv.exists()
    assert result_json.exists()

    with result_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["paper_id"] == "paper_a"
    assert rows[0]["source_type"] == "text-table"
    assert rows[0]["answer"] == "Generated answer"

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["rows"][0]["contexts"] == json.dumps(["Context chunk"], ensure_ascii=False)
