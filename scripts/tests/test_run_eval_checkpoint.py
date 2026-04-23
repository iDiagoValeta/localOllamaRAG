import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.run_eval as run_eval
import rag.chat_pdfs as chat_pdfs


def test_old_checkpoint_statuses_are_reconstructed_from_answers():
    answers = ["ready", "", None]

    statuses = run_eval._normalizar_estados_preguntas(None, answers, 3)

    assert [s["status"] for s in statuses] == ["ok", "pending", "pending"]
    assert [s["question_number"] for s in statuses] == [1, 2, 3]


def test_failed_empty_status_is_pending_but_valid_answer_wins():
    answers = ["", "later answer"]
    statuses = [
        {"status": "failed", "reason": "timeout", "attempts": 2},
        {"status": "failed", "reason": "timeout", "attempts": 2},
    ]

    normalized = run_eval._normalizar_estados_preguntas(statuses, answers, 2)
    pending = run_eval._indices_pendientes_generacion(answers, normalized, 2)

    assert pending == [0]
    assert normalized[1]["status"] == "ok"
    assert normalized[1]["reason"] is None


def test_generation_failure_writes_diagnostic_checkpoint(monkeypatch):
    work_dir = ROOT / "evaluation" / "debug" / "test_run_eval_checkpoint_tmp"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    dataset_path = work_dir / "dataset.json"
    checkpoint_path = work_dir / "checkpoint.json"
    output_path = work_dir / "scores.csv"
    debug_path = work_dir / "debug.json"
    dataset_path.write_text(
        json.dumps([{"question": "Why does this fail?", "ground_truth": "Because."}]),
        encoding="utf-8",
    )

    monkeypatch.setenv("EVAL_OLLAMA_ATTEMPTS", "1")
    monkeypatch.setenv("EVAL_OLLAMA_TIMEOUT", "1")
    monkeypatch.setattr(run_eval, "_conectar_e_indexar", lambda **_: ("collection", 1))
    monkeypatch.setattr(run_eval, "evaluar_pregunta_rag", lambda *_: ("", []))
    monkeypatch.setattr(
        run_eval,
        "_diagnosticar_fallo_generacion",
        lambda *_: "sin_contexto",
    )

    with pytest.raises(SystemExit):
        run_eval.generar_respuestas_rag(
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            debug_path=str(debug_path),
            checkpoint_path=str(checkpoint_path),
            eval_corpus="es",
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["answers"] == [""]
    assert checkpoint["contexts_list"] == [[]]
    assert checkpoint["question_statuses"][0]["status"] == "failed"
    assert checkpoint["question_statuses"][0]["reason"] == "sin_contexto"
    shutil.rmtree(work_dir)


def test_eval_generation_filters_low_scoring_reranker_candidates_by_default(monkeypatch):
    low_score_fragment = {
        "id": "doc_pag1_chunk0",
        "doc": "Relevant evidence that should still be passed to generation.",
        "metadata": {"source": "doc.pdf", "page": 1},
        "score_final": 0.1,
        "score_reranker": 0.1,
    }

    monkeypatch.setattr(chat_pdfs, "USAR_RERANKER", True)
    monkeypatch.setattr(chat_pdfs, "EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK", False)
    monkeypatch.setattr(chat_pdfs, "EXPANDIR_CONTEXTO", False)
    monkeypatch.setattr(
        chat_pdfs,
        "realizar_busqueda_hibrida",
        lambda *_: ([low_score_fragment], 0.1, {}),
    )
    monkeypatch.setattr(
        chat_pdfs,
        "generar_respuesta_silenciosa",
        lambda pregunta, fragmentos: "generated answer",
    )

    answer, contexts = chat_pdfs.evaluar_pregunta_rag("What should be answered?", object())

    assert answer == ""
    assert contexts == []


def test_ragbench_eval_generation_keeps_low_scoring_reranker_candidates(monkeypatch):
    low_score_fragment = {
        "id": "doc_pag1_chunk0",
        "doc": "Relevant evidence that should still be passed to generation.",
        "metadata": {"source": "doc.pdf", "page": 1},
        "score_final": 0.1,
        "score_reranker": 0.1,
    }

    monkeypatch.setattr(chat_pdfs, "USAR_RERANKER", True)
    monkeypatch.setattr(chat_pdfs, "EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK", True)
    monkeypatch.setattr(chat_pdfs, "EXPANDIR_CONTEXTO", False)
    monkeypatch.setattr(
        chat_pdfs,
        "realizar_busqueda_hibrida",
        lambda *_: ([low_score_fragment], 0.1, {}),
    )
    monkeypatch.setattr(
        chat_pdfs,
        "generar_respuesta_silenciosa",
        lambda pregunta, fragmentos: "generated answer",
    )

    answer, contexts = chat_pdfs.evaluar_pregunta_rag("What should be answered?", object())

    assert answer == "generated answer"
    assert contexts == [low_score_fragment["doc"]]


def test_prepared_ragbench_dataset_detection():
    assert run_eval._es_dataset_ragbench(
        "evaluation/debug/ragbench_prepared/dataset_ragbench_text_10p_5q.json",
        "en",
    )
    assert not run_eval._es_dataset_ragbench("evaluation/datasets/dataset_eval_en.json", "en")


def test_prepare_ragbench_eval_excludes_frozen_dev_docs(monkeypatch):
    work_dir = ROOT / "evaluation" / "debug" / "test_ragbench_prepare_tmp"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    docs_dir = work_dir / "docs"
    manifest_path = work_dir / "manifest.json"
    excluded_path = work_dir / "excluded.json"
    excluded_path.write_text(json.dumps(["paper_b"]), encoding="utf-8")

    queries = {
        "q1": {"query": "Question A", "source": "text"},
        "q2": {"query": "Question B", "source": "text"},
        "q3": {"query": "Question C", "source": "text"},
    }
    qrels = {
        "q1": {"doc_id": "paper_a"},
        "q2": {"doc_id": "paper_b"},
        "q3": {"doc_id": "paper_c"},
    }
    answers = {"q1": "A", "q2": "B", "q3": "C"}
    pdf_urls = {"paper_a": "http://a", "paper_b": "http://b", "paper_c": "http://c"}

    monkeypatch.setattr(
        run_eval,
        "descargar_metadatos",
        lambda: (queries, qrels, answers, pdf_urls),
    )
    monkeypatch.setattr(
        run_eval,
        "descargar_pdfs",
        lambda selected_papers, pdf_urls, pdfs_dir, skip_existing=True: list(selected_papers),
    )

    manifest = run_eval.preparar_ragbench_eval_en(
        source="text",
        n_papers=2,
        max_q=1,
        docs_dir=str(docs_dir),
        manifest_path=str(manifest_path),
        excluded_doc_ids_path=str(excluded_path),
    )

    assert manifest["selected_papers"] == ["paper_a", "paper_c"]
    assert manifest["indexed_files"] == ["paper_a.pdf", "paper_c.pdf"]
    dataset_rows = json.loads(Path(manifest["dataset_path"]).read_text(encoding="utf-8"))
    assert [row["paper_id"] for row in dataset_rows] == ["paper_a", "paper_c"]
    shutil.rmtree(work_dir)


def test_incremental_file_filter_indexes_only_missing_docs(monkeypatch):
    added_calls: list[list[str] | None] = []

    class FakeCollection:
        def __init__(self):
            self._count = 10

        def count(self):
            return self._count

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_or_create_collection(self, name):
            return FakeCollection()

    monkeypatch.setattr(run_eval.chromadb, "PersistentClient", FakeClient)
    monkeypatch.setattr(run_eval.rag_runtime, "PATH_DB", "fake_db_path")
    monkeypatch.setattr(run_eval.rag_runtime, "COLLECTION_NAME", "fake_collection")
    monkeypatch.setattr(run_eval.rag_runtime, "CARPETA_DOCS", "fake_docs_dir")
    monkeypatch.setattr(run_eval.rag_runtime, "obtener_documentos_indexados", lambda collection: ["a.pdf"])

    def _fake_indexar_documentos(carpeta, collection, solo_archivos=None, silent=False, progress_callback=None):
        added_calls.append(list(solo_archivos) if solo_archivos is not None else None)
        collection._count += 7
        return 7

    monkeypatch.setattr(run_eval, "indexar_documentos", _fake_indexar_documentos)

    collection, total = run_eval._conectar_e_indexar(
        force_reindex=False,
        solo_archivos=["a.pdf", "b.pdf"],
        add_missing_from_filter=True,
    )

    assert isinstance(collection, FakeCollection)
    assert added_calls == [["b.pdf"]]
    assert total == 17


def test_ragbench_eval_parser_defaults():
    parser = run_eval._build_parser()
    args = parser.parse_args(["ragbench-eval"])

    assert args.ragas_max_workers == 5
    assert args.ragas_batch_size == 15
    assert args.manifest == run_eval.RAGBENCH_EVAL_MANIFEST_PATH
