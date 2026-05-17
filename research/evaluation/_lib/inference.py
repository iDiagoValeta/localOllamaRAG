"""RAG inference orchestration: index, generate, persist checkpoints.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Ollama timeout-aware question execution
#  2. Failure diagnostics
#  3. ChromaDB connection + indexing wrapper
#  4. generar_respuestas_rag (main loop)
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path
from typing import Any

from .datasets import (
    artifact_suffix,
    cargar_dataset,
    default_debug_path,
    default_docs_dir_for_corpus,
    default_output_path,
    normalizar_columnas,
    resolver_ruta_dataset,
)
from .checkpoints import (
    cargar_checkpoint,
    checkpoint_models_match,
    checkpoint_pipeline_flags_match,
    current_models_signature,
    default_checkpoint_path,
    guardar_checkpoint_evaluacion,
    indices_pendientes_generacion,
    indices_respuestas_vacias,
    normalizar_estados_preguntas,
    respuesta_truncada,
    respuesta_vacia,
    resumen_estados_fallidos,
)

DEFAULT_EVAL_OLLAMA_TIMEOUT = 120


# ─────────────────────────────────────────────
# SECTION 1: OLLAMA TIMEOUT-AWARE EXECUTION
# ─────────────────────────────────────────────

def ejecutar_pregunta_con_timeout(
    pregunta: str,
    collection: Any,
    timeout_seconds: int,
) -> tuple[str, list[str], bool, Exception | None]:
    """Run one RAG question attempt with a best-effort wall-clock timeout."""
    import rag.chat_pdfs as rag_runtime

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(rag_runtime.evaluar_pregunta_rag, pregunta, collection)
    try:
        answer, contexts = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=True)
        return answer, contexts, False, None
    except _FuturesTimeout:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return "", [], True, None
    except Exception as exc:
        executor.shutdown(wait=True)
        return "", [], False, exc


def es_timeout_ollama(exc: Exception | None) -> bool:
    """Return True when an exception represents an Ollama/request timeout."""
    if exc is None:
        return False
    exc_name = exc.__class__.__name__.lower()
    exc_text = str(exc).lower()
    return (
        "timeout" in exc_name
        or "timed out" in exc_text
        or "read timed out" in exc_text
    )


def es_error_servidor_ollama(exc: Exception | None) -> bool:
    """Return True when an exception is an Ollama 500 server crash (runner died)."""
    if exc is None:
        return False
    text = str(exc)
    return "500" in text or "Internal Server Error" in text


# ─────────────────────────────────────────────
# SECTION 2: FAILURE DIAGNOSTICS
# ─────────────────────────────────────────────

def es_dataset_ragbench(dataset_path: str, eval_corpus: str) -> bool:
    """Return True for RagBench runs, including prepared datasets run as corpus en."""
    if eval_corpus == "ragbench":
        return True
    path = Path(dataset_path)
    parts = {part.lower() for part in path.parts}
    return "ragbench_prepared" in parts or "ragbench" in path.stem.lower()


def diagnosticar_fallo_generacion(
    pregunta: str,
    collection: Any,
    answer: str,
    contexts: list[str],
    timed_out: bool,
    error: Exception | None,
) -> str:
    """Classify why a question did not produce a usable answer."""
    import rag.chat_pdfs as rag_runtime

    if timed_out:
        return "timeout"
    if error is not None:
        return "excepcion"
    if not contexts:
        try:
            fragmentos_ranked, mejor_score, _ = rag_runtime.realizar_busqueda_hibrida(pregunta, collection)
            if not fragmentos_ranked:
                return "sin_contexto"
            fallback_ragbench = bool(
                getattr(rag_runtime, "EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK", False)
            )
            if rag_runtime.USAR_RERANKER and mejor_score < rag_runtime.UMBRAL_RELEVANCIA and not fallback_ragbench:
                return "filtrada_por_reranker"
            if rag_runtime.USAR_RERANKER:
                fragmentos_filtrados = [
                    f for f in fragmentos_ranked
                    if f.get("score_reranker", f.get("score_final", 0)) >= rag_runtime.UMBRAL_SCORE_RERANKER
                ]
                if not fragmentos_filtrados and not fallback_ragbench:
                    return "filtrada_por_reranker"
        except Exception:
            return "sin_contexto"
        return "sin_contexto"
    if respuesta_vacia(answer):
        return "respuesta_vacia"
    return "ok"


# ─────────────────────────────────────────────
# SECTION 3: CHROMADB CONNECTION + INDEXING
# ─────────────────────────────────────────────

def conectar_e_indexar(
    force_reindex: bool = False,
    solo_archivos: list[str] | None = None,
    add_missing_from_filter: bool = False,
) -> tuple[Any, int]:
    """Connect to the active Chroma collection and index documents if needed."""
    import chromadb
    import rag.chat_pdfs as rag_runtime
    from rag.chat_pdfs import indexar_documentos

    print(f"\nConnecting to ChromaDB: {rag_runtime.PATH_DB}")
    client = chromadb.PersistentClient(path=rag_runtime.PATH_DB)

    if force_reindex:
        print("   Force reindex requested. Rebuilding collection...")
        try:
            client.delete_collection(name=rag_runtime.COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
        total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection, solo_archivos=solo_archivos)
        print(f"   Indexed {total} fragments.")
        return collection, total

    collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
    if collection.count() == 0:
        print("   Database empty. Indexing documents...")
        total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection, solo_archivos=solo_archivos)
        print(f"   Indexed {total} fragments.")
    else:
        total = collection.count()
        print(f"   Fragments in collection: {total}")
        if solo_archivos and add_missing_from_filter:
            indexed_docs = set(rag_runtime.obtener_documentos_indexados(collection))
            missing_files = [f for f in solo_archivos if f not in indexed_docs]
            if missing_files:
                print(
                    "   Adding missing files to existing collection: "
                    + ", ".join(missing_files[:10])
                    + ("..." if len(missing_files) > 10 else "")
                )
                added = indexar_documentos(
                    rag_runtime.CARPETA_DOCS,
                    collection,
                    solo_archivos=missing_files,
                )
                total = collection.count()
                print(f"   Indexed {added} new fragments. Collection now has {total}.")
            else:
                print("   Collection already contains every file from the requested manifest.")
        elif solo_archivos:
            print("   Note: file filter only applies while indexing an empty/rebuilt collection.")
    return collection, total


# ─────────────────────────────────────────────
# SECTION 4: GENERAR_RESPUESTAS_RAG
# ─────────────────────────────────────────────

def generar_respuestas_rag(
    dataset_path: str,
    output_path: str | None = None,
    debug_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    force_reindex: bool = False,
    recomp_enabled: bool | None = None,
    pipeline_flags: dict[str, bool] | None = None,
    eval_corpus: str = "es",
    docs_dir: str | None = None,
    solo_archivos: list[str] | None = None,
    add_missing_from_filter: bool = False,
) -> dict[str, Any]:
    """Run only indexing/retrieval/generation and persist a reusable checkpoint."""
    os.environ.setdefault(
        "OLLAMA_REQUEST_TIMEOUT",
        os.getenv("EVAL_OLLAMA_TIMEOUT", str(DEFAULT_EVAL_OLLAMA_TIMEOUT)),
    )
    import rag.chat_pdfs as rag_runtime

    previous_pipeline_flags = None
    docs_previous: tuple[str, str, str] | None = None
    previous_ragbench_reranker_fallback: bool | None = None
    flag_overrides = dict(pipeline_flags or {})
    if recomp_enabled is not None:
        flag_overrides["USAR_RECOMP_SYNTHESIS"] = bool(recomp_enabled)
    if flag_overrides:
        previous_pipeline_flags = rag_runtime.set_pipeline_flags(flag_overrides)

    resolved_docs_dir = docs_dir or default_docs_dir_for_corpus(eval_corpus)
    if resolved_docs_dir is not None:
        docs_previous = rag_runtime.set_docs_folder_runtime(resolved_docs_dir)

    try:
        print("\nLoading dataset...")
        dataset_path = resolver_ruta_dataset(dataset_path)
        ragbench_reranker_fallback = es_dataset_ragbench(dataset_path, eval_corpus)
        previous_ragbench_reranker_fallback = (
            rag_runtime.set_ragbench_reranker_low_score_fallback(ragbench_reranker_fallback)
        )
        sfx = artifact_suffix(eval_corpus)
        resolved_output_path = os.path.abspath(output_path or default_output_path(dataset_path, sfx))
        resolved_debug_path = os.path.abspath(debug_path or default_debug_path(dataset_path, sfx))
        resolved_checkpoint_path = os.path.abspath(
            checkpoint_path
            or default_checkpoint_path(dataset_path, rag_runtime.USAR_RECOMP_SYNTHESIS, sfx)
        )

        df = normalizar_columnas(cargar_dataset(dataset_path))
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()
        tiene_ground_truth = any(gt.strip() for gt in ground_truths)

        print(f"   Questions to evaluate: {len(questions)}")
        print(f"   Ground truth available: {'Yes' if tiene_ground_truth else 'No'}")
        print(f"   RECOMP synthesis: {'Enabled' if rag_runtime.USAR_RECOMP_SYNTHESIS else 'Disabled'}")
        current_pipeline_flags = rag_runtime.get_pipeline_flags()
        print(
            "   Pipeline flags: "
            + ", ".join(
                f"{name}={'on' if value else 'off'}"
                for name, value in current_pipeline_flags.items()
            )
        )
        print(f"   Eval corpus: {eval_corpus.upper()} -- PDFs: {rag_runtime.CARPETA_DOCS}")
        if ragbench_reranker_fallback:
            print(
                "   RagBench reranker fallback: enabled "
                "(low-scored reranker candidates are kept for generation)."
            )

        collection, total = conectar_e_indexar(
            force_reindex=force_reindex,
            solo_archivos=solo_archivos,
            add_missing_from_filter=add_missing_from_filter,
        )

        print("\nRunning RAG pipeline for each question...")
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        question_statuses: list[dict[str, Any]] = []
        checkpoint = cargar_checkpoint(resolved_checkpoint_path)
        checkpoint_valid = False
        if checkpoint:
            models_match, models_note = checkpoint_models_match(
                checkpoint, current_models_signature()
            )
            structural_match = (
                checkpoint.get("dataset_path") == dataset_path
                and checkpoint.get("questions_count") == len(questions)
                and checkpoint_pipeline_flags_match(checkpoint, current_pipeline_flags)
                and checkpoint.get("eval_corpus", "es") == eval_corpus
                and checkpoint.get("docs_dir", rag_runtime.CARPETA_DOCS) == rag_runtime.CARPETA_DOCS
            )
            if structural_match and models_match:
                checkpoint_valid = True
                answers = checkpoint.get("answers", [])
                contexts_list = checkpoint.get("contexts_list", [])
                question_statuses = normalizar_estados_preguntas(
                    checkpoint.get("question_statuses"),
                    answers,
                    len(questions),
                )
                non_empty_answers = len([a for a in answers if not respuesta_vacia(a)])
                print(
                    "   Resuming from checkpoint: "
                    f"{non_empty_answers}/{len(questions)} questions with non-empty answers "
                    f"({len(answers)}/{len(questions)} slots present)."
                )
                if models_note:
                    print(f"   [aviso] {models_note}")
            else:
                if structural_match and not models_match:
                    print(
                        f"   Existing checkpoint does not match this run ({models_note}). "
                        "Starting fresh progress."
                    )
                else:
                    print("   Existing checkpoint does not match this run. Starting fresh progress.")

        t_start = time.time()
        if len(answers) > len(questions):
            answers = answers[:len(questions)]
        if len(contexts_list) > len(questions):
            contexts_list = contexts_list[:len(questions)]

        while len(answers) < len(questions):
            answers.append("")
        while len(contexts_list) < len(questions):
            contexts_list.append([])
        question_statuses = normalizar_estados_preguntas(
            question_statuses,
            answers,
            len(questions),
        )

        truncated_indexes = [
            i for i, (ans, st) in enumerate(zip(answers, question_statuses))
            if st.get("status") == "ok" and respuesta_truncada(ans)
        ]
        for i in truncated_indexes:
            question_statuses[i]["status"] = "failed"
            question_statuses[i]["reason"] = "respuesta_truncada"
        if truncated_indexes:
            listed = ", ".join(str(i + 1) for i in truncated_indexes[:20])
            suffix = "..." if len(truncated_indexes) > 20 else ""
            print(
                f"   Found {len(truncated_indexes)} truncated answer(s). "
                f"Will regenerate: {listed}{suffix}"
            )

        pending_answer_indexes = indices_pendientes_generacion(
            answers,
            question_statuses,
            len(questions),
        )
        if pending_answer_indexes and checkpoint_valid:
            first_missing = ", ".join(str(i + 1) for i in pending_answer_indexes[:10])
            suffix = "..." if len(pending_answer_indexes) > 10 else ""
            print(
                "   Checkpoint contains empty answers. "
                f"Regenerating {len(pending_answer_indexes)} question(s): {first_missing}{suffix}"
            )

        ollama_timeout = int(os.getenv("EVAL_OLLAMA_TIMEOUT", str(DEFAULT_EVAL_OLLAMA_TIMEOUT)))
        max_attempts = max(1, int(os.getenv("EVAL_OLLAMA_ATTEMPTS", "2")))
        try:
            for i in pending_answer_indexes:
                q = questions[i]
                if verbose:
                    print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
                answer, contexts = "", []
                timed_out = False
                last_error: Exception | None = None
                attempt_count = 0
                question_start = time.time()
                for attempt in range(max_attempts):
                    attempt_count = attempt + 1
                    if attempt > 0:
                        if es_error_servidor_ollama(last_error):
                            recovery_sleep = int(
                                os.getenv("EVAL_OLLAMA_RECOVERY_SLEEP", "60")
                            )
                            print(
                                f"   [RECOVERY] Q{i+1}: 500 server error — "
                                f"waiting {recovery_sleep}s for Ollama runner to restart..."
                            )
                            time.sleep(recovery_sleep)
                        print(
                            f"   [RETRY] Q{i+1} attempt {attempt + 1}/{max_attempts} "
                            f"(previous: empty or timeout)."
                        )
                    answer, contexts, timed_out, last_error = ejecutar_pregunta_con_timeout(
                        q, collection, ollama_timeout
                    )
                    if timed_out:
                        print(
                            f"   [TIMEOUT] Q{i+1} exceeded {ollama_timeout}s "
                            f"(attempt {attempt + 1}/{max_attempts})."
                        )
                        break
                    if last_error is not None:
                        print(
                            f"   [ERROR] Q{i+1} failed on attempt "
                            f"{attempt + 1}/{max_attempts}: {last_error}"
                        )
                        if es_timeout_ollama(last_error):
                            timed_out = True
                            print(
                                f"   [TIMEOUT] Q{i+1} hit Ollama/request timeout "
                                f"after {ollama_timeout}s. Moving to next question."
                            )
                            break
                    if not respuesta_vacia(answer) and not respuesta_truncada(answer):
                        break

                reason = diagnosticar_fallo_generacion(
                    q, collection, answer, contexts, timed_out, last_error,
                )
                is_bad = respuesta_vacia(answer) or respuesta_truncada(answer)
                if is_bad and max_attempts > 1 and not timed_out:
                    label = "empty" if respuesta_vacia(answer) else "truncated"
                    print(
                        f"   [WARN] Q{i+1} still {label} after {max_attempts} attempts; "
                        f"reason={reason}. Rerun to retry only incomplete answers."
                    )
                    if reason == "excepcion" and es_error_servidor_ollama(last_error):
                        cascade_sleep = int(
                            os.getenv("EVAL_OLLAMA_CASCADE_SLEEP", "30")
                        )
                        if cascade_sleep > 0:
                            print(
                                f"   [RECOVERY] Waiting {cascade_sleep}s before "
                                f"next question (Ollama post-crash cooldown)..."
                            )
                            time.sleep(cascade_sleep)
                final_reason = (
                    None if not is_bad
                    else (reason if respuesta_vacia(answer) else "respuesta_truncada")
                )
                answers[i] = answer
                contexts_list[i] = contexts
                question_statuses[i] = {
                    "index": i,
                    "question_number": i + 1,
                    "status": "ok" if not is_bad else "failed",
                    "attempts": attempt_count,
                    "duration_seconds": round(time.time() - question_start, 3),
                    "reason": final_reason,
                    "error": (
                        f"{type(last_error).__name__}: {last_error}"
                        if last_error is not None else None
                    ),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                guardar_checkpoint_evaluacion(
                    resolved_checkpoint_path,
                    dataset_path=dataset_path,
                    questions_count=len(questions),
                    recomp_enabled=rag_runtime.USAR_RECOMP_SYNTHESIS,
                    eval_corpus=eval_corpus,
                    output_path=resolved_output_path,
                    debug_path=resolved_debug_path,
                    answers=answers,
                    contexts_list=contexts_list,
                    question_statuses=question_statuses,
                    pipeline_flags=current_pipeline_flags,
                    docs_dir=rag_runtime.CARPETA_DOCS,
                    ragbench_reranker_low_score_fallback=ragbench_reranker_fallback,
                )
        except ConnectionError as e:
            print(f"\nError: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is running before launching the evaluation.")
            print("   Start Ollama with: ollama serve")
            raise SystemExit(1)

        t_rag = time.time() - t_start
        print(f"   Pipeline completed in {t_rag:.1f}s ({t_rag/len(questions):.1f}s/question)")

        empty_answer_indexes = indices_respuestas_vacias(answers, len(questions))
        if empty_answer_indexes:
            listed = ", ".join(str(i + 1) for i in empty_answer_indexes[:20])
            suffix = "..." if len(empty_answer_indexes) > 20 else ""
            failed_summary = resumen_estados_fallidos(answers, question_statuses, len(questions))
            print(
                "\nError: RAGAS evaluation was not launched because "
                f"{len(empty_answer_indexes)} answer(s) are empty."
            )
            print(f"   Empty question indexes: {listed}{suffix}")
            for reason, indexes in sorted(failed_summary.items()):
                reason_listed = ", ".join(str(n) for n in indexes[:20])
                reason_suffix = "..." if len(indexes) > 20 else ""
                print(f"   {reason}: {reason_listed}{reason_suffix}")
            print(f"   Checkpoint: {resolved_checkpoint_path}")
            print("   Fix the RAG generation issue and rerun; only empty answers will be retried.")
            raise SystemExit(1)

        return {
            "dataset_path": os.path.abspath(dataset_path),
            "output_path": resolved_output_path,
            "debug_path": resolved_debug_path,
            "checkpoint_path": resolved_checkpoint_path,
            "questions": questions,
            "ground_truths": ground_truths,
            "answers": answers,
            "contexts_list": contexts_list,
            "question_statuses": question_statuses,
            "questions_count": len(questions),
            "indexed_fragments": total,
            "indexed_files_filter": solo_archivos,
            "recomp_enabled": rag_runtime.USAR_RECOMP_SYNTHESIS,
            "pipeline_flags": current_pipeline_flags,
            "eval_corpus": eval_corpus,
            "docs_dir": rag_runtime.CARPETA_DOCS,
            "ragbench_reranker_low_score_fallback": ragbench_reranker_fallback,
            "pipeline_seconds": t_rag,
            "tiene_ground_truth": tiene_ground_truth,
        }
    finally:
        if previous_ragbench_reranker_fallback is not None:
            rag_runtime.set_ragbench_reranker_low_score_fallback(previous_ragbench_reranker_fallback)
        if docs_previous is not None:
            rag_runtime.CARPETA_DOCS, rag_runtime.PATH_DB, rag_runtime.COLLECTION_NAME = (
                docs_previous
            )
        if previous_pipeline_flags is not None:
            rag_runtime.set_pipeline_flags(previous_pipeline_flags)
