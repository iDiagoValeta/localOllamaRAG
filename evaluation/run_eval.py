"""
Evaluación con RAGAS del pipeline RAG.

Métricas empleadas (según documentación RAGAS):
    - Context Recall: Mide la capacidad del retriever para extraer toda la información relevante.
    - Factual Correctness (answer_correctness): Precisión factual vs ground truth (TP/FP/FN).
    - Context Precision: Evalúa el ranking de fragmentos recuperados.
    - Faithfulness: Consistencia factual de la respuesta con el contexto recuperado.
    - Response Relevancy (answer_relevancy): Grado en que la respuesta aborda la pregunta.
"""

import os
import sys
import json
import argparse
import time

# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pandas as pd
import chromadb

from rag.chat_pdfs import (
    evaluar_pregunta_rag,
    PATH_DB,
    COLLECTION_NAME,
    CARPETA_DOCS,
    indexar_documentos,
)

# ---------------------------------------------------------------------------
# Carga de dataset (JSON, CSV, Excel)
# ---------------------------------------------------------------------------

def cargar_dataset(ruta: str) -> pd.DataFrame:
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".json":
        with open(ruta, encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    if ext == ".csv":
        return pd.read_csv(ruta, encoding="utf-8")
    raise ValueError(f"Formato no soportado: {ext}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    mapeo = {
        "pregunta": "question",
        "question": "question",
        "preguntas": "question",
        "ground_truth": "ground_truth",
        "respuesta_esperada": "ground_truth",
        "respuesta_referencia": "ground_truth",
        "reference": "ground_truth",
    }
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for orig, target in mapeo.items():
        if orig in cols:
            out[target] = df[cols[orig]].tolist()
    if "question" not in out:
        raise ValueError("El dataset debe tener columnas 'question' o 'pregunta'")
    if "ground_truth" not in out:
        out["ground_truth"] = [""] * len(out["question"])
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Configuración del LLM y embeddings de evaluación
# ---------------------------------------------------------------------------

def configurar_llm_evaluacion():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("No se encontró GEMINI_API_KEY ni GOOGLE_API_KEY.")
        raise SystemExit(1)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        eval_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0,
        )
        eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key,
        )
        print("LLM de evaluación: Gemini 2.0 Flash (langchain-google-genai)")
        print("Embeddings de evaluación: Google gemini-embedding-001 (langchain-google-genai)")
        return eval_llm, eval_embeddings
    except ImportError as err:
        print(f"Error: {err}")
        print("  Instala con: pip install langchain-google-genai")
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "answer_correctness",  # Factual Correctness
    "faithfulness",        # Faithfulness
    "answer_relevancy",    # Response Relevancy
    "context_precision",   # Context Precision
    "context_recall",      # Context Recall
]

METRIC_DISPLAY_NAMES = {
    "answer_correctness": "Factual Correctness",
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Response Relevancy",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
}

METRIC_DESCRIPTIONS = {
    "answer_correctness": "Precisión factual vs ground truth (TP/FP/FN, F1)",
    "faithfulness":       "Consistencia factual de la respuesta con el contexto",
    "answer_relevancy":   "Grado en que la respuesta aborda la pregunta",
    "context_precision":  "Precisión del ranking de fragmentos recuperados",
    "context_recall":     "Cobertura de contextos necesarios",
}

def imprimir_resultados(df_scores: pd.DataFrame, questions: list[str]):
    """Imprime resultados detallados por pregunta y medias globales."""

    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    if not metric_cols:
        print("\nNo se encontraron columnas de métricas en los resultados.")
        print(f"   Columnas disponibles: {list(df_scores.columns)}")
        return

    print("\n" + "═" * 70)
    print("  RESULTADOS RAGAS — MEDIAS GLOBALES")
    print("═" * 70)

    medias = df_scores[metric_cols].mean(numeric_only=True).sort_values(ascending=False)
    for m, v in medias.items():
        desc = METRIC_DESCRIPTIONS.get(m, "")
        if pd.isna(v):
            print(f"  {m:25s}  {'N/A':>8s}   {desc}")
        else:
            print(f"  {m:25s}  {v:8.4f}   {desc}")


    media_global = medias.dropna().mean()
    if not pd.isna(media_global):
        print(f"\n  {'SCORE MEDIO GLOBAL':25s}  {media_global:8.4f}")

    print("\n" + "=" * 70)
    print("  DETALLE POR PREGUNTA")
    print("=" * 70)

    for i, row in df_scores.iterrows():
        q = questions[i] if i < len(questions) else "?"
        q_short = q[:80] + "..." if len(q) > 80 else q

        scores_str = " | ".join(
            f"{c}: {row[c]:.3f}" if not pd.isna(row[c]) else f"{c}: N/A"
            for c in metric_cols
        )

        row_scores = [row[c] for c in metric_cols if not pd.isna(row[c])]
        media_q = sum(row_scores) / len(row_scores) if row_scores else float("nan")

        print(f"\n  [{i+1}] {q_short}")
        print(f"      {scores_str}")
        if not pd.isna(media_q):
            print(f"      Score medio: {media_q:.4f}")

    print("\n" + "=" * 70)


def _extraer_justificaciones_traces(traces: list, metric_cols: list) -> list[dict]:
    """
    Extrae justificaciones de los traces de RAGAS cuando están disponibles.
    Los traces contienen los outputs de los prompts LLM (TP/FP/FN, statements, etc.).
    """
    justificaciones = []
    for i, trace in enumerate(traces):
        justif = {}
        if not hasattr(trace, "__getitem__"):
            justificaciones.append(justif)
            continue
        for metric_name in metric_cols:
            if metric_name not in trace:
                continue
            metric_data = trace[metric_name]
            if isinstance(metric_data, dict):
                prompts = []
                for prompt_name, prompt_io in metric_data.items():
                    if isinstance(prompt_io, dict) and "output" in prompt_io:
                        out = prompt_io["output"]
                        if isinstance(out, dict):
                            prompts.append({"prompt": prompt_name, "output": out})
                        elif out is not None:
                            prompts.append({"prompt": prompt_name, "output": str(out)[:500]})
                if prompts:
                    justif[metric_name] = prompts
            elif metric_data is not None:
                justif[metric_name] = str(metric_data)[:500]
        justificaciones.append(justif)
    return justificaciones


def guardar_debug(
    result,
    questions: list,
    answers: list,
    ground_truths: list,
    contexts_list: list,
    eval_dir: str,
) -> str:
    """
    Guarda un archivo JSON de debug con respuestas del modelo y justificaciones de puntuaciones.
    """
    df = result.to_pandas()
    metric_cols = [c for c in METRIC_NAMES if c in df.columns]

    traces = getattr(result, "traces", []) or []
    justificaciones = _extraer_justificaciones_traces(traces, metric_cols) if traces else [{}] * len(questions)

    debug_entries = []
    for i in range(len(questions)):
        ctx_preview = []
        for j, ctx in enumerate(contexts_list[i] if i < len(contexts_list) else []):
            ctx_preview.append(ctx[:300] + "..." if len(ctx) > 300 else ctx)

        entry = {
            "indice": i + 1,
            "pregunta": questions[i],
            "respuesta_modelo": answers[i] if i < len(answers) else "",
            "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
            "contextos_recuperados_preview": ctx_preview[:3],
            "contextos_count": len(contexts_list[i]) if i < len(contexts_list) else 0,
            "puntuaciones": {},
            "justificaciones": justificaciones[i] if i < len(justificaciones) else {},
        }
        for m in metric_cols:
            val = df.iloc[i][m] if i < len(df) else None
            entry["puntuaciones"][METRIC_DISPLAY_NAMES.get(m, m)] = (
                float(val) if val is not None and not pd.isna(val) else None
            )
        debug_entries.append(entry)

    debug_data = {
        "metricas_empleadas": {
            METRIC_DISPLAY_NAMES.get(m, m): METRIC_DESCRIPTIONS.get(m, "")
            for m in metric_cols
        },
        "resultados": debug_entries,
        "medias_globales": {
            METRIC_DISPLAY_NAMES.get(m, m): float(df[m].mean())
            for m in metric_cols
        },
    }

    debug_path = os.path.join(eval_dir, "ragas_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    return debug_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluar el sistema Teacher RAG con RAGAS v0.2+"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "dataset_eval.json"),
        help="Ruta al dataset (JSON, CSV o Excel)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida para CSV de resultados",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar progreso por pregunta",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="No guardar archivo ragas_debug.json",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Importar métricas RAGAS
    # ------------------------------------------------------------------

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.run_config import RunConfig
    except ImportError as e:
        print("Instala RAGAS y dependencias:")
        print("   pip install -r evaluation/requirements.txt")
        raise SystemExit(1) from e

    # ------------------------------------------------------------------
    # 2. Configurar LLM de evaluación
    # ------------------------------------------------------------------

    eval_llm, eval_embeddings = configurar_llm_evaluacion()

    # ------------------------------------------------------------------
    # 3. Cargar dataset
    # ------------------------------------------------------------------

    print(f"\nCargando dataset...")
    df = cargar_dataset(args.dataset)
    df = normalizar_columnas(df)
    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()

    tiene_ground_truth = any(gt.strip() for gt in ground_truths)

    print(f"   Preguntas a evaluar: {len(questions)}")
    print(f"   Ground truth disponible: {'Sí' if tiene_ground_truth else 'No'}")

    # ------------------------------------------------------------------
    # 4. Conectar a ChromaDB
    # ------------------------------------------------------------------

    print(f"\nConectando a ChromaDB: {PATH_DB}")
    client = chromadb.PersistentClient(path=PATH_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("   Base de datos vacía. Indexando documentos...")
        total = indexar_documentos(CARPETA_DOCS, collection)
        print(f"   Indexados {total} fragmentos.")
    else:
        print(f"   Fragmentos en la colección: {collection.count()}")

    # ------------------------------------------------------------------
    # 5. Ejecutar pipeline RAG para cada pregunta
    # ------------------------------------------------------------------

    print("\nEjecutando pipeline RAG por cada pregunta...")
    answers = []
    contexts_list = []
    t_start = time.time()

    try:
        for i, q in enumerate(questions):
            if args.verbose:
                print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
            answer, contexts = evaluar_pregunta_rag(q, collection)
            answers.append(answer)
            contexts_list.append(contexts)
    except ConnectionError as e:
        print(f"\nError: No se pudo conectar a Ollama: {e}")
        print("   Asegúrate de que Ollama está en ejecución antes de lanzar la evaluación.")
        print("   Inicia Ollama con: ollama serve")
        raise SystemExit(1)

    t_rag = time.time() - t_start
    print(f"   Pipeline completado en {t_rag:.1f}s ({t_rag/len(questions):.1f}s/pregunta)")

    # ------------------------------------------------------------------
    # 6. Construir EvaluationDataset con SingleTurnSample
    # ------------------------------------------------------------------

    print("\nConstruyendo EvaluationDataset para RAGAS...")
    samples = []
    for i in range(len(questions)):
        sample = SingleTurnSample(
            user_input=questions[i],
            response=answers[i] if answers[i] else "",
            retrieved_contexts=contexts_list[i] if contexts_list[i] else [],
            reference=ground_truths[i] if ground_truths[i] else "",
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)

    # ------------------------------------------------------------------
    # 7. Configurar métricas
    # ------------------------------------------------------------------

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if tiene_ground_truth:
        metrics.insert(0, answer_correctness)

    # ------------------------------------------------------------------
    # 8. Ejecutar evaluación RAGAS
    # ------------------------------------------------------------------

    print("\nEjecutando evaluación RAGAS (esto puede tardar unos minutos)...")
    t_eval_start = time.time()

    eval_run_config = RunConfig(timeout=600, max_retries=15)

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=eval_run_config,
    )

    t_eval = time.time() - t_eval_start
    print(f"   Evaluación completada en {t_eval:.1f}s")

    # ------------------------------------------------------------------
    # 9. Resultados
    # ------------------------------------------------------------------

    df_scores = result.to_pandas()
    imprimir_resultados(df_scores, questions)

    # ------------------------------------------------------------------
    # 10. Guardar CSV
    # ------------------------------------------------------------------
    
    out_path = args.output
    if not out_path:
        out_path = os.path.join(
            os.path.dirname(__file__),
            "ragas_scores.csv",
        )
    df_scores.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nResultados guardados en: {out_path}")

    # ------------------------------------------------------------------
    # 11. Guardar debug (respuesta del modelo + justificaciones)
    # ------------------------------------------------------------------

    if not args.no_debug:
        eval_dir = os.path.dirname(__file__)
        debug_path = guardar_debug(
            result=result,
            questions=questions,
            answers=answers,
            ground_truths=ground_truths,
            contexts_list=contexts_list,
            eval_dir=eval_dir,
        )
        print(f"Debug guardado en: {debug_path}")


if __name__ == "__main__":
    main()
