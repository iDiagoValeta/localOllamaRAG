"""
Evaluación con RAGAS del pipeline RAG.

Métricas evaluadas:
    - answer_correctness: Calidad global de la respuesta (similitud semántica + overlap factual vs ground truth).
    - faithfulness: ¿La respuesta se basa en los contextos recuperados?
    - answer_relevancy: ¿La respuesta es relevante a la pregunta?
    - context_precision: ¿Los contextos relevantes están bien rankeados?
    - context_recall: ¿Se recuperaron todos los contextos necesarios?
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
    """
    Configura el LLM y embeddings para que RAGAS evalúe las métricas.
    - LLM: Gemini (vía google-genai + llm_factory) o OpenAI como fallback.
    - Embeddings: HuggingFace local (all-MiniLM-L6-v2) para evitar problemas
      con la API de Google Embeddings.

    Returns:
        (evaluator_llm, evaluator_embeddings)
    """
    eval_llm = None
    eval_embeddings = None

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        eval_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        print("Embeddings de evaluación: HuggingFace (all-MiniLM-L6-v2, local)")
    except ImportError as err:
        print(f"No se pudo cargar embeddings locales: {err}")
        print("   Instala con: pip install sentence-transformers langchain-community")

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from google import genai
            from ragas.llms import llm_factory

            client = genai.Client(api_key=gemini_key)
            eval_llm = llm_factory(
                "gemini-2.0-flash",
                provider="google",
                client=client,
            )
            print("LLM de evaluación: Gemini 2.0 Flash (GEMINI_API_KEY)")
        except ImportError as err:
            print(f"GEMINI_API_KEY detectado pero google-genai no instalado: {err}")
            print("   Instala con: pip install google-genai")
    elif os.getenv("OPENAI_API_KEY"):
        try:
            from ragas.llms import llm_factory

            eval_llm = llm_factory("gpt-4o-mini")
            print("LLM de evaluación: OpenAI gpt-4o-mini (OPENAI_API_KEY)")
        except ImportError as err:
            print(f"OPENAI_API_KEY detectado pero openai no instalado: {err}")
    else:
        print("No se encontró GEMINI_API_KEY ni OPENAI_API_KEY.")
        print("RAGAS necesita un LLM externo para calcular métricas.")
        print("Configura una variable de entorno con tu API key.")
        raise SystemExit(1)

    if eval_llm is None:
        print("No se pudo configurar el LLM de evaluación.")
        raise SystemExit(1)

    return eval_llm, eval_embeddings

# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

METRIC_DESCRIPTIONS = {
    "answer_correctness": "Calidad global (semántica + factual vs ground truth)",
    "faithfulness":       "Fidelidad al contexto recuperado",
    "answer_relevancy":   "Relevancia de la respuesta a la pregunta",
    "context_precision":  "Precisión del ranking de contextos",
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
    print("  📊 RESULTADOS RAGAS — MEDIAS GLOBALES")
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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Importar métricas RAGAS v0.2+
    # ------------------------------------------------------------------

    try:
        from ragas import evaluate
        from ragas.metrics import (
            AnswerCorrectness,
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
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

    for i, q in enumerate(questions):
        if args.verbose:
            print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
        answer, contexts = evaluar_pregunta_rag(q, collection)
        answers.append(answer)
        contexts_list.append(contexts)

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
    # 7. Configurar métricas (usando objetos pre-instanciados y parcheando LLM)
    # ------------------------------------------------------------------

    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_embeddings
    context_precision.llm = eval_llm
    context_recall.llm = eval_llm

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    if tiene_ground_truth:
        ac = AnswerCorrectness(llm=eval_llm, embeddings=eval_embeddings)
        metrics.insert(0, ac)

    # ------------------------------------------------------------------
    # 8. Ejecutar evaluación RAGAS
    # ------------------------------------------------------------------

    print("\nEjecutando evaluación RAGAS (esto puede tardar unos minutos)...")
    t_eval_start = time.time()

    eval_run_config = RunConfig(timeout=600, max_retries=15)

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
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


if __name__ == "__main__":
    main()
