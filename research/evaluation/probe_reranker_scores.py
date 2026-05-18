"""Reranker score-distribution probe for threshold calibration.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Environment + imports
#  2. Corpus -> docs/dataset mapping
#  3. Per-question probe
#  4. CLI
#
# ─────────────────────────────────────────────

Usage:
    python research/evaluation/probe_reranker_scores.py --corpus es --n 8
    python research/evaluation/probe_reranker_scores.py --corpus ca --n 8
    python research/evaluation/probe_reranker_scores.py --corpus en_ragbench_dev --n 8

Goal:
    For each sampled question, run the hybrid retrieval pipeline with the
    reranker DISABLED (to capture all post-RRF candidates), then score those
    candidates with the Cross-Encoder manually so the full reranker-score
    distribution is exposed (not only the top-K kept by TOP_K_AFTER_RERANK).
    Output is a single JSON per corpus under rag/debug_rag/, suitable for
    threshold calibration (currently UMBRAL_SCORE_RERANKER = 0.55, candidate
    new value 0.70 — see research/docs/REINFERENCIA_FINAL.md).
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT + IMPORTS
# ─────────────────────────────────────────────

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / ".env")
except ImportError:
    pass

import chromadb

import rag.chat_pdfs as rag_runtime
from rag.engine.reranking import obtener_modelo_reranker


# ─────────────────────────────────────────────
# SECTION 2: CORPUS -> DOCS/DATASET MAPPING
# ─────────────────────────────────────────────

CORPUS_MAP = {
    "es": {
        "docs": _PROJ / "rag" / "docs" / "es",
        "dataset": _PROJ / "research" / "evaluation" / "datasets" / "local" / "dataset_eval_es.json",
    },
    "ca": {
        "docs": _PROJ / "rag" / "docs" / "ca",
        "dataset": _PROJ / "research" / "evaluation" / "datasets" / "local" / "dataset_eval_ca.json",
    },
    "en_ragbench_dev": {
        "docs": _PROJ / "rag" / "docs" / "en_ragbench_dev",
        "dataset": _PROJ / "research" / "evaluation" / "datasets" / "ragbench"
                   / "dev_frozen" / "dataset_ragbench_text_10p_5q_dev10_frozen.json",
    },
}

DEBUG_DIR = _PROJ / "rag" / "debug_rag"
THRESHOLD_CUTS = (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


# ─────────────────────────────────────────────
# SECTION 3: PER-QUESTION PROBE
# ─────────────────────────────────────────────

def _texto_para_rerank(doc: dict) -> str:
    texto = doc.get("doc", "")
    if "\\n\\n" in texto:
        texto = texto.split("\\n\\n", 1)[-1]
    return texto


def sondear_pregunta(pregunta: str, ground_truth: str | None, collection) -> dict:
    """Run retrieval w/o reranker, then rerank-score all candidates manually."""
    # Force reranker OFF for retrieval so we keep the full fusion candidate list.
    previous_flags = rag_runtime.set_pipeline_flags({"USAR_RERANKER": False})
    try:
        fragmentos, mejor_score_rrf, metricas = rag_runtime.realizar_busqueda_hibrida(
            pregunta, collection
        )
    finally:
        rag_runtime.set_pipeline_flags(previous_flags)

    # Now score all candidates with the reranker directly.
    reranker = obtener_modelo_reranker()
    rerank_scores: list[float] = []
    if reranker is not None and fragmentos:
        textos = [_texto_para_rerank(f) for f in fragmentos]
        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            ranks = reranker.rank(
                pregunta, textos, top_k=len(textos), return_documents=False
            )
        ordered = [None] * len(fragmentos)
        for r in ranks:
            ordered[r["corpus_id"]] = float(r["score"])
        rerank_scores = [s if s is not None else 0.0 for s in ordered]

    candidatos = []
    for i, frag in enumerate(fragmentos):
        score_rerank = rerank_scores[i] if i < len(rerank_scores) else None
        candidatos.append({
            "rank_rrf": i + 1,
            "source": frag.get("metadata", {}).get("source"),
            "page": frag.get("metadata", {}).get("page"),
            "score_final_rrf": float(frag.get("score_final", 0.0)),
            "score_reranker": score_rerank,
            "preview": (frag.get("doc", "")[:160] + ("..." if len(frag.get("doc", "")) > 160 else "")),
        })

    candidatos_ordenados = sorted(
        candidatos,
        key=lambda c: (c["score_reranker"] if c["score_reranker"] is not None else -1.0),
        reverse=True,
    )

    valid_scores = [c["score_reranker"] for c in candidatos if c["score_reranker"] is not None]
    cuts = {
        f"{t:.2f}": sum(1 for s in valid_scores if s >= t)
        for t in THRESHOLD_CUTS
    }
    stats = {}
    if valid_scores:
        stats = {
            "n": len(valid_scores),
            "min": min(valid_scores),
            "max": max(valid_scores),
            "mean": statistics.fmean(valid_scores),
            "median": statistics.median(valid_scores),
            "p25": statistics.quantiles(valid_scores, n=4)[0] if len(valid_scores) >= 4 else None,
            "p75": statistics.quantiles(valid_scores, n=4)[2] if len(valid_scores) >= 4 else None,
        }

    return {
        "pregunta": pregunta,
        "ground_truth": ground_truth,
        "n_candidatos_fusion": len(fragmentos),
        "mejor_score_rrf": float(mejor_score_rrf),
        "sub_queries": metricas.get("sub_queries", []),
        "queries_semanticas": metricas.get("queries_semanticas", []),
        "keywords": metricas.get("keywords", []),
        "terminos_criticos": metricas.get("terminos_criticos", []),
        "candidatos": candidatos_ordenados,
        "umbral_cuts": cuts,
        "stats_reranker": stats,
    }


# ─────────────────────────────────────────────
# SECTION 4: CLI
# ─────────────────────────────────────────────

def cargar_preguntas(dataset_path: Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data:
        q = row.get("question") or row.get("pregunta") or row.get("preguntas")
        gt = (row.get("ground_truth") or row.get("respuesta_esperada")
              or row.get("respuesta_referencia") or row.get("reference"))
        if q:
            out.append({"question": q, "ground_truth": gt})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe reranker score distribution per corpus.")
    parser.add_argument("--corpus", choices=sorted(CORPUS_MAP), required=True)
    parser.add_argument("--n", type=int, default=8, help="Number of questions to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", default=None,
                        help="Comma-separated 1-based indices to override random sampling.")
    args = parser.parse_args()

    cfg = CORPUS_MAP[args.corpus]
    docs_dir = cfg["docs"]
    dataset_path = cfg["dataset"]
    if not docs_dir.exists():
        raise SystemExit(f"ERROR: docs dir not found: {docs_dir}")
    if not dataset_path.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset_path}")

    rag_runtime.set_docs_folder_runtime(str(docs_dir))
    print(f"\nCorpus:      {args.corpus}")
    print(f"Docs:        {docs_dir}")
    print(f"Dataset:     {dataset_path}")
    print(f"ChromaDB:    {rag_runtime.PATH_DB}")
    print(f"Collection:  {rag_runtime.COLLECTION_NAME}")

    client = chromadb.PersistentClient(path=rag_runtime.PATH_DB)
    collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
    n_frag = collection.count()
    if n_frag == 0:
        raise SystemExit(
            f"ERROR: Chroma collection {rag_runtime.COLLECTION_NAME} is empty. "
            f"Index the corpus first (research/evaluation/index.py)."
        )
    print(f"Fragments:   {n_frag}")

    preguntas = cargar_preguntas(dataset_path)
    if not preguntas:
        raise SystemExit("ERROR: no questions parsed from dataset.")

    if args.indices:
        idxs = [int(s.strip()) - 1 for s in args.indices.split(",") if s.strip()]
        sample = [preguntas[i] for i in idxs if 0 <= i < len(preguntas)]
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(preguntas, k=min(args.n, len(preguntas)))

    print(f"Sampled {len(sample)} questions (seed={args.seed}).")

    # Make sure reranker is loadable up front (fail fast).
    if obtener_modelo_reranker() is None:
        raise SystemExit("ERROR: reranker model not available — cannot probe.")

    resultados = []
    t0 = time.time()
    for i, item in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {item['question'][:90]}")
        try:
            res = sondear_pregunta(item["question"], item.get("ground_truth"), collection)
        except Exception as exc:
            print(f"   ERROR: {exc}")
            resultados.append({
                "pregunta": item["question"],
                "ground_truth": item.get("ground_truth"),
                "error": str(exc),
            })
            continue
        cuts = res["umbral_cuts"]
        print(f"   fusion={res['n_candidatos_fusion']:>3} cuts: "
              + " ".join(f">={k}:{cuts[k]}" for k in cuts))
        resultados.append(res)

    # Aggregate stats across all candidates of all probed questions.
    all_scores: list[float] = []
    for r in resultados:
        for c in r.get("candidatos", []):
            if c.get("score_reranker") is not None:
                all_scores.append(c["score_reranker"])

    agg_cuts = {f"{t:.2f}": sum(1 for s in all_scores if s >= t) for t in THRESHOLD_CUTS}
    aggregate = {
        "total_candidatos": len(all_scores),
        "umbral_cuts": agg_cuts,
        "stats": {
            "min": min(all_scores) if all_scores else None,
            "max": max(all_scores) if all_scores else None,
            "mean": statistics.fmean(all_scores) if all_scores else None,
            "median": statistics.median(all_scores) if all_scores else None,
            "p10": statistics.quantiles(all_scores, n=10)[0] if len(all_scores) >= 10 else None,
            "p25": statistics.quantiles(all_scores, n=4)[0] if len(all_scores) >= 4 else None,
            "p75": statistics.quantiles(all_scores, n=4)[2] if len(all_scores) >= 4 else None,
            "p90": statistics.quantiles(all_scores, n=10)[-1] if len(all_scores) >= 10 else None,
        },
    }

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = DEBUG_DIR / f"probe_{args.corpus}_{stamp}.json"
    payload = {
        "corpus": args.corpus,
        "docs_dir": str(docs_dir),
        "dataset": str(dataset_path),
        "chroma_path": rag_runtime.PATH_DB,
        "collection": rag_runtime.COLLECTION_NAME,
        "fragments_in_collection": n_frag,
        "sample_size": len(sample),
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "thresholds_evaluated": list(THRESHOLD_CUTS),
        "current_umbral_score_reranker": rag_runtime.UMBRAL_SCORE_RERANKER,
        "aggregate": aggregate,
        "questions": resultados,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nProbe finished in {payload['elapsed_seconds']}s.")
    print(f"Output: {out_path}")
    print("Aggregate cuts (candidates passing each threshold):")
    for t, n in agg_cuts.items():
        pct = (n / aggregate["total_candidatos"] * 100) if aggregate["total_candidatos"] else 0
        print(f"   >= {t}: {n:>4} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
