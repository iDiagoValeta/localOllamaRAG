"""MonkeyGrab corpus indexing CLI.

Indexa un corpus de PDFs en su ChromaDB asociado y termina. Es la primera
fase del flujo de evaluación: indexar → inferir → evaluar con RAGAS.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Environment setup
#  2. Corpus resolution and indexing
#  3. CLI entry point
#
# ─────────────────────────────────────────────

Usage:
    python research/evaluation/index.py --corpus es
    python research/evaluation/index.py --corpus ca --force
    python research/evaluation/index.py --docs-dir custom/path --force
    python research/evaluation/index.py --corpus ragbench-eval     # uses prepared manifest
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

import argparse
import os
import sys
from pathlib import Path

_this_file = Path(__file__).resolve()
_proj_root = _this_file.parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from dotenv import load_dotenv
    load_dotenv(_proj_root / ".env")
except ImportError:
    pass

import chromadb

import rag.chat_pdfs as rag_runtime
from rag.chat_pdfs import indexar_documentos

from research.evaluation._lib.datasets import (
    SUPPORTED_CORPORA,
    default_docs_dir_for_corpus,
)
from research.evaluation._lib.ragbench import (
    RAGBENCH_EVAL_MANIFEST_PATH,
    cargar_manifest_ragbench_eval,
    selected_pdf_filenames_from_manifest,
)


# ─────────────────────────────────────────────
# SECTION 2: CORPUS RESOLUTION AND INDEXING
# ─────────────────────────────────────────────

EXTENDED_CORPORA = SUPPORTED_CORPORA + ("ragbench-eval",)


def _resolve_docs_dir(corpus: str, override: str | None) -> tuple[str, list[str] | None]:
    """Return ``(docs_dir, solo_archivos)`` for the requested corpus."""
    if override:
        return os.path.abspath(override), None

    if corpus == "ragbench-eval":
        manifest = cargar_manifest_ragbench_eval(RAGBENCH_EVAL_MANIFEST_PATH)
        docs_dir = os.path.abspath(manifest["docs_dir"])
        indexed_files = selected_pdf_filenames_from_manifest(manifest)
        if not indexed_files:
            raise SystemExit("ERROR: el manifiesto RagBench EN no contiene indexed_files.")
        return docs_dir, indexed_files

    docs_dir = default_docs_dir_for_corpus(corpus)
    if docs_dir is None:
        # ``es`` falls back to the RAG module default folder.
        docs_dir = rag_runtime.CARPETA_DOCS
    return os.path.abspath(docs_dir), None


def ejecutar_indexacion(
    corpus: str,
    docs_dir_override: str | None = None,
    force: bool = False,
) -> int:
    """Index the requested corpus and return the number of stored fragments."""
    docs_dir, solo_archivos = _resolve_docs_dir(corpus, docs_dir_override)
    rag_runtime.set_docs_folder_runtime(docs_dir)

    print("=" * 70)
    print("MonkeyGrab indexing")
    print("=" * 70)
    print(f"   Corpus:       {corpus}")
    print(f"   Docs dir:     {docs_dir}")
    print(f"   ChromaDB:     {rag_runtime.PATH_DB}")
    print(f"   Collection:   {rag_runtime.COLLECTION_NAME}")
    print(f"   Embeddings:   {rag_runtime.MODELO_EMBEDDING}")
    if solo_archivos:
        listed = ", ".join(solo_archivos[:5]) + ("..." if len(solo_archivos) > 5 else "")
        print(f"   File filter:  {len(solo_archivos)} file(s) ({listed})")

    client = chromadb.PersistentClient(path=rag_runtime.PATH_DB)
    if force:
        print("\n   Force flag set. Dropping existing collection if present...")
        try:
            client.delete_collection(name=rag_runtime.COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
    existing = collection.count()
    if existing > 0 and not force and not solo_archivos:
        print(f"\n   Collection already has {existing} fragments. Use --force to rebuild.")
        return existing

    print("\n   Indexing documents...")
    added = indexar_documentos(rag_runtime.CARPETA_DOCS, collection, solo_archivos=solo_archivos)
    total = collection.count()
    print(f"\n   Added {added} fragments. Collection now holds {total}.")
    return total


# ─────────────────────────────────────────────
# SECTION 3: CLI ENTRY POINT
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index a MonkeyGrab corpus into its ChromaDB collection."
    )
    parser.add_argument(
        "--corpus",
        choices=EXTENDED_CORPORA,
        default="es",
        help="Corpus preset (default: es). Use 'ragbench-eval' to read the prepared manifest.",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Override the PDF folder. Skips corpus auto-resolution.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the existing collection and re-index from scratch.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    ejecutar_indexacion(corpus=args.corpus, docs_dir_override=args.docs_dir, force=args.force)


if __name__ == "__main__":
    main()
