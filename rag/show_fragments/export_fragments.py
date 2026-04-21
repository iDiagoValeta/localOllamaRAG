# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Imports
# +-- 2. Constants, classification, CLI
#
# BUSINESS LOGIC
# +-- 3. Chroma export (batched get)
# +-- 4. Writers (text / jsonl)
#
# ENTRY
# +-- 5. main()
#
# ─────────────────────────────────────────────
"""
export_ragbench_fragments.py -- Dump indexed ChromaDB chunks to text or JSONL files.

**Default:** exports **two** files — production ``vector_db`` (PDFs en ``DOCS_FOLDER``)
and RAGBench (``vector_db``, PDFs de ``rag/docs/en``). Each chunk is
labeled as **imagen** (descripción de figura), **texto plano** o **texto Markdown**
according to ``metadata["format"]`` from ``chat_pdfs.indexar_documentos``.

**Single-store modes:** ``--mi-only`` or ``--ragbench-only`` (``--ragbench`` alias).

Usage (from repository root):
    python rag/show_fragments/export_fragments.py
    python rag/show_fragments/export_fragments.py --mi-only -o salida.txt
    python rag/show_fragments/export_fragments.py --format jsonl --out-dir ./mi_salida

Dependencies:
    - chromadb (see rag/requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────

# Three levels up: rag/show_fragments/export_fragments.py -> repo root.
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import chromadb  # noqa: E402

# ─────────────────────────────────────────────
# SECTION 2: CONSTANTS, CLASSIFICATION, CLI
# ─────────────────────────────────────────────

RAG_DIR = os.path.join(_proj_root, "rag")
RAGBENCH_DB_PATH = os.path.join(RAG_DIR, "vector_db", "en_embeddinggemma")
RAGBENCH_COLLECTION = "ragbench_arxiv_eval"

DEFAULT_OUT_DIR = os.path.join(RAG_DIR, "show_fragments")
DEFAULT_FILE_MI = "chunks_vector_db.txt"
DEFAULT_FILE_RAGBENCH = "chunks_ragbench_en.txt"
DEFAULT_OUTPUT_MONKEYGRAB = os.path.join(DEFAULT_OUT_DIR, "monkeygrab_chunks_export.txt")
DEFAULT_OUTPUT_RAGBENCH = os.path.join(DEFAULT_OUT_DIR, "ragbench_chunks_export.txt")
BATCH_SIZE = 500


def resolve_monkeygrab_chroma() -> Tuple[str, str]:
    """Resolve PATH_DB and COLLECTION_NAME the same way as ``rag/chat_pdfs.py``."""
    carpeta_docs = os.getenv("DOCS_FOLDER", os.path.join(RAG_DIR, "docs", "es"))
    modelo_embed = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
    carpeta_nombre = os.path.basename(os.path.abspath(carpeta_docs))
    embed_slug = modelo_embed.split(":")[0].replace("/", "_")
    path_db = os.path.join(RAG_DIR, "vector_db", f"{carpeta_nombre}_{embed_slug}")
    collection_name = f"docs_{carpeta_nombre}"
    return path_db, collection_name


def classify_fragment(meta: Dict[str, Any]) -> Dict[str, str]:
    """Map ``metadata['format']`` to Spanish labels (image vs text modalities).

    ``chat_pdfs.py`` sets ``format`` to ``markdown``, ``plain_text``, or ``image``.

    Returns:
        Dict with keys: kind_key, label_es, format_raw.
    """
    raw = meta.get("format")
    fmt = (str(raw).strip().lower() if raw is not None else "") or ""
    if fmt == "image":
        return {
            "kind_key": "imagen",
            "label_es": "Imagen (descripción indexada)",
            "format_raw": str(raw) if raw is not None else "",
        }
    if fmt == "plain_text":
        return {
            "kind_key": "texto_plano",
            "label_es": "Texto plano",
            "format_raw": str(raw) if raw is not None else "",
        }
    if fmt == "markdown":
        return {
            "kind_key": "texto_markdown",
            "label_es": "Texto (Markdown / pymupdf4llm)",
            "format_raw": str(raw) if raw is not None else "",
        }
    return {
        "kind_key": "desconocido",
        "label_es": f"Desconocido (format={raw!r})",
        "format_raw": str(raw) if raw is not None else "",
    }


def _sort_key(row: Tuple[str, str, Dict[str, Any]]) -> Tuple:
    """Build a stable sort key from (id, document, metadata)."""
    _id, _doc, meta = row
    source = meta.get("source") or ""
    page = meta.get("page")
    chunk = meta.get("chunk")
    try:
        page_i = int(page) if page is not None else -1
    except (TypeError, ValueError):
        page_i = -1
    try:
        chunk_i = int(chunk) if chunk is not None else -1
    except (TypeError, ValueError):
        chunk_i = -1
    return (source, page_i, chunk_i, _id)


def fetch_all_rows(
    collection: chromadb.Collection,
    batch_size: int = BATCH_SIZE,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Retrieve all ids, documents, and metadatas in batches."""
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, chunk_id in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) and metas[i] else {}
            rows.append((chunk_id, doc or "", meta))
        offset += len(ids)
    return rows


def _kind_counts(rows: List[Tuple[str, str, Dict[str, Any]]]) -> Counter:
    c: Counter = Counter()
    for _id, _doc, meta in rows:
        c[classify_fragment(meta)["kind_key"]] += 1
    return c


# ─────────────────────────────────────────────
# SECTION 4: WRITERS
# ─────────────────────────────────────────────

def write_text(
    path: str,
    rows: List[Tuple[str, str, Dict[str, Any]]],
    store_label: str,
    pdf_folder_hint: str,
) -> None:
    """Write a human-readable UTF-8 report with tipo de fragmento destacado."""
    rows_sorted = sorted(rows, key=_sort_key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = _kind_counts(rows_sorted)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ChromaDB — exportación de fragmentos\n")
        f.write(f"# almacén: {store_label}\n")
        f.write(f"# carpeta PDFs (referencia): {pdf_folder_hint}\n")
        f.write(f"# total fragmentos: {len(rows_sorted)}\n")
        f.write("# Resumen por tipo:\n")
        for key in ("imagen", "texto_plano", "texto_markdown", "desconocido"):
            if counts.get(key, 0):
                f.write(f"#   - {key}: {counts[key]}\n")
        f.write("\n")

        for idx, (chunk_id, doc, meta) in enumerate(rows_sorted):
            kind = classify_fragment(meta)
            f.write("=" * 72 + "\n")
            f.write(f"FRAGMENTO {idx + 1} / {len(rows_sorted)}\n")
            f.write(f"tipo_contenido: {kind['label_es']}\n")
            f.write(f"tipo (clave): {kind['kind_key']}\n")
            f.write(f"id: {chunk_id}\n")
            for k in sorted(meta.keys()):
                f.write(f"meta.{k}: {meta[k]}\n")
            f.write("-" * 72 + "\n")
            f.write((doc or "").rstrip() + "\n\n")


def write_jsonl(
    path: str,
    rows: List[Tuple[str, str, Dict[str, Any]]],
    store_label: str,
) -> None:
    """Write one JSON object per line including chunk kind fields."""
    rows_sorted = sorted(rows, key=_sort_key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk_id, doc, meta in rows_sorted:
            kind = classify_fragment(meta)
            line = {
                "id": chunk_id,
                "document": doc,
                "metadata": meta,
                "almacen": store_label,
                "chunk_kind": kind["kind_key"],
                "chunk_kind_label_es": kind["label_es"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def open_collection_safe(
    db_path: str,
    collection_name: str,
) -> Tuple[Optional[chromadb.Collection], Optional[str]]:
    """Return (collection, error_message)."""
    if not os.path.isdir(db_path):
        return None, f"path no existe: {db_path}"
    try:
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(name=collection_name)
        return col, None
    except Exception as e:
        return None, str(e)


def run_export(
    db_path: str,
    collection_name: str,
    out_path: str,
    out_format: str,
    store_label: str,
    pdf_folder_hint: str,
) -> int:
    """Export one collection. Returns number of rows or -1 on skip/failure."""
    col, err = open_collection_safe(db_path, collection_name)
    if col is None:
        print(f"  [omitido] {store_label}: {err}")
        return -1
    n = col.count()
    if n == 0:
        print(f"  [vacío] {store_label}: 0 fragmentos — no se escribe archivo")
        return 0
    rows = fetch_all_rows(col)
    if out_format == "jsonl":
        write_jsonl(out_path, rows, store_label)
    else:
        write_text(out_path, rows, store_label, pdf_folder_hint)
    print(f"  {store_label}: {len(rows)} fragmentos -> {out_path}")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export ChromaDB fragments; por defecto vector_db producción + ragbench/en en archivos separados.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--mi-only",
        action="store_true",
        help="Solo base de producción (vector_db según DOCS_FOLDER / OLLAMA_EMBED_MODEL)",
    )
    mode.add_argument(
        "--ragbench-only",
        action="store_true",
        help=f"Solo RAGBench ({RAGBENCH_DB_PATH})",
    )
    p.add_argument(
        "--ragbench",
        action="store_true",
        help="Alias de --ragbench-only",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help="Chroma persistent path manual (requiere --collection)",
    )
    p.add_argument(
        "--collection",
        default=None,
        help="Nombre de colección (obligatorio con --db-path)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Archivo de salida (solo con un único destino: --mi-only, --ragbench-only o --db-path)",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directorio para modo dual (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--format",
        choices=("text", "jsonl"),
        default="text",
        help="text: legible; jsonl: una línea JSON por fragmento con chunk_kind",
    )
    return p


# ─────────────────────────────────────────────
# SECTION 5: ENTRY
# ─────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()
    ragbench_only = args.ragbench_only or args.ragbench
    out_fmt = args.format
    out_dir = os.path.abspath(args.out_dir)

    # --- Manual single target ---
    if args.db_path:
        if not args.collection:
            print("ERROR: --collection es obligatorio con --db-path")
            raise SystemExit(1)
        db_path = os.path.abspath(args.db_path)
        if args.output:
            out_path = os.path.abspath(args.output)
        elif ragbench_only:
            out_path = os.path.abspath(DEFAULT_OUTPUT_RAGBENCH)
        else:
            out_path = os.path.abspath(DEFAULT_OUTPUT_MONKEYGRAB)
        hint = os.getenv("DOCS_FOLDER", os.path.join(RAG_DIR, "docs", "es"))
        n = run_export(
            db_path,
            args.collection,
            out_path,
            out_fmt,
            "personalizado",
            hint,
        )
        raise SystemExit(0 if n >= 0 else 1)

    # --- Dual default: vector_db producción + ragbench/en ---
    if not args.mi_only and not ragbench_only:
        os.makedirs(out_dir, exist_ok=True)
        ext = ".jsonl" if out_fmt == "jsonl" else ".txt"
        path_mi = os.path.join(out_dir, DEFAULT_FILE_MI.replace(".txt", ext))
        path_rb = os.path.join(out_dir, DEFAULT_FILE_RAGBENCH.replace(".txt", ext))

        db_mi, coll_mi = resolve_monkeygrab_chroma()
        db_mi = os.path.abspath(db_mi)
        pdf_mi = os.getenv("DOCS_FOLDER", os.path.join(RAG_DIR, "docs", "es"))

        print("Exportación dual (vector_db producción + ragbench en)\n")
        run_export(
            db_mi,
            coll_mi,
            path_mi,
            out_fmt,
            "vector_db (producción)",
            os.path.abspath(pdf_mi),
        )
        run_export(
            os.path.abspath(RAGBENCH_DB_PATH),
            RAGBENCH_COLLECTION,
            path_rb,
            out_fmt,
            "vector_db/en (evaluación RAGBench)",
            os.path.join(RAG_DIR, "docs", "en"),
        )
        print("\nListo. Archivos:")
        print(f"   {path_mi}")
        print(f"   {path_rb}")
        return

    # --- mi-only ---
    if args.mi_only:
        db_path, collection_name = resolve_monkeygrab_chroma()
        db_path = os.path.abspath(db_path)
        out_path = os.path.abspath(args.output) if args.output else DEFAULT_OUTPUT_MONKEYGRAB
        pdf_hint = os.getenv("DOCS_FOLDER", os.path.join(RAG_DIR, "docs", "es"))
        n = run_export(
            db_path,
            collection_name,
            out_path,
            out_fmt,
            "vector_db (producción)",
            os.path.abspath(pdf_hint),
        )
        raise SystemExit(0 if n != -1 else 1)

    # --- ragbench-only ---
    out_path = os.path.abspath(args.output) if args.output else DEFAULT_OUTPUT_RAGBENCH
    n = run_export(
        os.path.abspath(RAGBENCH_DB_PATH),
        RAGBENCH_COLLECTION,
        out_path,
        out_fmt,
        "vector_db/en (evaluación RAGBench)",
        os.path.join(RAG_DIR, "docs", "en"),
    )
    raise SystemExit(0 if n != -1 else 1)


if __name__ == "__main__":
    main()
