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
export_fragments.py -- Dump indexed ChromaDB chunks to text or JSONL files.

**Default:** exports every local vector store used by the project under
``rag/vector_db``: the language corpora ``ca_embeddinggemma``,
``en_embeddinggemma``, ``es_embeddinggemma`` and the two RagBench EN corpora
``en_ragbench_dev_embeddinggemma`` and ``en_ragbench_eval_embeddinggemma``.
Stores that are absent from disk are silently skipped. Each chunk is labeled as
**imagen** (descripción de figura), **texto plano** or **texto Markdown**
according to ``metadata["format"]`` from ``chat_pdfs.indexar_documentos``.

Usage (from repository root):
    python rag/show_fragments/export_fragments.py                # all stores
    python rag/show_fragments/export_fragments.py --language es  # one language
    python rag/show_fragments/export_fragments.py --ragbench dev # RagBench dev
    python rag/show_fragments/export_fragments.py --ragbench eval
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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
VECTOR_DB_DIR = os.path.join(RAG_DIR, "vector_db")
DEFAULT_LANGUAGES = ("ca", "en", "es")
RAGBENCH_SPLITS = ("dev", "eval")
DEFAULT_STORE_SLUGS = DEFAULT_LANGUAGES + tuple(f"en_ragbench_{s}" for s in RAGBENCH_SPLITS)

DEFAULT_OUT_DIR = os.path.join(RAG_DIR, "show_fragments")
DEFAULT_FILE_TEMPLATE = "chunks_vector_db_{slug}.txt"
DEFAULT_OUTPUT_MONKEYGRAB = os.path.join(DEFAULT_OUT_DIR, "monkeygrab_chunks_export.txt")
BATCH_SIZE = 500


class StoreSpec(NamedTuple):
    slug: str
    db_path: str
    collection_name: str
    store_label: str
    pdf_folder_hint: str


def _embed_slug() -> str:
    modelo_embed = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
    return modelo_embed.split(":")[0].replace("/", "_")


def build_store_spec(slug: str) -> StoreSpec:
    """Build the Chroma path and collection name for one store slug.

    ``slug`` is the basename of the PDF folder under ``rag/docs/`` (e.g.
    ``es``, ``ca``, ``en``, ``en_ragbench_dev``, ``en_ragbench_eval``). It is
    matched verbatim against the Chroma path and collection names produced by
    ``chat_pdfs.py`` (see ``PATH_DB`` / ``COLLECTION_NAME`` there).
    """
    s = slug.strip().lower()
    embed_slug = _embed_slug()
    return StoreSpec(
        slug=s,
        db_path=os.path.join(VECTOR_DB_DIR, f"{s}_{embed_slug}"),
        collection_name=f"docs_{s}",
        store_label=f"vector_db/{s}_{embed_slug} ({s})",
        pdf_folder_hint=os.path.join(RAG_DIR, "docs", s),
    )


def default_store_specs() -> List[StoreSpec]:
    """Return the local vector stores exported by default (languages + RagBench)."""
    return [build_store_spec(slug) for slug in DEFAULT_STORE_SLUGS]


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
    """Return (collection, error_message).

    If the expected ``collection_name`` is missing but the persistent store
    contains exactly one collection, fall back to it and print a notice. This
    tolerates legacy indexes created before the ``COLLECTION_NAME`` convention
    stabilized (e.g. a folder renamed from ``en`` to ``en_ragbench_dev`` after
    indexing).
    """
    if not os.path.isdir(db_path):
        return None, f"path no existe: {db_path}"
    try:
        client = chromadb.PersistentClient(path=db_path)
        try:
            return client.get_collection(name=collection_name), None
        except Exception:
            available = [c.name for c in client.list_collections()]
            if len(available) == 1:
                fallback = available[0]
                print(
                    f"  [aviso] colección esperada '{collection_name}' no existe; "
                    f"usando '{fallback}' (único contenido del store)"
                )
                return client.get_collection(name=fallback), None
            if not available:
                return None, f"store sin colecciones ({db_path})"
            return None, (
                f"colección '{collection_name}' no existe; "
                f"disponibles: {', '.join(available)}"
            )
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
        description=(
            "Export ChromaDB fragments; por defecto exporta los almacenes locales "
            "bajo rag/vector_db (idiomas ca/en/es + RagBench dev/eval)."
        ),
    )
    target = p.add_mutually_exclusive_group()
    target.add_argument(
        "--language",
        choices=DEFAULT_LANGUAGES,
        default=None,
        help="Exportar una única base vectorial por idioma (ca, en, es).",
    )
    target.add_argument(
        "--ragbench",
        choices=RAGBENCH_SPLITS,
        default=None,
        help="Exportar un corpus RagBench EN ('dev' o 'eval').",
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
        help="Archivo de salida (solo con un único destino: --language, --ragbench o --db-path)",
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

    # --- Single store target (--language or --ragbench) ---
    single_slug = args.language or (f"en_ragbench_{args.ragbench}" if args.ragbench else None)
    if single_slug:
        spec = build_store_spec(single_slug)
        out_path = (
            os.path.abspath(args.output)
            if args.output
            else os.path.join(
                out_dir,
                DEFAULT_FILE_TEMPLATE.format(slug=spec.slug).replace(
                    ".txt",
                    ".jsonl" if out_fmt == "jsonl" else ".txt",
                ),
            )
        )
        n = run_export(
            os.path.abspath(spec.db_path),
            spec.collection_name,
            out_path,
            out_fmt,
            spec.store_label,
            os.path.abspath(spec.pdf_folder_hint),
        )
        raise SystemExit(0 if n != -1 else 1)

    if args.output:
        print("ERROR: --output solo se puede usar con --language, --ragbench o --db-path")
        raise SystemExit(1)

    # --- Default: all local stores under rag/vector_db ---
    os.makedirs(out_dir, exist_ok=True)
    ext = ".jsonl" if out_fmt == "jsonl" else ".txt"
    exported_paths: List[str] = []
    failures = 0

    slugs_humanos = ", ".join(DEFAULT_STORE_SLUGS)
    print(f"Exportación de las bases vectoriales locales ({slugs_humanos})\n")
    for spec in default_store_specs():
        out_path = os.path.join(
            out_dir,
            DEFAULT_FILE_TEMPLATE.format(slug=spec.slug).replace(".txt", ext),
        )
        n = run_export(
            os.path.abspath(spec.db_path),
            spec.collection_name,
            out_path,
            out_fmt,
            spec.store_label,
            os.path.abspath(spec.pdf_folder_hint),
        )
        if n == -1:
            failures += 1
        else:
            exported_paths.append(out_path)

    print("\nListo. Archivos:")
    for path in exported_paths:
        print(f"   {path}")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
