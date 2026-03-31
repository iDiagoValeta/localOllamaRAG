"""
test_image_rag.py -- Consulta RAG dirigida a los chunks de imagen indexados por glm-ocr.

Conecta a la colección ragbench ya indexada, muestra las descripciones de imagen
generadas por glm-ocr, y luego lanza preguntas al pipeline RAG marcando qué
fragmentos recuperados son de texto y cuáles vienen de imágenes.

Usage (desde la raíz del repositorio):
    python scripts/tests/test_image_rag.py
    python scripts/tests/test_image_rag.py --list-images
    python scripts/tests/test_image_rag.py --question "Describe Figure 1 of the paper"

Dependencies:
    - chromadb
    - rag.chat_pdfs (evaluar_pregunta_rag, MODELO_EMBEDDING)
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports & entorno
#  +-- 2. Constantes
#
#  FUNCIONES
#  +-- 3. listar_chunks_imagen   muestra las descripciones OCR indexadas
#  +-- 4. busqueda_solo_imagenes consulta ChromaDB filtrando format=image
#  +-- 5. preguntar_al_rag       llama a evaluar_pregunta_rag y etiqueta contextos
#
#  ENTRY
#  +-- 6. main()
#
# ─────────────────────────────────────────────

import os
import sys
import argparse

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS Y ENTORNO
# ─────────────────────────────────────────────

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_proj_root, ".env"))
except ImportError:
    pass

import chromadb
from rag.chat_pdfs import evaluar_pregunta_rag, MODELO_EMBEDDING

# ─────────────────────────────────────────────
# SECTION 2: CONSTANTES
# ─────────────────────────────────────────────

RAGBENCH_DB_PATH   = os.path.join(_proj_root, "rag", "ragbench_vector_db")
RAGBENCH_COLLECTION = "ragbench_arxiv_eval"

# Preguntas de ejemplo orientadas a contenido visual del paper 2408.07618v3
PREGUNTAS_IMAGEN = [
    "Describe the snapshots of infected cells shown in Figure 3 of the paper.",
    "What does Figure 1 show about the tube-shaped tissue geometry?",
    "What is shown in the heatmap figures about tissue damage?",
    "Describe the schematic of the immune response model shown in Figure 8.",
]


# ─────────────────────────────────────────────
# SECTION 3: LISTAR CHUNKS DE IMAGEN
# ─────────────────────────────────────────────

def listar_chunks_imagen(collection: chromadb.Collection) -> list[dict]:
    """Fetch and display all image chunks stored in the collection.

    Image chunks are identified by metadata field ``format == "image"``.

    Args:
        collection: ChromaDB collection to inspect.

    Returns:
        List of image chunk dicts with keys: id, page, description.
    """
    resultado = collection.get(
        where={"format": "image"},
        include=["documents", "metadatas"],
    )

    ids       = resultado.get("ids", [])
    docs      = resultado.get("documents", [])
    metas     = resultado.get("metadatas", [])

    if not ids:
        print("  (ningún chunk de imagen encontrado — ¿se indexó con USAR_EMBEDDINGS_IMAGEN=True?)")
        return []

    chunks = []
    for cid, doc, meta in sorted(zip(ids, docs, metas), key=lambda x: (x[2].get("page", 0), x[0])):
        chunks.append({"id": cid, "page": meta.get("page", "?"), "description": doc})

    print(f"\n{'='*70}")
    print(f"  CHUNKS DE IMAGEN EN LA COLECCIÓN ({len(chunks)} total)")
    print(f"{'='*70}")
    for ch in chunks:
        print(f"\n  [pag {ch['page']}] ID: {ch['id']}")
        preview = ch["description"][:400] + "..." if len(ch["description"]) > 400 else ch["description"]
        print(f"  {preview}")

    return chunks


# ─────────────────────────────────────────────
# SECTION 4: BÚSQUEDA SOLO POR IMÁGENES
# ─────────────────────────────────────────────

def busqueda_solo_imagenes(collection: chromadb.Collection, pregunta: str, top_k: int = 3):
    """Search the collection restricted to image chunks only.

    Uses the same embedding model as the RAG pipeline but applies a
    ``where`` filter so only image chunks can be returned.

    Args:
        collection: ChromaDB collection to query.
        pregunta: Question string to embed and search.
        top_k: Number of results to return.
    """
    import ollama

    prefix = "search_query: "
    try:
        resp = ollama.embeddings(model=MODELO_EMBEDDING, prompt=f"{prefix}{pregunta}")
        query_embedding = resp["embedding"]
    except Exception as e:
        print(f"  ERROR al generar embedding: {e}")
        return

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"format": "image"},
        include=["documents", "metadatas", "distances"],
    )

    ids    = results["ids"][0]
    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    dists  = results["distances"][0]

    print(f"\n{'='*70}")
    print(f"  BÚSQUEDA SOLO IMÁGENES — top {top_k}")
    print(f"  Pregunta: {pregunta}")
    print(f"{'='*70}")

    if not ids:
        print("  (sin resultados)")
        return

    for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        # ChromaDB usa distancia L2; para embeddings normalizados: cos_sim = 1 - dist²/2
        cos_sim = max(0.0, 1.0 - (dist ** 2) / 2)
        print(f"\n  [{i+1}] pag={meta.get('page','?')}  cos_sim={cos_sim:.4f}  l2_dist={dist:.4f}  id={cid}")
        preview = doc[:500] + "..." if len(doc) > 500 else doc
        print(f"  {preview}")


# ─────────────────────────────────────────────
# SECTION 5: PREGUNTA AL PIPELINE RAG
# ─────────────────────────────────────────────

def preguntar_al_rag(collection: chromadb.Collection, pregunta: str):
    """Run the full RAG pipeline and label each retrieved context as text or image.

    Calls ``evaluar_pregunta_rag`` and then inspects the retrieved chunks
    against the collection to find which ones have ``format == "image"``.

    Args:
        collection: ChromaDB collection to search.
        pregunta: Question to run through the RAG pipeline.
    """
    print(f"\n{'='*70}")
    print(f"  PIPELINE RAG COMPLETO")
    print(f"  Pregunta: {pregunta}")
    print(f"{'='*70}")

    print("\n  Ejecutando RAG (puede tardar ~30-60s)...")
    respuesta, contextos = evaluar_pregunta_rag(pregunta, collection)

    if not respuesta:
        print("  (sin respuesta — umbral de relevancia no superado o colección vacía)")
        return

    # Obtener los IDs de chunks de imagen para etiquetar los contextos recuperados
    img_result = collection.get(where={"format": "image"}, include=["documents"])
    img_docs_set = set(img_result.get("documents", []))

    print(f"\n  CONTEXTOS RECUPERADOS ({len(contextos)}):")
    for i, ctx in enumerate(contextos):
        tipo = "🖼  IMAGEN" if ctx in img_docs_set else "📄  TEXTO"
        preview = ctx[:300] + "..." if len(ctx) > 300 else ctx
        print(f"\n  [{i+1}] {tipo}")
        print(f"  {preview}")

    print(f"\n  RESPUESTA DEL MODELO:")
    print(f"  {respuesta}")


# ─────────────────────────────────────────────
# SECTION 6: MAIN
# ─────────────────────────────────────────────

def main():
    """Parse arguments and run the selected mode."""
    parser = argparse.ArgumentParser(
        description="Test de recuperación de chunks de imagen en la colección ragbench"
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="Listar todos los chunks de imagen indexados con sus descripciones OCR",
    )
    parser.add_argument(
        "--search-images",
        action="store_true",
        help="Buscar en ChromaDB filtrando solo chunks de imagen (sin LLM)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Pregunta a lanzar al pipeline RAG completo. "
             "Si no se especifica, se usan las preguntas de ejemplo.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Número de chunks de imagen a devolver en --search-images (default: 3)",
    )
    args = parser.parse_args()

    print(f"\nConectando a ChromaDB: {RAGBENCH_DB_PATH}")
    client = chromadb.PersistentClient(path=RAGBENCH_DB_PATH)

    try:
        collection = client.get_collection(name=RAGBENCH_COLLECTION)
    except Exception:
        print(f"ERROR: colección '{RAGBENCH_COLLECTION}' no encontrada.")
        print("  Ejecuta primero: python evaluation/run_eval_ragbench.py --n-papers 1 --max-q 3 --force-reindex")
        raise SystemExit(1)

    total = collection.count()
    img_count = len(collection.get(where={"format": "image"}, include=[])["ids"])
    print(f"  Total fragmentos: {total}  (imagen: {img_count}, texto: {total - img_count})")

    if img_count == 0:
        print("\nAVISO: no hay chunks de imagen. Activa USAR_EMBEDDINGS_IMAGEN=True en chat_pdfs.py")
        print("  y re-indexa con --force-reindex.")
        raise SystemExit(1)

    # --- Modos de ejecución ---

    if args.list_images:
        listar_chunks_imagen(collection)
        return

    preguntas = [args.question] if args.question else PREGUNTAS_IMAGEN

    for pregunta in preguntas:
        if args.search_images:
            busqueda_solo_imagenes(collection, pregunta, top_k=args.top_k)
        else:
            preguntar_al_rag(collection, pregunta)

        if len(preguntas) > 1:
            print()


if __name__ == "__main__":
    main()
