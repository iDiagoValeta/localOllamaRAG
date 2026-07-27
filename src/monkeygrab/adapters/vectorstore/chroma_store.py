"""ChromaVectorStore -- VectorStore adapter over a ChromaDB collection."""

from typing import Any, Dict, List, Optional, Sequence

import chromadb

from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment

# Not subclassed from monkeygrab.ports.vector_store.VectorStore: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.


# METADATA CONVERSION


def _metadata_to_dict(metadata: ChunkMetadata) -> Dict[str, Any]:
    """Serialize a ``ChunkMetadata`` into the dict shape ``collection.add`` stores.

    Optional fields absent from ``metadata`` are omitted entirely rather than
    written as ``None`` -- Chroma metadata values must be str/int/float/bool,
    and this matches exactly what ``indexar_documentos`` writes today (e.g.
    image chunks never include a ``total_chunks_in_page`` key at all, text
    chunks never include ``image_width``/``image_height``).
    """
    data: Dict[str, Any] = {
        "source": metadata.source,
        "page": metadata.page,
        "chunk": metadata.chunk,
        "section_header": metadata.section_header,
    }
    if metadata.total_chunks_in_page is not None:
        data["total_chunks_in_page"] = metadata.total_chunks_in_page
    if metadata.format is not None:
        data["format"] = metadata.format
    if metadata.image_width is not None:
        data["image_width"] = metadata.image_width
    if metadata.image_height is not None:
        data["image_height"] = metadata.image_height
    return data


def _metadata_from_dict(meta: Dict[str, Any]) -> ChunkMetadata:
    """Rebuild a ``ChunkMetadata`` from a Chroma metadata dict.

    ``chunk`` defaults to 0 on read (``meta.get("chunk", 0)``), the same
    defensive default used throughout ``rag/engine/retrieval.py`` and
    ``rag/engine/chunking.py`` for legacy/incomplete metadata.
    """
    return ChunkMetadata(
        source=meta["source"],
        page=meta["page"],
        chunk=meta.get("chunk", 0),
        total_chunks_in_page=meta.get("total_chunks_in_page"),
        format=meta.get("format"),
        section_header=meta.get("section_header", ""),
        image_width=meta.get("image_width"),
        image_height=meta.get("image_height"),
    )


# ADAPTER


class ChromaVectorStore:
    """Stores and retrieves embedded chunks in a persistent ChromaDB collection.

    The client and collection are created here, at construction time -- not
    handed in from outside -- so an adapter instance owns exactly one
    collection for its whole lifetime, matching ``rag/cli/app.py`` and
    ``rag/web/app.py``'s ``chromadb.PersistentClient(path=...).get_or_create_
    collection(name=...)`` pattern.

    Failure policy: hard-fail. Every method calls straight through to the
    underlying ``chromadb.Collection`` and lets its exceptions propagate --
    unlike ``obtener_documentos_indexados`` today (``rag/engine/indexing.py``),
    which catches any error from its full-collection ``get`` and returns an
    empty list, this adapter never turns a storage failure into a
    quiet empty result.
    """

    def __init__(self, paths: PathsConfig):
        """Args:
            paths: Path config; ``paths.path_db`` and ``paths.collection_name``
                select the on-disk ChromaDB store and collection.
        """
        client = chromadb.PersistentClient(path=paths.path_db)
        self._collection = client.get_or_create_collection(name=paths.collection_name)

    @classmethod
    def wrap_collection(cls, collection: chromadb.Collection) -> "ChromaVectorStore":
        """Adapt a caller-owned Chroma collection without opening a second client.

        CLI/web/eval open the collection before calling ``indexar_documentos``
        and pass it in; writing through that same object keeps their
        ``collection.count()`` / ``get`` views consistent with what was indexed.
        """
        store = cls.__new__(cls)
        store._collection = collection
        return store

    def add(self, chunk: Chunk, embedding: Sequence[float]) -> None:
        self._collection.add(
            ids=[chunk.id],
            embeddings=[list(embedding)],
            documents=[chunk.text],
            metadatas=[_metadata_to_dict(chunk.metadata)],
        )

    def query(self, embedding: Sequence[float], n_results: int) -> List[Fragment]:
        results = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        return [
            Fragment(doc=doc, metadata=_metadata_from_dict(meta), distancia=distancia)
            for doc, distancia, meta in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0]
            )
        ]

    def get_by_ids(self, ids: Sequence[str]) -> List[Fragment]:
        if not ids:
            return []
        result = self._collection.get(ids=list(ids), include=["documents", "metadatas"])
        return [
            Fragment(doc=doc, metadata=_metadata_from_dict(meta))
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    def get_page(self, limit: Optional[int], offset: int) -> List[Fragment]:
        result = self._collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        return [
            Fragment(doc=doc, metadata=_metadata_from_dict(meta))
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    def count(self) -> int:
        return self._collection.count()
