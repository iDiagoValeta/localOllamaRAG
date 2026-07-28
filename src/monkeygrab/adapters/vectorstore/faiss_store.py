"""FaissVectorStore -- VectorStore adapter over a persisted FAISS index."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment

# Not subclassed from monkeygrab.ports.vector_store.VectorStore: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.

# A row kept in memory per stored chunk: (id, doc text, metadata). Position in
# this list is always the FAISS vector's position in the index -- both grow by
# one, together, on every add() -- so no separate id<->position table needs
# its own persistence; it is rebuilt from this list on load (see SECTION 3).
_Row = Tuple[str, str, ChunkMetadata]


# PERSISTENCE FORMAT

# Bump on any change to the on-disk layout (vector dtype/metric, JSONL schema,
# file names): a store built under an older version must not be silently read
# as the new one -- see _load_or_init's version check.
_FORMAT_VERSION = "1"

_INDEX_FILENAME = "index.faiss"
_META_FILENAME = "meta.jsonl"
_VERSION_FILENAME = "version.txt"

# Deliberately outside the all-or-none rule in _load_or_init: every store
# written before fingerprinting existed lacks this file, and those stores must
# keep loading. Absent means "recipe unknown", which callers treat as stale.
_FINGERPRINT_FILENAME = "fingerprint.txt"


# METADATA CONVERSION


def _metadata_to_dict(metadata: ChunkMetadata) -> Dict[str, Any]:
    """Serialize a ``ChunkMetadata`` into the dict shape written to ``meta.jsonl``.

    JSONL keeps every metadata field present, using ``null`` when unset.
    """
    return {
        "source": metadata.source,
        "page": metadata.page,
        "chunk": metadata.chunk,
        "section_header": metadata.section_header,
        "total_chunks_in_page": metadata.total_chunks_in_page,
        "format": metadata.format,
        "image_width": metadata.image_width,
        "image_height": metadata.image_height,
    }


def _metadata_from_dict(meta: Dict[str, Any]) -> ChunkMetadata:
    """Rebuild a ``ChunkMetadata`` from a ``meta.jsonl`` row's ``metadata`` dict."""
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


# DISK I/O


def _corrupt(store_dir: str, reason: str) -> RuntimeError:
    """Build the one hard-fail message shape used for every corruption case."""
    return RuntimeError(
        f"FAISS store at {store_dir!r} is corrupted ({reason}). "
        "Delete the directory and reindex; a partially written or hand-edited "
        "store cannot be repaired in place."
    )


def _load_or_init(store_dir: str) -> Tuple[Optional["faiss.Index"], List[_Row]]:
    """Load an existing store directory, or recognize a brand-new (empty) one.

    A freshly created directory with none of the three files present is a
    valid empty store (``add`` has simply never been called yet) -- returns
    ``(None, [])``. Any other partial combination of the three files is
    corruption: a complete store is always all-three-or-none, because
    ``_save`` only ever leaves the directory in one of those two states.
    """
    index_path = os.path.join(store_dir, _INDEX_FILENAME)
    meta_path = os.path.join(store_dir, _META_FILENAME)
    version_path = os.path.join(store_dir, _VERSION_FILENAME)
    present = [p for p in (index_path, meta_path, version_path) if os.path.exists(p)]

    if not present:
        return None, []
    if len(present) != 3:
        missing = [
            os.path.basename(p)
            for p in (index_path, meta_path, version_path)
            if p not in present
        ]
        raise _corrupt(store_dir, f"missing {missing} while other store files exist")

    with open(version_path, "r", encoding="utf-8") as f:
        version = f.read().strip()
    if version != _FORMAT_VERSION:
        raise _corrupt(
            store_dir,
            f"built with format version {version!r}, this adapter reads {_FORMAT_VERSION!r}",
        )

    rows: List[_Row] = []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append((row["id"], row["doc"], _metadata_from_dict(row["metadata"])))
    except (json.JSONDecodeError, KeyError) as e:
        raise _corrupt(store_dir, f"{_META_FILENAME} could not be parsed: {e}") from e

    try:
        index = faiss.read_index(index_path)
    except RuntimeError as e:
        raise _corrupt(store_dir, f"{_INDEX_FILENAME} could not be loaded: {e}") from e

    if index.ntotal != len(rows):
        raise _corrupt(
            store_dir,
            f"index has {index.ntotal} vectors but {_META_FILENAME} has {len(rows)} rows",
        )

    return index, rows


# ADAPTER


class FaissVectorStore:
    """Stores and retrieves embedded chunks in a FAISS ``IndexFlatIP`` index.

    Exact (not approximate) search: an ``IndexFlatIP`` scores every stored
    vector against the query rather than using a graph/quantized index. This
    project's corpora are small enough that an ANN index would only add
    approximation error for no measurable speed win (see
    ``docs/design/2026-07-26-monkeygrab-v2.md``, section 3). Vectors are
    L2-normalized before insertion and before querying, which turns the
    index's raw inner product into cosine similarity.

    Persistence layout: four files under ``<path_db>/<collection_name>/``
    (one client path holding several named collections):

    - ``index.faiss`` -- the FAISS index itself (``faiss.write_index``).
    - ``meta.jsonl`` -- one JSON object per stored chunk, in insertion
      order: ``{"id", "doc", "metadata"}``.
    - ``version.txt`` -- the on-disk format version, checked on load so a
      future layout change fails loudly on old stores instead of
      misreading them.
    - ``fingerprint.txt`` -- an opaque recipe fingerprint the caller hands
      this adapter to persist; this store never interprets it (meaning
      lives in ``monkeygrab.application.index_fingerprint``). Deliberately
      optional, and deliberately excluded from ``_load_or_init``'s
      all-or-none presence check: every store written before fingerprinting
      existed lacks this file and must keep loading. Do not fold it into
      that check -- doing so would turn every pre-existing index into a
      hard failure. Its absence means "recipe unknown", which callers
      treat as stale.

    Id-to-position mapping: FAISS indexes by integer position, but chunk ids
    are strings (``Chunk.id``, e.g. ``"paper.pdf_pag2_chunk3"``). Row *i* of
    ``meta.jsonl`` is always FAISS vector *i* -- both are appended to
    together, in the same call, and never reordered -- so an id-position map
    is never itself persisted; it is rebuilt in ``__init__`` by reading
    ``meta.jsonl`` once and enumerating it. Reopening an existing index is
    therefore just replaying that enumeration, not a separate migration step.

    Determinism: ``IndexFlatIP.search`` is exact, but does not document a
    tie-break rule for equal scores, and this adapter must not depend on
    whatever FAISS happens to do internally. Every query re-scores the
    *entire* index (cheap at this project's scale, and already the chosen
    tradeoff above) and re-sorts the candidates by ``(-score, id)`` before
    truncating to ``n_results`` -- so a tie is always broken by ascending
    chunk id, regardless of FAISS's internal ordering or thread count.

    Failure policy: hard-fail. A corrupted directory, a query/insert vector
    whose dimension does not match the index, or a failed write all raise
    with an actionable message -- never an empty result or a silent
    fallback. An id with no match in ``get_by_ids`` is *not* a failure (see
    the ``VectorStore`` port docstring): callers use it for neighbor-context
    expansion, where asking for a chunk past a document's first/last page is
    an expected, not exceptional, outcome.
    """

    def __init__(self, paths: PathsConfig):
        """Args:
            paths: Path config; the store directory is
                ``<paths.path_db>/<paths.collection_name>``, created if absent.
        """
        self._dir = os.path.join(paths.path_db, paths.collection_name)
        os.makedirs(self._dir, exist_ok=True)
        self._index_path = os.path.join(self._dir, _INDEX_FILENAME)
        self._meta_path = os.path.join(self._dir, _META_FILENAME)
        self._version_path = os.path.join(self._dir, _VERSION_FILENAME)
        self._fingerprint_path = os.path.join(self._dir, _FINGERPRINT_FILENAME)

        self._index, self._rows = _load_or_init(self._dir)
        self._id_to_pos: Dict[str, int] = {row[0]: i for i, row in enumerate(self._rows)}

    def add(self, chunk: Chunk, embedding: Sequence[float]) -> None:
        if chunk.id in self._id_to_pos:
            raise ValueError(
                f"chunk id {chunk.id!r} already exists in the FAISS store at "
                f"{self._dir!r}; positions are append-only, re-adding would "
                "desync the id-to-position map"
            )

        vector = np.asarray([list(embedding)], dtype=np.float32)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vector.shape[1])
        elif vector.shape[1] != self._index.d:
            raise ValueError(
                f"embedding has dimension {vector.shape[1]}, FAISS index at "
                f"{self._dir!r} expects {self._index.d}"
            )

        faiss.normalize_L2(vector)
        self._index.add(vector)
        position = len(self._rows)
        row: _Row = (chunk.id, chunk.text, chunk.metadata)
        self._rows.append(row)
        self._id_to_pos[chunk.id] = position
        self._save(row)

    def query(self, embedding: Sequence[float], n_results: int) -> List[Fragment]:
        if self._index is None or self._index.ntotal == 0:
            return []

        vector = np.asarray([list(embedding)], dtype=np.float32)
        if vector.shape[1] != self._index.d:
            raise ValueError(
                f"query embedding has dimension {vector.shape[1]}, FAISS index "
                f"at {self._dir!r} expects {self._index.d}"
            )
        faiss.normalize_L2(vector)

        # Score every stored vector (k = ntotal) so the tie-break re-sort below
        # sees the full candidate set -- see the class docstring's determinism
        # note for why a partial top-k from FAISS would not be safe here.
        k = self._index.ntotal
        scores, positions = self._index.search(vector, k)
        candidates = [
            (float(scores[0][j]), self._rows[positions[0][j]][0], int(positions[0][j]))
            for j in range(k)
        ]
        candidates.sort(key=lambda c: (-c[0], c[1]))

        fragments = []
        for score, _chunk_id, position in candidates[:n_results]:
            _, doc, metadata = self._rows[position]
            # Cosine distance (1 - cosine similarity) on the normalized
            # vectors this store indexes -- not a literal L2 magnitude like
            # "Smaller is nearer", matching the Fragment.distancia contract; ordering is
            # what downstream ranking (RRF, min/max/mean debug stats) uses.
            fragments.append(Fragment(doc=doc, metadata=metadata, distancia=1.0 - score))
        return fragments

    def get_by_ids(self, ids: Sequence[str]) -> List[Fragment]:
        fragments = []
        for chunk_id in ids:
            position = self._id_to_pos.get(chunk_id)
            if position is None:
                continue
            _, doc, metadata = self._rows[position]
            fragments.append(Fragment(doc=doc, metadata=metadata))
        return fragments

    def get_page(self, limit: Optional[int], offset: int) -> List[Fragment]:
        rows = self._rows[offset:] if limit is None else self._rows[offset : offset + limit]
        return [Fragment(doc=doc, metadata=metadata) for _, doc, metadata in rows]

    def count(self) -> int:
        return len(self._rows)

    def read_fingerprint(self) -> Optional[str]:
        """Return the index recipe fingerprint this store recorded, if any.

        Returns:
            The stored value, or ``None`` when the store predates
            fingerprinting or recorded a blank one. ``None`` means *unknown*,
            never *matches*: an index whose recipe cannot be established is
            precisely the one that must not be trusted.
        """
        if not os.path.exists(self._fingerprint_path):
            return None
        with open(self._fingerprint_path, "r", encoding="utf-8") as f:
            return f.read().strip() or None

    def write_fingerprint(self, fingerprint: str) -> None:
        """Record the fingerprint of the recipe that produced this content.

        The value is opaque to this adapter: it persists the string and never
        interprets it. What it means lives in
        ``monkeygrab.application.index_fingerprint``.

        Args:
            fingerprint: The digest to record.
        """
        with open(self._fingerprint_path, "w", encoding="utf-8") as f:
            f.write(fingerprint + "\n")

    def delete_source(self, source: str) -> int:
        positions = [
            position
            for position, (_, _, metadata) in enumerate(self._rows)
            if metadata.source != source
        ]
        removed = len(self._rows) - len(positions)
        if removed == 0:
            return 0

        kept_rows = [self._rows[position] for position in positions]
        if kept_rows:
            vectors = np.vstack([self._index.reconstruct(position) for position in positions])
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(np.asarray(vectors, dtype=np.float32))
            self._index = index
        else:
            self._index = None
        self._rows = kept_rows
        self._id_to_pos = {row[0]: i for i, row in enumerate(self._rows)}
        self._rewrite()
        return removed

    def clear(self) -> None:
        self._index = None
        self._rows = []
        self._id_to_pos = {}
        for path in (
            self._index_path,
            self._meta_path,
            self._version_path,
            self._fingerprint_path,
        ):
            if os.path.exists(path):
                os.remove(path)

    def _save(self, new_row: _Row) -> None:
        """Persist the just-added row: rewrite the index, append to the sidecar.

        The index has no incremental on-disk append, so each ``add`` rewrites
        it whole (written to a temp path, then ``os.replace``d in, so a failed
        write never leaves a half-written ``index.faiss``); the JSONL sidecar
        only ever grows, so it is opened in append mode. If the index write
        fails, this raises before the sidecar is touched, leaving the
        directory exactly as it was before this ``add`` -- still consistent
        and reloadable, just missing the chunk that failed to persist.
        """
        try:
            tmp_path = self._index_path + ".tmp"
            faiss.write_index(self._index, tmp_path)
            os.replace(tmp_path, self._index_path)

            with open(self._meta_path, "a", encoding="utf-8") as f:
                chunk_id, doc, metadata = new_row
                f.write(json.dumps({"id": chunk_id, "doc": doc, "metadata": _metadata_to_dict(metadata)}))
                f.write("\n")

            if not os.path.exists(self._version_path):
                with open(self._version_path, "w", encoding="utf-8") as f:
                    f.write(_FORMAT_VERSION)
        except OSError as e:
            raise RuntimeError(f"failed to persist FAISS store at {self._dir!r}: {e}") from e

    def _rewrite(self) -> None:
        if not self._rows:
            self.clear()
            return
        try:
            index_tmp = self._index_path + ".tmp"
            meta_tmp = self._meta_path + ".tmp"
            version_tmp = self._version_path + ".tmp"
            faiss.write_index(self._index, index_tmp)
            with open(meta_tmp, "w", encoding="utf-8") as f:
                for chunk_id, doc, metadata in self._rows:
                    f.write(json.dumps({
                        "id": chunk_id,
                        "doc": doc,
                        "metadata": _metadata_to_dict(metadata),
                    }))
                    f.write("\n")
            with open(version_tmp, "w", encoding="utf-8") as f:
                f.write(_FORMAT_VERSION)
            os.replace(index_tmp, self._index_path)
            os.replace(meta_tmp, self._meta_path)
            os.replace(version_tmp, self._version_path)
        except OSError as e:
            raise RuntimeError(f"failed to persist FAISS store at {self._dir!r}: {e}") from e
