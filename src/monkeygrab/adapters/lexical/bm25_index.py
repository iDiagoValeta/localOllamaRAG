"""Bm25LexicalIndex -- LexicalIndex adapter over rank_bm25's Okapi BM25.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- SECTION 1: TOKENIZATION  -- stopwords + tokenizer (corpus and query)
#  +-- SECTION 2: ADAPTER       -- index build/cache + search
#
# ─────────────────────────────────────────────
"""

import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.vector_store import VectorStore

# Not subclassed from monkeygrab.ports.lexical_index.LexicalIndex: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.


# ─────────────────────────────────────────────
# SECTION 1: TOKENIZATION
# ─────────────────────────────────────────────

# Copied verbatim from rag/engine/lexical.py's STOPWORDS / _tokenizar_bm25
# rather than imported: that module binds `cfg = get_runtime()` at import
# time, which resolves to `import rag.chat_pdfs` -- pulling the whole
# service-locator module (chromadb, ollama, sentence_transformers,
# pymupdf4llm, dotenv, ...) into an adapter meant to depend on nothing but
# rank_bm25 and the injected VectorStore port. Neither STOPWORDS nor the
# tokenizer function actually reads `cfg`, so this is pure duplication of
# infrastructure-free logic to avoid an infrastructure-heavy import path;
# a future application-layer text-processing module is the right home to
# unify this with the original once rag/ stops owning it.
STOPWORDS = {
    # Castellano
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'a', 'al',
    'por', 'para', 'con', 'sin', 'sobre', 'entre', 'hacia', 'desde', 'hasta', 'durante', 'mediante',
    'según', 'contra', 'que', 'quien', 'cual', 'cuales', 'cuyo', 'cuya', 'cuyos', 'cuyas',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
    'esto', 'eso', 'aquello', 'y', 'o', 'pero', 'sino', 'aunque', 'si', 'porque', 'cuando', 'donde',
    'como', 'más', 'menos', 'muy', 'poco', 'mucho', 'algo', 'nada', 'todo', 'toda', 'todos', 'todas',
    'cada', 'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas',
    'es', 'son', 'está', 'están', 'era', 'eran', 'fue', 'fueron', 'ser', 'estar',
    'hay', 'había', 'han', 'haber', 'tiene', 'tienen', 'tenía', 'tener', 'puede', 'pueden', 'poder',
    'se', 'me', 'te', 'nos', 'os', 'le', 'lo', 'les', 'su', 'sus', 'mi', 'tu', 'nuestro', 'vuestro',
    'aquí', 'ahí', 'allí', 'así', 'ya', 'también', 'solo', 'sólo', 'siempre', 'nunca', 'después', 'antes',
    'explica', 'explicar', 'describe', 'describir', 'detalla', 'detallar', 'indica', 'indicar',
    'respuesta', 'pregunta', 'preguntas', 'siguientes', 'siguiente', 'puntos', 'punto', 'ejemplo',
    'manera', 'forma', 'tipo', 'tipos', 'parte', 'partes', 'primer', 'primera', 'segundo', 'segunda',
    'tercer', 'tercera', 'uno', 'dos', 'tres', 'cuáles', 'cómo', 'qué', 'podrías', 'decirme', 'puedes',
    'principales', 'llaman', 'tanto', 'tan', 'fue', 'sido', 'siendo', 'hacer', 'ir',
    # English
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'here', 'there', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
    'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'will', 'this', 'that',
    'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her',
    'what', 'which', 'who', 'whom', 'whose',
    # Valencian
    'els', 'les', 'uns', 'unes', 'dels', 'als',
    'per', 'per a', 'amb', 'sense', 'des de', 'fins a', 'fins', 'dins', 'envers',
    'que', 'qui', 'qual', 'quals', 'què', 'aquest', 'aquesta', 'aquests', 'aquestes',
    'aquell', 'aquella', 'aquells', 'aquelles', 'això', 'allò', 'i', 'o', 'però', 'sinó',
    'perquè', 'quan', 'com', 'més', 'menys', 'molt', 'poc', 'res', 'tot', 'tota', 'tots', 'totes',
    'cada', 'altre', 'altra', 'altres', 'mateix', 'mateixa', 'mateixos', 'mateixes',
    'és', 'són', 'està', 'estan', 'era', 'eren', 'ha', 'han', 'hi ha', 'hi havia',
    'pot', 'poden', 'ser', 'estar', 'tenir', 'fer', 'anar', 'dir', 'veure',
    'aquí', 'allà', 'així', 'ja', 'també', 'només', 'sempre', 'mai', 'després', 'abans',
    'explica', 'explicar', 'descriu', 'detalla', 'detallar', 'indica', 'indicar',
    'resposta', 'pregunta', 'preguntes', 'següents', 'següent', 'punts', 'punt', 'exemple',
    'manera', 'forma', 'tipus', 'part', 'parts', 'primer', 'primera', 'segon', 'segona',
    'tercer', 'tercera', 'un', 'dos', 'tres', 'quins', 'quines', 'quin', 'quina',
}

_BM25_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Tokenize for BM25: lowercase, split on non-alphanumeric, drop stopwords
    and sub-3-char tokens unless they contain a digit (keeps identifiers like
    "q4"). Shared by corpus and query, as BM25 term matching requires."""
    tokens = []
    for token in _BM25_TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not any(c.isdigit() for c in token):
            continue
        tokens.append(token)
    return tokens


# ─────────────────────────────────────────────
# SECTION 2: ADAPTER
# ─────────────────────────────────────────────


class Bm25LexicalIndex:
    """Ranks chunks from an injected ``VectorStore`` against a query via Okapi BM25.

    The ``LexicalIndex`` port has no "index these chunks" step and no
    ``collection`` parameter on ``search`` -- building and keeping the index in
    sync with the corpus is explicitly an adapter concern (see the port
    docstring). This adapter gets its corpus by scanning a ``VectorStore``
    port instance (``get_page``/``count``), so it depends on the storage
    *port*, not on ChromaDB specifically -- swapping the ``VectorStore``
    adapter (e.g. Chroma -> FAISS) swaps the BM25 corpus source for free.

    Caching: ``rag/engine/lexical.py:170-217`` caches the built index in a
    module-level global (``_bm25_index_cache``), shared and invalidated
    across every caller in the process. Here the same ``(count, k1, b)``
    cache key lives on ``self``, so two ``Bm25LexicalIndex`` instances (two
    corpora, or two tests) never share or invalidate each other's index.

    Failure policy: hard-fail for anything that reaches the underlying
    ``VectorStore`` (its own hard-fail policy applies transitively --
    nothing here catches its exceptions). An empty result for a query with
    no positive BM25 match is not a failure, matching
    ``busqueda_lexica_bm25`` today.
    """

    def __init__(self, vector_store: VectorStore, retrieval: RetrievalConfig):
        """Args:
            vector_store: ``VectorStore``-conforming instance whose stored
                chunks make up the BM25 corpus.
            retrieval: Retrieval config; only ``bm25_k1``/``bm25_b`` are read.
        """
        self._vector_store = vector_store
        self._k1 = retrieval.bm25_k1
        self._b = retrieval.bm25_b
        self._cache_key: Optional[Tuple[int, float, float]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._entries: List[Fragment] = []

    def _ensure_index(self) -> None:
        """Rebuild the BM25 index iff the corpus size or BM25 parameters changed."""
        key = (self._vector_store.count(), self._k1, self._b)
        if key == self._cache_key:
            return

        entries = self._vector_store.get_page(limit=None, offset=0)
        corpus_tokens = [_tokenize(entry.doc) for entry in entries]

        self._entries = entries
        self._bm25 = BM25Okapi(corpus_tokens, k1=self._k1, b=self._b) if any(corpus_tokens) else None
        self._cache_key = key

    def search(self, query: str, top_n: int) -> List[Fragment]:
        """Rank stored chunks against ``query`` by Okapi BM25 relevance.

        Args:
            query: User query text (tokenized internally).
            top_n: Maximum number of ranked results to return.

        Returns:
            Fragments ranked best-first (``score_keyword``/``score_final``
            left at their defaults -- fusion happens outside this port), or
            an empty list when the query has no tokens or no chunk scores
            above zero.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        self._ensure_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Fragment] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                break
            results.append(self._entries[i])
            if len(results) >= top_n:
                break
        return results
