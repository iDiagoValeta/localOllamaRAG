"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

import chromadb

from rag.cli.display import ui
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# SECTION 7: KEYWORD EXTRACTION AND BM25 LEXICAL SEARCH
# ─────────────────────────────────────────────


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


GENERIC_TERMS_BLACKLIST = {
    "paper", "according", "specific", "specifically", "terms", "allows",
    "allow", "achieve", "system", "model", "approach", "method", "results",
    "three", "two", "one", "first", "second", "following", "based",
    "using", "used", "show", "shows", "provide", "provides", "propose",
    "proposed", "models", "methods", "approaches", "direct",
    "training", "learning", "optimize", "scores", "phases", "primary",
    "compare", "evaluate", "section", "table", "figure", "described",
}


def extraer_keywords(texto: str) -> List[str]:
    """Extract acronyms, technical tokens and content words from a query.

    BM25 drives lexical retrieval now; this only feeds a fallback retrieval
    query (when no LLM sub-queries exist) and the debug metrics. Output is
    ordered most-specific first (acronyms and technical tokens before plain
    words) and deduplicated case-insensitively, so the caller can take the
    leading keywords.

    Args:
        texto: Input text (typically a user query).

    Returns:
        Deduplicated list of keywords, most specific first.
    """
    _STRIP = '¿?.,;:()[]{}"\'-'
    keywords = set()

    # Acronyms (ALL-CAPS tokens) are high-signal; preserve their casing.
    keywords.update(re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', texto))

    palabras = texto.split()

    # Technical tokens: internal capitals (CamelCase), digits, or hyphens.
    terminos_tecnicos = [
        clean
        for palabra in palabras
        if len((clean := palabra.strip(_STRIP))) > 1
        and (any(c.isupper() for c in palabra[1:])
             or any(c.isdigit() for c in palabra)
             or '-' in palabra)
    ]
    keywords.update(terminos_tecnicos)
    keywords.update(t.lower() for t in terminos_tecnicos)

    # Plain content words.
    for palabra in palabras:
        clean = palabra.strip(_STRIP)
        if len(clean) > 3 and clean.lower() not in STOPWORDS:
            keywords.add(clean.lower())

    def _valida(kw: str) -> bool:
        return (len(kw) <= 50 and '?' not in kw
                and not kw.startswith('¿')
                and kw.lower() not in GENERIC_TERMS_BLACKLIST)

    candidatas = sorted(
        (k for k in keywords if _valida(k)),
        key=lambda x: (0 if (x.isupper() or any(c.isupper() for c in x[1:]) or '-' in x) else 1, len(x)),
    )

    seen, resultado = set(), []
    for kw in candidatas:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            resultado.append(kw)
    return resultado


_BM25_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenizar_bm25(texto: str) -> List[str]:
    """Tokenize text for BM25, shared by corpus and query.

    Lowercases, splits on non-alphanumeric boundaries, drops stopwords and
    tokens shorter than 3 characters unless they contain a digit (identifiers
    and metrics such as "q4" are kept). Using the same tokenizer for the
    corpus and the query is required for BM25 term matching to be consistent.

    Args:
        texto: Raw text to tokenize.

    Returns:
        List of normalized tokens.
    """
    tokens = []
    for token in _BM25_TOKEN_RE.findall(texto.lower()):
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not any(c.isdigit() for c in token):
            continue
        tokens.append(token)
    return tokens


# ponytail: cache the tokenized corpus + BM25 index so we don't rescan and
# re-tokenize the whole collection on every query (it only changes on reindex).
# Ceiling: the cache key is (collection name, chunk count, k1, b), so in-place
# edits that keep the same chunk count won't refresh the index; upgrade path =
# version/hash the collection.
_bm25_index_cache: Dict[str, Any] = {"key": None, "bm25": None, "entries": None}


def _construir_indice_bm25(
    collection: chromadb.Collection,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Scan and tokenize the whole collection, returning (BM25 index, entries).

    Returns ``(None, entries)`` when the corpus has no usable tokens.
    """
    corpus_tokens: List[List[str]] = []
    corpus_entries: List[Dict[str, Any]] = []
    total_docs = collection.count()
    batch_size = 100

    for offset in range(0, total_docs, batch_size):
        try:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=['documents', 'metadatas']
            )
        except Exception as e:
            logging.warning(f"Error loading BM25 batch (offset {offset}): {e}")
            continue

        for doc, meta, doc_id in zip(batch['documents'], batch['metadatas'], batch['ids']):
            corpus_tokens.append(_tokenizar_bm25(doc))
            corpus_entries.append({'doc': doc, 'metadata': meta, 'id': doc_id})

    if not corpus_tokens or not any(corpus_tokens):
        return None, corpus_entries

    return cfg.BM25Okapi(corpus_tokens, k1=cfg.BM25_K1, b=cfg.BM25_B), corpus_entries


def _obtener_indice_bm25(
    collection: chromadb.Collection,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Return a cached BM25 index for the collection, rebuilding only on change."""
    # Real collections have a stable name; fall back to object identity so
    # distinct unnamed collections (e.g. test fakes) never share a cache entry.
    coll_key = getattr(collection, "name", None) or id(collection)
    key = (coll_key, collection.count(), cfg.BM25_K1, cfg.BM25_B)
    if _bm25_index_cache["key"] != key:
        bm25, entries = _construir_indice_bm25(collection)
        _bm25_index_cache.update(key=key, bm25=bm25, entries=entries)
    return _bm25_index_cache["bm25"], _bm25_index_cache["entries"]


def busqueda_lexica_bm25(
    pregunta: str,
    collection: chromadb.Collection,
    top_n: int = cfg.N_RESULTADOS_KEYWORD
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Lexical search via Okapi BM25 over the whole chunk collection.

    Implements classic sparse retrieval (Robertson & Zaragoza, 2009): every
    chunk is scored by term frequency, inverse document frequency and length
    normalization, yielding a real relevance ranking instead of a substring
    filter. The tokenized corpus and BM25 index are cached per collection (see
    ``_obtener_indice_bm25``) and only rebuilt when the collection changes.

    Args:
        pregunta: User query.
        collection: ChromaDB collection to search.
        top_n: Maximum number of ranked chunks to return.

    Returns:
        Tuple of (chunks ranked by BM25 score, metrics dict).
    """
    metricas = {
        'disponible': cfg.BM25_AVAILABLE,
        'documentos_indexados': 0,
        'terminos_query': 0,
        'resultados_totales': 0,
        'mejor_score': 0.0,
    }

    query_tokens = _tokenizar_bm25(pregunta)
    metricas['terminos_query'] = len(query_tokens)

    if not query_tokens:
        return [], metricas

    bm25, corpus_entries = _obtener_indice_bm25(collection)
    metricas['documentos_indexados'] = len(corpus_entries)

    if bm25 is None:
        return [], metricas

    scores = bm25.get_scores(query_tokens)

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    resultados = []
    for i in ranked_idx:
        score = float(scores[i])
        if score <= 0:
            break
        entry = corpus_entries[i]
        resultados.append({
            'doc': entry['doc'],
            'metadata': entry['metadata'],
            'distancia': 0.5,
            'id': entry['id'],
            'bm25_score': score,
        })
        if len(resultados) >= top_n:
            break

    metricas['resultados_totales'] = len(resultados)
    metricas['mejor_score'] = resultados[0]['bm25_score'] if resultados else 0.0

    if resultados:
        ui.debug(f"BM25: {len(resultados)} chunks (top score {metricas['mejor_score']:.2f})")
    else:
        ui.debug("BM25: no positive matches")

    if cfg.LOGGING_METRICAS:
        logging.info(
            f"BM25 search: {metricas['documentos_indexados']} docs indexed, "
            f"{metricas['terminos_query']} query terms, "
            f"{metricas['resultados_totales']} results"
        )

    return resultados, metricas




