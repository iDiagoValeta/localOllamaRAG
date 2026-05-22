"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration remains owned by rag.chat_pdfs and is synchronized before each
function call so web/API toggles and test monkeypatches keep working.
"""

import base64
import contextlib
import io
import json
import logging
import os
import re
import requests
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import ollama
from pypdf import PdfReader

from rag.cli.display import ui
from rag.engine.runtime import sync_runtime_globals


def _sync_runtime_globals() -> None:
    sync_runtime_globals(globals())


_sync_runtime_globals()
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


TERMINOS_EXPANSION = {}

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
    """Extract acronyms, bigrams, parenthesized terms, and technical tokens.

    Filters stopwords, deduplicates case-insensitively, and removes
    single-word tokens already covered by multi-word n-grams.

    Args:
        texto: Input text (typically a user query).

    Returns:
        Deduplicated list of keywords sorted by specificity.
    """
    keywords = set()

    siglas = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', texto)
    keywords.update(s for s in siglas if len(s) >= 2)

    parentesis = re.findall(r'\(([^)]+)\)', texto)
    for term in parentesis:
        term_clean = term.strip()
        if len(term_clean) > 2 and term_clean.lower() not in STOPWORDS:
            for sub in term_clean.split(','):
                sub_clean = sub.strip()
                if len(sub_clean) > 2 and sub_clean.lower() not in STOPWORDS and len(sub_clean) <= 50:
                    keywords.add(sub_clean)
            if len(term_clean) <= 50:
                keywords.add(term_clean)

    palabras = texto.split()
    for i in range(len(palabras) - 1):
        p1 = palabras[i].strip('¿?.,;:()[]{}"\'-')
        p2 = palabras[i + 1].strip('¿?.,;:()[]{}"\'-')

        if (len(p1) > 3 and len(p2) > 3 and
            p1.lower() not in STOPWORDS and p2.lower() not in STOPWORDS):
            bigrama = f"{p1} {p2}"
            keywords.add(bigrama.lower())

    for palabra in palabras:
        clean = palabra.strip('¿?.,;:()[]{}"\'-')
        if len(clean) > 3 and clean.lower() not in STOPWORDS:   # lowered from 4 to 3
            keywords.add(clean.lower())

    terminos_tecnicos = [
        palabra.strip('¿?.,;:()[]{}"\'-')
        for palabra in palabras
        if (any(c.isupper() for c in palabra[1:]) or
           any(c.isdigit() for c in palabra) or
           '-' in palabra) and
        len(palabra.strip('¿?.,;:()[]{}"\'-')) > 1
    ]
    keywords.update(t.lower() for t in terminos_tecnicos)
    keywords.update(terminos_tecnicos)

    keywords_expandidas = set(keywords)
    for kw in list(keywords):
        kw_lower = kw.lower()
        if kw_lower in TERMINOS_EXPANSION:
            keywords_expandidas.update(TERMINOS_EXPANSION[kw_lower])


    def _es_keyword_valida(kw: str) -> bool:
        if len(kw) > 50:
            return False
        if kw.strip().startswith('¿') or '?' in kw:
            return False
        return True

    keywords_filtradas = {k for k in keywords_expandidas if _es_keyword_valida(k)}

    seen_lower = set()
    deduped = []
    for kw in sorted(keywords_filtradas,
                     key=lambda x: (
                         0 if (x.isupper() or any(c.isupper() for c in x[1:]) or '-' in x) else 1,
                         len(x)
                     )):
        kw_lower = kw.lower()
        if kw_lower not in seen_lower:
            seen_lower.add(kw_lower)
            deduped.append(kw)

    ngrams = [kw for kw in deduped if len(kw.split()) >= 2]
    ngram_words = set()
    for ng in ngrams:
        for word in ng.lower().split():
            ngram_words.add(word)

    resultado = []
    for kw in deduped:
        if len(kw.split()) == 1 and kw.lower() in ngram_words:
            continue
        if len(kw.split()) == 1 and kw.lower() in GENERIC_TERMS_BLACKLIST:
            continue
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


def busqueda_lexica_bm25(
    pregunta: str,
    collection: chromadb.Collection,
    top_n: int = N_RESULTADOS_KEYWORD
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Lexical search via Okapi BM25 over the whole chunk collection.

    Implements classic sparse retrieval (Robertson & Zaragoza, 2009): every
    chunk is scored by term frequency, inverse document frequency and length
    normalization, yielding a real relevance ranking instead of a substring
    filter. The BM25 index is rebuilt per query by scanning the collection in
    batches.

    Args:
        pregunta: User query.
        collection: ChromaDB collection to search.
        top_n: Maximum number of ranked chunks to return.

    Returns:
        Tuple of (chunks ranked by BM25 score, metrics dict).
    """
    metricas = {
        'disponible': BM25_AVAILABLE,
        'documentos_indexados': 0,
        'terminos_query': 0,
        'resultados_totales': 0,
        'mejor_score': 0.0,
    }

    query_tokens = _tokenizar_bm25(pregunta)
    metricas['terminos_query'] = len(query_tokens)

    if not BM25_AVAILABLE:
        ui.debug("BM25 unavailable (rank-bm25 not installed)")
        return [], metricas
    if not query_tokens:
        return [], metricas

    total_docs = collection.count()
    corpus_tokens: List[List[str]] = []
    corpus_entries: List[Dict[str, Any]] = []
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

    metricas['documentos_indexados'] = len(corpus_tokens)

    if not corpus_tokens or not any(corpus_tokens):
        return [], metricas

    bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)
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

    if LOGGING_METRICAS:
        logging.info(
            f"BM25 search: {metricas['documentos_indexados']} docs indexed, "
            f"{metricas['terminos_query']} query terms, "
            f"{metricas['resultados_totales']} results"
        )

    return resultados, metricas


# ─────────────────────────────────────────────



def _with_runtime_sync(func):
    def wrapper(*args, **kwargs):
        _sync_runtime_globals()
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    return wrapper


for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", None) == __name__ and _name not in {
        "_sync_runtime_globals", "_with_runtime_sync"
    }:
        globals()[_name] = _with_runtime_sync(_obj)

_sync_runtime_globals()

