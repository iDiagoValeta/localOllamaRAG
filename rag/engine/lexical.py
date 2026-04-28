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
# SECTION 7: KEYWORD AND LEXICAL SEARCH
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


def busqueda_por_keywords(
    pregunta: str,
    collection: chromadb.Collection,
    n_results: int = N_RESULTADOS_KEYWORD
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Search by keyword containment using ChromaDB ``where_document``.

    Iterates over extracted keywords (with case variants and hyphen
    alternatives) and collects matching chunks via ``collection.get()``.

    Args:
        pregunta: User query to extract keywords from.
        collection: ChromaDB collection to search.
        n_results: Maximum results per keyword variant.

    Returns:
        Tuple of (matched chunks, metrics dict).
    """
    keywords = extraer_keywords(pregunta)

    metricas = {
        'keywords_totales': len(keywords),
        'keywords_encontradas': 0,
        'resultados_totales': 0,
        'errores': 0
    }

    if not keywords:
        return [], metricas

    resultados_keyword = []
    keywords_encontradas = set()
    ids_vistos = set()

    for keyword_base in keywords[:20]:
        variantes = list({
            keyword_base,
            keyword_base.lower(),
            keyword_base.capitalize(),
        })
        if '-' in keyword_base:
            variantes.append(keyword_base.replace('-', ' '))
            variantes.append(keyword_base.replace('-', ' ').lower())

        for keyword in variantes:
            if len(keyword) < 3:
                continue

            try:
                results = collection.get(
                    where_document={"$contains": keyword},
                    include=['documents', 'metadatas'],
                    limit=n_results
                )

                if results['documents']:
                    for doc, meta, doc_id in zip(
                        results['documents'],
                        results['metadatas'],
                        results['ids']
                    ):
                        if doc_id not in ids_vistos:
                            ids_vistos.add(doc_id)
                            resultados_keyword.append({
                                'doc': doc,
                                'metadata': meta,
                                'distancia': 0.5,
                                'keyword_match': keyword_base,
                                'id': doc_id
                            })
                            keywords_encontradas.add(keyword_base)
            except Exception as e:
                metricas['errores'] += 1
                logging.warning(f"Error searching keyword '{keyword}': {e}")

    metricas['keywords_encontradas'] = len(keywords_encontradas)
    metricas['resultados_totales'] = len(resultados_keyword)

    if keywords_encontradas:
        ui.debug(f"keywords: {', '.join(list(keywords_encontradas)[:10])}")
    else:
        ui.debug("no direct keyword matches")

    if LOGGING_METRICAS:
        logging.info(f"Keyword search: {metricas['keywords_encontradas']}/{metricas['keywords_totales']} found, {metricas['resultados_totales']} results")

    return resultados_keyword, metricas


def busqueda_exhaustiva_texto(
    terminos_criticos: List[str],
    collection: chromadb.Collection,
    max_results: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Batch-scan all documents for critical terms.

    Iterates over the entire collection in batches and returns chunks
    containing the specified terms, sorted by match count.

    Args:
        terminos_criticos: Domain-specific terms to search for.
        collection: ChromaDB collection to scan.
        max_results: Maximum number of results to return.

    Returns:
        Tuple of (matched chunks sorted by match count, metrics dict).
    """
    resultados = []
    total_docs = collection.count()
    batch_size = 100

    metricas = {
        'documentos_escaneados': 0,
        'documentos_con_matches': 0,
        'errores': 0
    }

    for offset in range(0, total_docs, batch_size):
        try:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=['documents', 'metadatas']
            )

            metricas['documentos_escaneados'] += len(batch['documents'])

            for doc, meta, doc_id in zip(
                batch['documents'],
                batch['metadatas'],
                batch['ids']
            ):
                doc_lower = doc.lower()
                matches_encontrados = []

                for termino in terminos_criticos:
                    termino_lower = termino.lower()
                    if termino_lower in doc_lower:
                        matches_encontrados.append(termino)

                if matches_encontrados:
                    metricas['documentos_con_matches'] += 1
                    resultados.append({
                        'doc': doc,
                        'metadata': meta,
                        'id': doc_id,
                        'matches': matches_encontrados,
                        'num_matches': len(matches_encontrados)
                    })
        except Exception as e:
            metricas['errores'] += 1
            logging.warning(f"Error in exhaustive search (offset {offset}): {e}")

    resultados.sort(key=lambda x: x['num_matches'], reverse=True)

    if LOGGING_METRICAS:
        logging.info(f"Exhaustive search: {metricas['documentos_con_matches']} docs with matches out of {metricas['documentos_escaneados']} scanned")

    return resultados[:max_results], metricas


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

