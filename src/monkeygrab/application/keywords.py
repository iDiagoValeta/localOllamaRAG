"""Pure text analysis shared by lexical retrieval and query expansion.

Holds the vocabulary and the three pure functions that both the BM25 adapter
and the ``Retrieve`` use case need: the stopword set, the BM25 tokenizer, the
keyword extractor that builds a fallback query variant, and the coherence
check that rejects bag-of-words queries.

These used to exist twice -- once in ``rag/engine/lexical.py`` for the running
pipeline and once copied into the BM25 adapter to keep it free of the legacy
package's import chain. This module is the single home the adapter's comment
anticipated; it depends on nothing but the standard library.
"""

import re
from collections import Counter
from typing import List

# Words carrying no retrieval signal, dropped from both BM25 tokens and
# extracted keywords. Covers the three corpus languages plus the phrasing
# users put in questions ("explica", "following", ...), which would otherwise
# match every chunk equally.
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
# Words common enough in academic prose that using one as a search term
# retrieves everything and discriminates nothing. Applied only to extracted
# keywords, never to BM25 tokens: BM25's own IDF already discounts them.
GENERIC_TERMS_BLACKLIST = {
    "paper", "according", "specific", "specifically", "terms", "allows",
    "allow", "achieve", "system", "model", "approach", "method", "results",
    "three", "two", "one", "first", "second", "following", "based",
    "using", "used", "show", "shows", "provide", "provides", "propose",
    "proposed", "models", "methods", "approaches", "direct",
    "training", "learning", "optimize", "scores", "phases", "primary",
    "compare", "evaluate", "section", "table", "figure", "described",
}


_BM25_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_KEYWORD_STRIP_CHARS = '\u00bf?.,;:()[]{}"\'-'

_ACRONYM_RE = re.compile(r'\b[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00d1]{2,}\b')

# Function words whose presence makes a long query read as a sentence rather
# than a keyword dump. Deliberately separate from STOPWORDS: these are the
# words a coherent question keeps, not the ones retrieval should ignore.
_CONNECTORS = {
    # English
    "the", "a", "an", "is", "are", "how", "what", "why",
    "when", "where", "which", "does", "do", "to", "in", "of",
    "that", "for", "and", "with", "by", "on", "as",
    # Castellano
    "c\u00f3mo", "qu\u00e9", "cu\u00e1l", "cu\u00e1les", "cu\u00e1ndo", "d\u00f3nde", "por",
    "para", "que", "son", "est\u00e1", "entre", "con", "los", "las",
    # Valencia
    "com", "quins", "quines", "quan", "quin", "quina", "per", "que",
}


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 scoring.

    Lowercases, splits on non-alphanumeric boundaries, and drops stopwords
    and tokens shorter than three characters unless they contain a digit, so
    identifiers and metrics such as "q4" survive. The corpus and the query
    must go through this same function or BM25 term matching breaks.

    Args:
        text: Raw text to tokenize.

    Returns:
        Normalized tokens, in order of appearance.
    """
    tokens = []
    for token in _BM25_TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not any(c.isdigit() for c in token):
            continue
        tokens.append(token)
    return tokens


def extract_keywords(text: str) -> List[str]:
    """Extract acronyms, technical tokens and content words from a query.

    Feeds the fallback semantic query variant used when no LLM sub-queries
    are available, and the debug metrics. Ordering is most-specific first --
    acronyms and technical tokens ahead of plain words, shorter ahead of
    longer -- so a caller can simply take the leading keywords.

    Args:
        text: Input text, typically a user query.

    Returns:
        Keywords, most specific first, deduplicated case-insensitively.
    """
    keywords = set()

    # Acronyms (ALL-CAPS tokens) are high-signal; preserve their casing.
    keywords.update(_ACRONYM_RE.findall(text))

    palabras = text.split()

    # Technical tokens: internal capitals (CamelCase), digits, or hyphens.
    terminos_tecnicos = [
        clean
        for palabra in palabras
        if len((clean := palabra.strip(_KEYWORD_STRIP_CHARS))) > 1
        and (any(c.isupper() for c in palabra[1:])
             or any(c.isdigit() for c in palabra)
             or '-' in palabra)
    ]
    keywords.update(terminos_tecnicos)
    keywords.update(t.lower() for t in terminos_tecnicos)

    # Plain content words.
    for palabra in palabras:
        clean = palabra.strip(_KEYWORD_STRIP_CHARS)
        if len(clean) > 3 and clean.lower() not in STOPWORDS:
            keywords.add(clean.lower())

    def _usable(kw: str) -> bool:
        return (len(kw) <= 50 and '?' not in kw
                and not kw.startswith('\u00bf')
                and kw.lower() not in GENERIC_TERMS_BLACKLIST)

    # The trailing `x` breaks ties alphabetically. Without it, equal-priority
    # keywords come out in set-iteration order, which Python derives from
    # randomized string hashes -- the joined fallback query, and therefore
    # what retrieval returns, would differ between runs of the same question.
    candidatas = sorted(
        (k for k in keywords if _usable(k)),
        key=lambda x: (0 if (x.isupper() or any(c.isupper() for c in x[1:]) or '-' in x) else 1, len(x), x),
    )

    seen, resultado = set(), []
    for kw in candidatas:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            resultado.append(kw)
    return resultado


def is_coherent_query(query: str) -> bool:
    """Report whether a query reads as a sentence rather than a bag of words.

    Guards the fallback query variant: a keyword dump embedded as if it were
    a question retrieves noise. Three signals -- unique-word ratio, repetition
    of any single word, and the absence of connectors in a long query -- each
    reject on their own. Queries of fewer than two words are always accepted,
    since none of the signals is meaningful at that length.

    Args:
        query: Candidate query string.

    Returns:
        ``True`` when the query looks coherent.
    """
    words = query.lower().split()
    if len(words) < 2:
        return True

    if len(set(words)) / len(words) < 0.7:
        return False

    if Counter(words).most_common(1)[0][1] >= 3:
        return False

    if len(words) > 8 and not any(w in _CONNECTORS for w in words):
        return False

    return True
