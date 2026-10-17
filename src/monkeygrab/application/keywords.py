"""Pure query-text analysis for retrieval.

The BM25 vocabulary and tokenizer live in the neutral domain module so the
BM25 adapter and this application layer share them without an adapter
importing application code. They are re-exported here for the existing
``rag.chat_pdfs`` compatibility surface.
"""

import re
from collections import Counter
from typing import List

from monkeygrab.domain.lexical_text import STOPWORDS
from monkeygrab.domain.lexical_text import tokenize_bm25 as tokenize_bm25

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


_KEYWORD_STRIP_CHARS = '\u00bf\u00a1?.,;:()[]{}"\'!-'

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


def extract_keywords(text: str) -> List[str]:
    """Extract acronyms, technical tokens and content words from a query.

    Feeds the fallback semantic query variant used when no LLM sub-queries
    are available, and the debug metrics. Ordering is most-specific first --
    acronyms and technical tokens ahead of plain words -- and, within each
    group, the order the query wrote them, so a caller that takes the leading
    keywords does not discard the long terms the question is about.

    Args:
        text: Input text, typically a user query.

    Returns:
        Keywords, most specific first, deduplicated case-insensitively.
    """
    palabras = text.split()

    # Acronyms (ALL-CAPS tokens) are high-signal; preserve their casing.
    tecnicos: List[str] = _ACRONYM_RE.findall(text)

    # Technical tokens: internal capitals (CamelCase), digits, or hyphens.
    # The test runs on the stripped token: on the raw word, a leading "¿" or
    # "¡" shifts the slice that looks for an internal capital, which turns a
    # function word such as "Se" or "No" into a false technical candidate.
    for palabra in palabras:
        clean = palabra.strip(_KEYWORD_STRIP_CHARS)
        if len(clean) < 2:
            continue
        if not (
            any(c.isupper() for c in clean[1:])
            or any(c.isdigit() for c in clean)
            or '-' in clean
        ):
            continue
        # Technical-looking or not, a stopword carries no retrieval signal.
        if clean.lower() in STOPWORDS:
            continue
        tecnicos.append(clean)
        tecnicos.append(clean.lower())

    # Plain content words, in the order the query wrote them.
    palabras_planas: List[str] = []
    for palabra in palabras:
        clean = palabra.strip(_KEYWORD_STRIP_CHARS)
        if len(clean) > 3 and clean.lower() not in STOPWORDS:
            palabras_planas.append(clean.lower())

    def _usable(kw: str) -> bool:
        return (len(kw) <= 50 and '?' not in kw
                and not kw.startswith('\u00bf')
                and kw.lower() not in GENERIC_TERMS_BLACKLIST)

    candidatas = [k for k in tecnicos + palabras_planas if _usable(k)]

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
