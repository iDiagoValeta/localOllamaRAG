"""Unit tests for monkeygrab.application.keywords.

This vocabulary and these three functions are shared by BM25 indexing and by
the fallback query variant, so a change here moves retrieval results for
every question. The cases below pin the behaviors that are easy to break
without noticing: what counts as a keyword, what the tokenizer keeps, and
what makes a query incoherent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.keywords import (  # noqa: E402
    extract_keywords,
    is_coherent_query,
    tokenize_bm25,
)


class TestTokenizeBm25:
    def test_lowercases_and_splits_on_punctuation_and_underscores(self):
        """Underscores split like punctuation, so identifiers do not survive as
        one term; "is", "all" and "you" are stopwords."""
        assert tokenize_bm25("Attention-Is All_You Need!") == ["attention", "need"]

    def test_drops_stopwords_and_short_tokens(self):
        assert tokenize_bm25("the cat is on a mat") == ["cat", "mat"]

    def test_keeps_short_tokens_that_contain_a_digit(self):
        """Identifiers and metrics like "q4" or "k1" are exactly the terms a
        lexical search is best at, and they fall below the generic minimum
        token length that would otherwise drop them."""
        assert tokenize_bm25("results for q4 and k1") == ["results", "q4", "k1"]

    def test_empty_text_yields_no_tokens(self):
        assert tokenize_bm25("") == []


class TestExtractKeywords:
    def test_acronyms_and_technical_tokens_come_before_plain_words(self):
        keywords = extract_keywords("How does RAG use bge-reranker for ranking documents?")

        # Acronyms and hyphenated/technical tokens sort ahead of plain words.
        assert keywords[0] == "RAG"
        assert "bge-reranker" in keywords
        assert keywords.index("bge-reranker") < keywords.index("documents")

    def test_stopwords_and_generic_academic_terms_are_excluded(self):
        """"paper", "shows", "approach", "allows" and "results" are blacklisted
        as terms that appear in every academic document; the rest are
        stopwords. Only "better" carries any signal, weak as it is."""
        keywords = extract_keywords("the paper shows that this approach allows better results")

        assert keywords == ["better"]

    def test_deduplicates_case_insensitively(self):
        keywords = extract_keywords("Transformer transformer TRANSFORMER")

        assert [k.lower() for k in keywords].count("transformer") == 1

    def test_ordering_is_stable_across_processes(self):
        """Equal-priority keywords must not come out in set-iteration order:
        Python randomizes string hashes per process, which would make the
        fallback query -- and therefore retrieval -- differ between runs."""
        keywords = extract_keywords("delta gamma alpha kappa")

        assert keywords == ["alpha", "delta", "gamma", "kappa"]

    def test_question_marks_and_inverted_marks_never_become_keywords(self):
        assert all("?" not in k and not k.startswith("¿") for k in extract_keywords("¿Qué es RAG?"))


class TestIsCoherentQuery:
    def test_a_natural_question_is_coherent(self):
        assert is_coherent_query("how does the reranker score each candidate")

    @pytest.mark.parametrize("query", ["", "single", "two words"])
    def test_very_short_queries_are_always_accepted(self, query):
        """None of the three signals is meaningful below two words, so a short
        query is accepted rather than guessed at."""
        assert is_coherent_query(query)

    def test_low_unique_word_ratio_is_rejected(self):
        """Below 70% distinct words the query is mostly repetition."""
        assert not is_coherent_query("alpha alpha beta")

    def test_a_word_repeated_three_times_is_rejected(self):
        assert not is_coherent_query("alpha beta gamma delta alpha epsilon alpha zeta theta iota")

    def test_a_long_query_without_connectors_is_rejected(self):
        """More than eight words and not one function word among them is a
        keyword dump, not a question."""
        assert not is_coherent_query(
            "transformer attention encoder decoder embedding reranker corpus retrieval fusion"
        )

    def test_a_long_query_with_connectors_is_accepted(self):
        assert is_coherent_query(
            "how the transformer attention encoder and decoder produce embedding vectors"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
