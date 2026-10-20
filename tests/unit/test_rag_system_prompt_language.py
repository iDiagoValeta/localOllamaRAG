"""The RAG prompt answers in the language of the question, not of the context (issue #378).

A Spanish question about an English PDF used to be answered in English because
the prompt said "same language as the context". Read via AST for the same
reason as test_rag_system_prompt_hierarchy.py: importing the facade pulls the
retrieval stack the fast gate does not install.
"""

import ast
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "rag" / "chat_pdfs.py"


def _rag_system_prompt() -> str:
    tree = ast.parse(_PROMPT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT_RAG":
                    return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT_RAG not found in rag/chat_pdfs.py")


def test_language_follows_the_question():
    prompt = _rag_system_prompt().lower()
    assert "language of the user's question" in prompt


def test_language_no_longer_follows_the_context():
    assert "same language as the context" not in _rag_system_prompt().lower()
