"""SECURITY-001: hierarchy clause in the RAG system prompt.

The generator receives untrusted retrieved text inside <context> tags
(Answer.build_user_message). The system prompt must state explicitly that
this content is data, never instructions, so an injection inside a document
cannot override the user question or exfiltrate data.

Read via AST instead of importing rag.chat_pdfs: importing the facade pulls
the whole retrieval stack (sentence-transformers, FAISS, ...), which the
fast gate deliberately does not install (see tests/conftest.py). Parsing
the constant's literal value verifies the shipped text with stdlib only.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

_PROMPT_PATH = ROOT / "rag" / "chat_pdfs.py"


def _rag_system_prompt() -> str:
    """Return the literal value of SYSTEM_PROMPT_RAG without importing the engine."""
    tree = ast.parse(_PROMPT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT_RAG":
                    return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT_RAG not found in rag/chat_pdfs.py")


def test_rag_system_prompt_declares_context_untrusted():
    prompt = _rag_system_prompt().lower()
    assert "<context>" in prompt
    assert "untrusted" in prompt
    assert "never instructions" in prompt


def test_rag_system_prompt_ignores_instructions_inside_context():
    prompt = _rag_system_prompt().lower()
    assert "ignore any instruction" in prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
