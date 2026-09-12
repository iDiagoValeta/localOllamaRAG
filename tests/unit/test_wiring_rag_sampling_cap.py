"""The rag role's generation is capped in tokens, and the cap clears every
real answer on record.

Decided 2026-09-12 after issue #249: with ``num_predict = -1`` one quiz call
produced ~96,000 tokens and held Ollama's only slot for fifteen minutes,
and in the web chat nothing but the user stops such a generation. Across the
3,167 answers the model campaign measured, the 99th percentile was 2,138
tokens and the longest 8,883 -- every one of the long ones a failing
answer. 4,096 truncates none of the real ones and cuts a runaway in about
forty seconds on this card.

Lives here rather than under tests/characterization because it pins a
decision, not observed behaviour to be preserved as-is.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from rag.engine import wiring  # noqa: E402

# The 99th percentile of answer length across the 2026-09-11 campaign
# (3,167 answers; medians 27-335 tokens). The cap sits nearly twice above it
# and far below a runaway.
_P99_ANSWER_TOKENS = 2_138


def test_rag_generation_is_capped_above_every_real_answer_on_record():
    cap = wiring.RAG_SAMPLING_OPTIONS["num_predict"]
    assert cap > 0, "an unbounded rag generation is what #249 measured at 96,000 tokens"
    assert cap >= 1.5 * _P99_ANSWER_TOKENS
    assert cap == 4096


def test_the_cap_reaches_the_built_rag_model():
    model = wiring.rag_chat_model(AppConfig())
    assert model._options()["num_predict"] == 4096
