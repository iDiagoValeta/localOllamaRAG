"""The study context budget must leave room for the longest answer the role can give.

``_calculate_context_char_budget`` lives in the core and cannot import the
wiring layer that owns the generator's ``num_predict`` cap, so the two numbers
are tied here instead (issue #373). Before the fix it reserved 2048 tokens
against a 4096 cap.
"""

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.application import study
from monkeygrab.config.app_config import AppConfig
from rag.engine import wiring


def test_study_reserve_covers_the_generator_output_cap():
    assert study._OUTPUT_RESERVE_TOKENS >= wiring.RAG_SAMPLING_OPTIONS["num_predict"]


def test_budget_leaves_the_reserve_free_in_the_context_window():
    config = AppConfig.from_env()
    num_ctx = config.models.ollama.rag_num_ctx

    chars = study._calculate_context_char_budget(config)

    assert chars / 3.0 + study._OUTPUT_RESERVE_TOKENS <= num_ctx
