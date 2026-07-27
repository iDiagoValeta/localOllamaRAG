"""Chat adapters -- ChatModel and ModelUnloader implementations.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- ollama_chat.py            OllamaChatModel     -- ollama.chat (generate) + raw HTTP (stream)
#  +-- ollama_model_unloader.py  OllamaModelUnloader  -- keep_alive=0 VRAM reclaim across roles
#
# ─────────────────────────────────────────────
"""
