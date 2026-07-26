# Diagnostics

Manual probes for Ollama behaviour. Run them by hand when a model starts
answering oddly; they are not part of the test suite.

They used to live under `tests/` with `test_` prefixes, which made them look
like coverage they never provided: neither file defines a test function, so
pytest collected zero tests from both while still paying for their imports.

| Script | What it answers |
|---|---|
| `ollama_aux_nothink.py` | Does this model actually suppress its reasoning trace when asked to? Some models fill `message.thinking` and leave `content` empty. |
| `ollama_stream_nothink.py` | Same question for the streaming endpoint, where the trace can arrive interleaved with the answer. |

Both need a live Ollama at `localhost:11434`. Use `--help` for options.

```bash
python tools/diagnostics/ollama_aux_nothink.py --model gemma4:e4b
```
