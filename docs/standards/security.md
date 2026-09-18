# Security

- Never commit `.env`, `MONKEYGRAB_OPENAI_API_KEY`, or `MONKEYGRAB_HEADLESS_TOKEN`. `.env.example` documents names only.
- Dumps under `DATA_DIR/debug_rag` hold prompts and fragments, never keys. Do not paste dumps into issues.
- Headless binds `127.0.0.1` and auth is optional. Starting without a token prints unauthenticated loopback; exposing that port beyond loopback without a token is unsupported.
- Report suspected secret leaks as `bug` issues with the file path and commit hash, then rotate the secret.
