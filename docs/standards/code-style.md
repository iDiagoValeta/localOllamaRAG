# Code style

Source of truth for layer rules is `AGENTS.md` sections 6 and 7. This file only adds what is missing there.

- `rag/`: Spanish function names, English constants, English docstrings. Imports stdlib, third-party, local. No banners, plain `# Title` only.
- `src/monkeygrab/`: English throughout. Use cases take ports plus `AppConfig` via constructor, never via default argument. Ports are `Protocol`, adapters satisfy them structurally.
- Google-style docstrings with Args, Returns, Raises on every module and public function. One-line docstrings stay on one line.
- Run `ruff check .` before every PR. Fix the lint, do not silence it inline without a why comment.
