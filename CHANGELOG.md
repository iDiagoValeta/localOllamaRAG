# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow tags. The measured detail for each release (gate figures, migration notes) lives in its GitHub release; this file is the index.

## [Unreleased]

### Added

- Gitflow: `develop` default branch, `release/` and `hotfix/` branches, protected `main` and `develop`, `Branch policy` check (#479).
- Area, priority and status labels; documentation and release issue and PR templates; `CODEOWNERS` (#479).

## [4.0.1] - 2026-10-20

Bug-fix release on 4.0.0; no change to retrieval or generation on the normal path. Desktop stores in the per-user data directory (#344), atomic `settings.json` (#452), indexing without partial documents (#465), study-mode fixes (#464, #468). [Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v4.0.1).

## [4.0.0] - 2026-10-20

embeddinggemma-2 served by llama.cpp replaces Jina CLIP v2 (every 3.x index must be rebuilt), headless service, OpenAI-compatible backend per model role, 4 GB profile on `rag-lite`. [Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v4.0.0).

## [3.0.0] - 2026-09-12

[Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v3.0.0).

[Unreleased]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v4.0.1...develop
[4.0.1]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v3.0.0
