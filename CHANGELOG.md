# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow tags. The measured detail for each release (gate figures, migration notes) lives in its GitHub release; this file is the index.

## [Unreleased]

### Changed

- The lint job fails on a `noqa` directive that silences nothing and on commented-out code (#524).
- The frontend's CI fails on unused locals, parameters, exports and dependencies, and the ones it found are gone (#525).
- `CLAUDE.md` is the one line `@AGENTS.md`, and `.agents/skills` points Codex at the same runbooks; the pointer files under `.claude/` and `.codex/` are gone (#528).
- The fast gate fails on Python code that only tests reach; the dead code it found is gone (#526).
- The fast gate fails if `rag/engine` imports the web or headless layers, a module imports a private name from another top-level package (`monkeygrab`, `rag`, `harness`, `tools`), an adapter is built outside the composition root, or a new import cycle appears (#527).

### Fixed

- The frontend type-check sees React's types; before, every React import and JSX element was typed `any`, so it verified almost nothing in the components. It now runs on TypeScript 7 (#532).

### Security

- The frontend build no longer pulls `source-map-js` 1.2.1, inside a high event-loop denial-of-service advisory; 1.2.2 is the only copy left (#540).

## [4.1.0] - 2026-10-21

[Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v4.1.0).

Reindexing required: #366 and #426 change what an index stores, so every existing store reads stale until it is reindexed.

### Security

- Web API rejects Host and Origin outside loopback, and the headless service can confine `docs_folder` and `data_dir` to the roots in `MONKEYGRAB_HEADLESS_ALLOWED_ROOTS`, failing closed (#346).

### Changed

- RAG answers follow the language of the question, not the language of the context (#378).
- RAG debug dumps are off by default, keeping question and context text out of the debug directory (#399).
- Study context budget reserves the generator's real output, 4096 tokens, leaving about 34k characters of document at a 16k context (#373).
- The first chunk of a page is expanded with the tail of the previous page, so context that starts mid-sentence keeps its start (#397).
- Grader accepts typesetting variants of a correct answer (`\%`, `\,`, `^2`, thousands separators); gate figures recorded before 2026-10-21 are not comparable, see `docs/model-history.md` (#367).

### Fixed

- Settings panel reports why Ollama failed to start instead of staying on "stopped" (#474).
- Closed sidebar leaves the tab order and its buttons have accessible names (#476).
- Algorithm and code blocks are indexed instead of dropped, and figure placeholders use 1-based page numbers; every store must be reindexed (#366, #375).
- The MinerU version is part of the index fingerprint, so every existing store reads stale once (#426).
- README llama-server command raises the physical batch to 2048, which stops 500s on long chunks during indexing (#487).

### Added

- Gitflow: `develop` default branch, `release/` and `hotfix/` branches, protected `main` and `develop`, `Branch policy` check (#479).
- The eval gate records per case whether RECOMP synthesis was used or fell back to raw context, and counts the fallbacks in the run summary (#497).
- Area, priority and status labels; documentation and release issue and PR templates; `CODEOWNERS` (#479).

## [4.0.1] - 2026-10-20

Bug-fix release on 4.0.0; no change to retrieval or generation on the normal path. Desktop stores in the per-user data directory (#344), atomic `settings.json` (#452), indexing without partial documents (#465), study-mode fixes (#464, #468). [Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v4.0.1).

## [4.0.0] - 2026-10-20

embeddinggemma-2 served by llama.cpp replaces Jina CLIP v2 (every 3.x index must be rebuilt), headless service, OpenAI-compatible backend per model role, 4 GB profile on `rag-lite`. [Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v4.0.0).

## [3.0.0] - 2026-09-12

[Release notes](https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v3.0.0).

[Unreleased]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v4.1.0...develop
[4.1.0]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v4.0.1...v4.1.0
[4.0.1]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/iDiagoValeta/localOllamaRAG/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/iDiagoValeta/localOllamaRAG/releases/tag/v3.0.0
