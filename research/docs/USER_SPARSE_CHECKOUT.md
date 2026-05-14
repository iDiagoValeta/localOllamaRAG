# Lighter clone (sparse checkout)

If you only want the **end-user RAG stack** and can skip thesis corpora and all of `research/`, use Git sparse checkout after cloning.

Example (POSIX shell) — adjust branch name if needed:

```bash
git clone --filter=blob:none --sparse https://github.com/iDiagoValeta/localOllamaRAG
cd localOllamaRAG
git sparse-checkout set rag README.md CLAUDE.md pytest.ini
```

To include the default Spanish PDF folder but skip RagBench trees:

```bash
git sparse-checkout set rag README.md CLAUDE.md pytest.ini rag/docs/es
```

**Caveats**

- Sparse checkout hides paths locally; they still exist on the remote. Do not run `research/evaluation/infer.py ragbench-eval` or `infer.py visual` unless you restore the matching `rag/docs/en_ragbench_*` paths and datasets (or clone `research/` fully).
- `git pull` behaviour with sparse patterns is normal Git behaviour; re-run `git sparse-checkout set …` if you add paths later.

PowerShell (Git 2.25+):

```powershell
git clone --filter=blob:none --sparse https://github.com/iDiagoValeta/localOllamaRAG
Set-Location localOllamaRAG
git sparse-checkout set rag README.md CLAUDE.md pytest.ini
```
