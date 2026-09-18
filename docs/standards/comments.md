# Comments

Comments explain why, not what. If a comment restates the line below it, delete it.

Good, records a constraint:

```python
# BGE reranker needs the full 200 candidates; trimming here drops two gold cases.
candidates = rerank(candidates[:200])
```

Bad, restates the code:

```python
# Loop over candidates.
for c in candidates:
```

Python follows PEP 257. TSX follows the same why rule. Never commit commented-out code; delete it, git remembers.
