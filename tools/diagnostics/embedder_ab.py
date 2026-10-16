"""A/B retrieval benchmark of two embedders over the same stored chunks.

Compares ``jinaai/jina-clip-v2`` (the previous default, whose vectors are
already in the on-disk FAISS stores) against ``google/embeddinggemma-2`` served
by a local ``llama-server``, on an identical set of chunks and the same gold
questions. Dense retrieval is scored alone and fused with the project's own
BM25 branch (weighted RRF), and both are reported as Recall@k and MRR at the
overall, per-source, per-language and per-case-type level.

Arm A (``jina``) reuses the stored ``index.faiss`` vectors verbatim -- no
document is re-embedded -- and only embeds the questions, through the project's
own ``JinaClipEmbedder`` (which drives the isolated ``.venv-mineru`` worker).
Arm B (``gemma512``) and arm C (``gemma768``) embed every document with
``LlamaServerEmbedder`` (Matryoshka widths 512 and 768) and build an in-memory
``IndexFlatIP`` per store. Document vectors are cached under
``/tmp/embedder_ab/`` so a rerun skips the (CPU-bound) re-embedding.

Image chunks are matched to the MinerU extraction cache by ``(page, img_idx)``,
where ``img_idx = metadata.chunk - 10000`` (the offset the indexer applies to
image chunks); the matching mirrors ``_content_list_to_images``, including its
skip of unreadable files. Matched images are embedded as pixels plus their raw
caption (empty for a caption-less figure, exactly as the indexer feeds the
production embedder); unmatched rows fall back to caption-only text and are
counted in the report.

The script is read-only over the repo: it never writes inside it, only under the
output directory (default ``/tmp/embedder_ab/``).

Usage::

    .venv/bin/python tools/diagnostics/embedder_ab.py --limit 10 --no-images
    .venv/bin/python tools/diagnostics/embedder_ab.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder  # noqa: E402
from monkeygrab.adapters.embedding.llama_server_embedder import LlamaServerEmbedder  # noqa: E402
from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex  # noqa: E402
from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore  # noqa: E402
from monkeygrab.application.rrf_fusion import fuse_semantic_and_keyword  # noqa: E402
from monkeygrab.composition import _isolated_python  # noqa: E402
from monkeygrab.config.paths import PathsConfig  # noqa: E402
from monkeygrab.config.retrieval import RetrievalConfig  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration

DEFAULT_BASE_URL = "http://127.0.0.1:8091"
DEFAULT_OUT_DIR = Path("/tmp/embedder_ab")
GOLD_FILE = REPO_ROOT / "tests" / "eval" / "gold_cases.jsonl"
MINERU_CACHE = REPO_ROOT / ".mineru_cache"

# Gold ``source`` value -> on-disk store directory (same mapping run_eval.py uses).
STORE_DIRS: Dict[str, Path] = {
    "corpus": REPO_ROOT / "rag/vector_db/dev_docs_jina_clip/docs_dev_docs",
    "corpus_es": REPO_ROOT / "rag/vector_db/dev_docs_es_jina_clip/docs_dev_docs_es",
    "corpus_ca": REPO_ROOT / "rag/vector_db/dev_docs_ca_jina_clip/docs_dev_docs_ca",
    "arxiv": REPO_ROOT / "rag/vector_db/blind_docs_jina_clip/docs_blind_docs",
}

# Image chunks carry ``chunk = 10000 + img_idx`` (see index_corpus.py).
IMAGE_CHUNK_OFFSET = 10_000
_FORMAT_TO_KIND = {"image": "image", "table": "table", "markdown": "text"}

DENSE_K = 50
HYBRID_DENSE_K = 80
HYBRID_BM25_K = 40
RECALL_KS = (1, 5, 8, 20, 50)
MRR_KS = (10, 50)
EMBED_WORKERS = 4


@dataclass(frozen=True)
class Arm:
    """One embedding configuration under test."""

    name: str
    kind: str  # "jina" or "gemma"
    dimensions: int


ARMS: Dict[str, Arm] = {
    "jina": Arm("jina", "jina", 512),
    "gemma512": Arm("gemma512", "gemma", 512),
    "gemma768": Arm("gemma768", "gemma", 768),
}


@dataclass
class StoreData:
    """A loaded FAISS store: fragments in stored order plus a BM25 index."""

    key: str
    store_dir: Path
    fragments: List[Fragment]
    index: faiss.Index
    bm25: Bm25LexicalIndex
    id_to_row: Dict[str, int]
    image_paths: Dict[int, Tuple[str, str]]


# ---------------------------------------------------------------------------
# Loading


def log(message: str) -> None:
    """Print a progress line to stderr (the report owns stdout)."""
    print(message, file=sys.stderr, flush=True)


def load_gold_cases(limit: int) -> List[dict]:
    """Load the non-study gold cases that carry a question and verified pages.

    Args:
        limit: Keep at most this many cases (0 or less keeps them all).

    Returns:
        The selected case dictionaries, in file order.
    """
    cases: List[dict] = []
    with GOLD_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if str(case.get("case_type", "")).startswith("study_"):
                continue
            if not case.get("question") or not case.get("verified_pages"):
                continue
            cases.append(case)
    if limit > 0:
        cases = cases[:limit]
    return cases


def build_image_lookup(stem: str) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """Map ``(page, img_idx)`` to ``(image_path, raw_caption)`` for one source.

    Mirrors ``MineruExtractor._content_list_to_images``: only ``image``/``chart``
    blocks count, files that cannot be read are skipped (so they do not shift
    the per-page index the way the indexer did not count them either), and the
    caption is the space-joined caption list.

    Args:
        stem: PDF stem (matches the gold case ``paper`` field).

    Returns:
        Lookup keyed by zero-based page and image index on that page.
    """
    candidates = sorted(MINERU_CACHE.glob(f"{stem}-*/{stem}/*/*_content_list.json"))
    if not candidates:
        candidates = sorted(MINERU_CACHE.glob(f"{stem}-*/**/{stem}_content_list.json"))
    if not candidates:
        return {}
    content_list = candidates[0]
    blocks = json.loads(content_list.read_text(encoding="utf-8"))
    auto_dir = content_list.parent

    lookup: Dict[Tuple[int, int], Tuple[str, str]] = {}
    counters: Dict[int, int] = {}
    for block in blocks:
        block_type = block.get("type")
        if block_type not in ("image", "chart"):
            continue
        rel_path = block.get("img_path")
        page_idx = block.get("page_idx")
        if not rel_path or page_idx is None:
            continue
        image_path = auto_dir / rel_path
        if not image_path.is_file():
            continue  # same skip the extractor applies
        index = counters.get(int(page_idx), 0)
        counters[int(page_idx)] = index + 1
        caption_key = "image_caption" if block_type == "image" else "chart_caption"
        caption = " ".join(block.get(caption_key) or []).strip()
        lookup[(int(page_idx), index)] = (str(image_path), caption)
    return lookup


def load_store(key: str) -> StoreData:
    """Load one FAISS store, its fragments and a BM25 index over the same rows.

    Args:
        key: Gold ``source`` value (a key of ``STORE_DIRS``).

    Returns:
        The loaded store.

    Raises:
        RuntimeError: If a fragment id is not unique (it is the fusion identity).
    """
    store_dir = STORE_DIRS[key]
    store = FaissVectorStore(
        PathsConfig(path_db=str(store_dir.parent), collection_name=store_dir.name)
    )
    fragments = store.get_page(None, 0)
    index = faiss.read_index(str(store_dir / "index.faiss"))
    if index.ntotal != len(fragments):
        raise RuntimeError(
            f"{key}: index has {index.ntotal} vectors but {len(fragments)} rows"
        )

    id_to_row: Dict[str, int] = {}
    for row, fragment in enumerate(fragments):
        if fragment.id in id_to_row:
            raise RuntimeError(f"{key}: duplicate fragment id {fragment.id!r}")
        id_to_row[fragment.id] = row

    return StoreData(
        key=key,
        store_dir=store_dir,
        fragments=fragments,
        index=index,
        bm25=Bm25LexicalIndex(store, RetrievalConfig()),
        id_to_row=id_to_row,
        image_paths=resolve_image_paths(key, fragments),
    )


def resolve_image_paths(key: str, fragments: Sequence[Fragment]) -> Dict[int, Tuple[str, str]]:
    """Match every image fragment to its MinerU file, keyed by row index.

    Args:
        key: Store key (for logging).
        fragments: Fragments in stored order.

    Returns:
        Row index -> ``(image_path, raw_caption)`` for the rows that matched.
    """
    by_stem: Dict[str, Dict[Tuple[int, int], Tuple[str, str]]] = {}
    matched: Dict[int, Tuple[str, str]] = {}
    total = 0
    for row, fragment in enumerate(fragments):
        meta = fragment.metadata
        if meta.format != "image":
            continue
        total += 1
        stem = Path(meta.source).stem
        if stem not in by_stem:
            by_stem[stem] = build_image_lookup(stem)
        img_idx = meta.chunk - IMAGE_CHUNK_OFFSET
        match = by_stem[stem].get((meta.page, img_idx))
        if match is not None:
            matched[row] = match
    log(f"[images] {key}: {len(matched)}/{total} image rows matched to files")
    return matched


# ---------------------------------------------------------------------------
# Embedding


def compute_doc_vectors(
    store: StoreData,
    arm: Arm,
    embedder: LlamaServerEmbedder,
    cache_path: Path,
    use_images: bool,
) -> np.ndarray:
    """Embed every document of one store, caching the matrix as ``.npy``.

    Text and table rows are embedded with the document prefix the adapter
    applies; image rows are embedded as pixels plus their raw caption (or
    caption-only text when ``use_images`` is false, or when the image did not
    match a file). Images run last, since they are the expensive calls.

    Args:
        store: Store whose fragments are embedded.
        arm: Arm being computed (for the log line).
        embedder: Configured ``LlamaServerEmbedder``.
        cache_path: Where to read/write the cached float32 matrix.
        use_images: Whether to embed image pixels.

    Returns:
        A ``(n_rows, dimensions)`` float32 array in stored order.
    """
    n_rows = len(store.fragments)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape == (n_rows, arm.dimensions):
            log(f"[docs] {store.key}/{arm.name}: cache hit {cache_path.name}")
            return cached
        log(f"[docs] {store.key}/{arm.name}: stale cache {cached.shape}, recomputing")

    vectors: List[Optional[List[float]]] = [None] * n_rows
    text_jobs: List[Tuple[int, str]] = []
    image_jobs: List[Tuple[int, str, str]] = []

    for row, fragment in enumerate(store.fragments):
        if fragment.metadata.format == "image":
            match = store.image_paths.get(row)
            if use_images and match is not None:
                image_jobs.append((row, match[0], match[1]))
            else:
                text_jobs.append((row, fragment.doc or ""))
        else:
            text_jobs.append((row, fragment.doc))

    log(
        f"[docs] {store.key}/{arm.name}: {len(text_jobs)} text + "
        f"{len(image_jobs)} image rows to embed"
    )

    def embed_text(job: Tuple[int, str]) -> Tuple[int, List[float]]:
        row, text = job
        return row, embedder.embed(text)

    def embed_image(job: Tuple[int, str, str]) -> Tuple[int, List[float]]:
        row, image_path, caption = job
        return row, embedder.embed_image(image_path, caption=caption)

    _run_jobs(text_jobs, embed_text, vectors, store.key, arm.name, "text")
    _run_jobs(image_jobs, embed_image, vectors, store.key, arm.name, "image")

    matrix = np.asarray(vectors, dtype="float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    log(f"[docs] {store.key}/{arm.name}: wrote {cache_path.name} {matrix.shape}")
    return matrix


def _run_jobs(
    jobs: Sequence,
    worker,
    vectors: List[Optional[List[float]]],
    store_key: str,
    arm_name: str,
    label: str,
) -> None:
    """Embed ``jobs`` concurrently, filling ``vectors`` in place.

    Args:
        jobs: Sequence of job tuples passed to ``worker``.
        worker: Callable returning ``(row, vector)``.
        vectors: Output list, indexed by row.
        store_key: Store key (for progress lines).
        arm_name: Arm name (for progress lines).
        label: "text" or "image" (for progress lines).
    """
    if not jobs:
        return
    started = time.time()
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            row, vector = future.result()
            vectors[row] = vector
            if done % 200 == 0 or done == len(futures):
                elapsed = time.time() - started
                log(
                    f"[docs] {store_key}/{arm_name}: {label} {done}/{len(futures)} "
                    f"({elapsed:.0f}s)"
                )


def build_index(matrix: np.ndarray, dimensions: int) -> faiss.Index:
    """Build an in-memory inner-product index over L2-normalized rows.

    Args:
        matrix: Document vectors (already normalized by the adapter).
        dimensions: Expected vector width.

    Returns:
        A populated ``IndexFlatIP``.

    Raises:
        RuntimeError: If ``matrix`` width does not match ``dimensions``.
    """
    if matrix.shape[1] != dimensions:
        raise RuntimeError(f"expected {dimensions}-d vectors, got {matrix.shape[1]}")
    normalized = np.ascontiguousarray(matrix, dtype="float32")
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(dimensions)
    index.add(normalized)
    return index


# ---------------------------------------------------------------------------
# Retrieval


def search_index(index: faiss.Index, query: Sequence[float], top_k: int) -> List[int]:
    """Return the ``top_k`` row indices for a query vector, best first.

    Args:
        index: FAISS index over the store's rows.
        query: Query embedding (normalized in place).
        top_k: Number of rows to fetch.

    Returns:
        Row indices, descending by inner-product score.
    """
    vector = np.asarray([query], dtype="float32")
    faiss.normalize_L2(vector)
    _, ids = index.search(vector, top_k)
    return [int(i) for i in ids[0] if i != -1]


def rank_first_relevant(
    case: dict, ranked_rows: Sequence[int], store: StoreData
) -> Optional[int]:
    """Return the 1-based rank of the first relevant row, or ``None``.

    Relevance is page-level and mirrors ``grade.py``: same paper stem, page in
    ``verified_pages`` (1-based), and -- for cases that carry ``expect_kind_any``
    -- the row's format mapped to a kind in that set.

    Args:
        case: Gold case.
        ranked_rows: Row indices, best first.
        store: Store the rows belong to.

    Returns:
        The rank of the first relevant row, or ``None`` if none is relevant.
    """
    paper = case["paper"]
    verified = set(case["verified_pages"])
    kinds = case.get("expect_kind_any")
    for rank, row in enumerate(ranked_rows, start=1):
        meta = store.fragments[row].metadata
        if Path(meta.source).stem != paper or (meta.page + 1) not in verified:
            continue
        if kinds is not None and _FORMAT_TO_KIND.get(meta.format, "text") not in kinds:
            continue
        return rank
    return None


def hybrid_rows(store: StoreData, dense_rows: Sequence[int], question: str) -> List[int]:
    """Fuse the dense top-80 with the BM25 top-40 via weighted RRF.

    Args:
        store: Store under test.
        dense_rows: Dense row indices (top-80), best first.
        question: Query text for the lexical branch.

    Returns:
        Fused row indices, ``score_final``-descending.
    """
    config = RetrievalConfig()
    semantic_fragments = [store.fragments[row] for row in dense_rows]
    keyword_fragments = store.bm25.search(question, HYBRID_BM25_K)
    fused = fuse_semantic_and_keyword(
        [semantic_fragments],
        keyword_fragments,
        rrf_k=config.rrf_k,
        weight_semantic=config.weight_semantic_rrf,
        weight_keyword=config.weight_bm25_rrf,
    )
    return [store.id_to_row[fragment.id] for fragment in fused]


# ---------------------------------------------------------------------------
# Metrics


def metric_block(ranks: Sequence[Optional[int]]) -> dict:
    """Compute Recall@k and MRR@k for a list of first-relevant ranks.

    Args:
        ranks: First-relevant rank per case (``None`` when nothing relevant).

    Returns:
        Mapping of metric name to value, plus the case count ``n``.
    """
    n = len(ranks)
    block: dict = {"n": n}
    for k in RECALL_KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        block[f"recall@{k}"] = hits / n if n else 0.0
    for k in MRR_KS:
        total = sum(1.0 / r for r in ranks if r is not None and r <= k)
        block[f"mrr@{k}"] = total / n if n else 0.0
    return block


def grouped_metrics(
    cases: Sequence[dict],
    ranks: Dict[str, Dict[str, Dict[str, Optional[int]]]],
    arm: str,
    mode: str,
    attribute: str,
) -> Dict[str, dict]:
    """Compute metric blocks grouped by one case attribute.

    Args:
        cases: Selected gold cases.
        ranks: ``ranks[arm][mode][case_id]`` -> first-relevant rank.
        arm: Arm name.
        mode: "dense" or "hybrid".
        attribute: Case field to group by.

    Returns:
        Mapping of attribute value to metric block.
    """
    groups: Dict[str, List[Optional[int]]] = defaultdict(list)
    for case in cases:
        groups[case[attribute]].append(ranks[arm][mode][case["id"]])
    return {value: metric_block(values) for value, values in groups.items()}


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial (sign) test p-value.

    Args:
        better: Number of cases where the second arm ranked higher.
        worse: Number of cases where the second arm ranked lower.

    Returns:
        The two-sided p-value under ``p = 0.5`` (1.0 when there are no pairs).
    """
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_comparison(
    cases: Sequence[dict],
    ranks: Dict[str, Dict[str, Dict[str, Optional[int]]]],
    first: str,
    second: str,
    mode: str,
) -> dict:
    """Compare two arms by how well each ranks the first relevant hit.

    Args:
        cases: Selected gold cases.
        ranks: ``ranks[arm][mode][case_id]`` -> first-relevant rank.
        first: Baseline arm name.
        second: Challenger arm name.
        mode: "dense" or "hybrid".

    Returns:
        Better/worse/tie counts (from the challenger's perspective) and the
        two-sided sign-test p-value.
    """
    better = worse = tie = 0
    for case in cases:
        case_id = case["id"]
        a = ranks[first][mode][case_id]
        b = ranks[second][mode][case_id]
        if a is None and b is None:
            tie += 1
        elif a is None:
            better += 1
        elif b is None:
            worse += 1
        elif b < a:
            better += 1
        elif b > a:
            worse += 1
        else:
            tie += 1
    return {
        "better": better,
        "worse": worse,
        "tie": tie,
        "p_value": sign_test(better, worse),
    }


# ---------------------------------------------------------------------------
# Report


def _fmt(value: float) -> str:
    """Format a metric value to three decimals."""
    return f"{value:.3f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: Row cells (already strings).

    Returns:
        The table as a string ending in a newline.
    """
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def metric_columns() -> List[str]:
    """Return the metric column headers shared by every metric table."""
    return [f"R@{k}" for k in RECALL_KS] + [f"MRR@{k}" for k in MRR_KS]


def metric_row(label: str, block: dict) -> List[str]:
    """Render one metric block as a table row.

    Args:
        label: Group label (arm or attribute value).
        block: Metric block from ``metric_block``.

    Returns:
        The row cells.
    """
    cells = [f"{label} (n={block['n']})"]
    cells += [_fmt(block[f"recall@{k}"]) for k in RECALL_KS]
    cells += [_fmt(block[f"mrr@{k}"]) for k in MRR_KS]
    return cells


def render_report(
    cases: Sequence[dict],
    arms: Sequence[str],
    ranks: Dict[str, Dict[str, Dict[str, Optional[int]]]],
    paired: Dict[str, Dict[str, dict]],
    image_stats: Dict[str, dict],
    wall_times: Dict[str, float],
    use_images: bool,
) -> str:
    """Render the full Markdown report.

    Args:
        cases: Selected gold cases.
        arms: Arm names in report order.
        ranks: ``ranks[arm][mode][case_id]`` -> first-relevant rank.
        paired: Paired-comparison blocks keyed by mode then pair label.
        image_stats: Per-store image-matching counts.
        wall_times: Phase wall times in seconds.
        use_images: Whether image pixels were embedded.

    Returns:
        The report as Markdown text.
    """
    columns = metric_columns()
    out: List[str] = []
    out.append("# Embedder A/B retrieval benchmark\n")
    out.append(f"- Cases: {len(cases)} (non-study gold cases)\n")
    out.append(f"- Arms: {', '.join(arms)}\n")
    out.append(f"- Image pixels embedded: {'yes' if use_images else 'no (caption-only)'}\n")
    out.append(
        "- Dense: top-50. Hybrid: dense top-80 + BM25 top-40, "
        "weighted RRF (k=60, 0.55/0.45).\n"
    )
    out.append("- Wall times (s): ")
    out.append(", ".join(f"{name}={value:.0f}" for name, value in wall_times.items()))
    out.append("\n")

    out.append("\n## Image matching\n")
    out.append(md_table(
        ["store", "image rows", "matched", "unmatched"],
        [
            [
                store,
                str(stats["total"]),
                str(stats["matched"]),
                str(stats["total"] - stats["matched"]),
            ]
            for store, stats in image_stats.items()
        ],
    ))

    for mode in ("dense", "hybrid"):
        out.append(f"\n## Overall ({mode})\n")
        rows = [metric_row(arm, metric_block([ranks[arm][mode][c["id"]] for c in cases]))
                for arm in arms]
        out.append(md_table(["arm"] + columns, rows))

        for attribute, title in (
            ("source", "By source"),
            ("lang", "By language"),
            ("case_type", "By case type"),
        ):
            out.append(f"\n## {title} ({mode})\n")
            groups = sorted({c[attribute] for c in cases})
            table_rows = []
            for group in groups:
                group_cases = [c for c in cases if c[attribute] == group]
                for arm in arms:
                    block = metric_block([ranks[arm][mode][c["id"]] for c in group_cases])
                    table_rows.append(metric_row(f"{group} / {arm}", block))
            out.append(md_table(["group / arm"] + columns, table_rows))

        out.append(f"\n## Paired comparison ({mode})\n")
        pair_rows = []
        for label, result in paired[mode].items():
            pair_rows.append([
                label,
                str(result["better"]),
                str(result["worse"]),
                str(result["tie"]),
                _fmt(result["p_value"]),
            ])
        out.append(md_table(["pair (second vs first)", "better", "worse", "tie", "p"], pair_rows))

    return "".join(out)


# ---------------------------------------------------------------------------
# Main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stores",
        default=",".join(STORE_DIRS),
        help="Comma-separated gold sources to evaluate (default: all).",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="Comma-separated arms to evaluate (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Keep only the first N gold cases (0 = all); for a smoke test.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Embed image rows as caption text instead of pixels.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="llama-server base URL for embeddinggemma-2.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for report.md, results.json and cached vectors.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark and write the report.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for store in stores:
        if store not in STORE_DIRS:
            raise SystemExit(f"unknown store {store!r}; choose from {sorted(STORE_DIRS)}")
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")

    use_images = not args.no_images
    wall_times: Dict[str, float] = {}
    grand_start = time.time()

    cases = load_gold_cases(args.limit)
    needed_stores = [s for s in stores if any(c["source"] == s for c in cases)]
    log(f"[main] {len(cases)} cases across stores {needed_stores}")

    started = time.time()
    loaded = {key: load_store(key) for key in needed_stores}
    wall_times["load"] = time.time() - started

    image_stats = {
        key: {"total": sum(1 for f in data.fragments if f.metadata.format == "image"),
              "matched": len(data.image_paths)}
        for key, data in loaded.items()
    }

    # Query embeddings: jina questions come from the isolated worker, gemma
    # questions from the HTTP server. One embedder per arm, closed at the end.
    started = time.time()
    query_vectors: Dict[str, Dict[str, List[float]]] = {arm: {} for arm in arms}
    jina_query_embedder: Optional[JinaClipEmbedder] = None
    gemma_embedders: Dict[str, LlamaServerEmbedder] = {}
    try:
        for arm in arms:
            if ARMS[arm].kind == "jina":
                jina_query_embedder = JinaClipEmbedder(_isolated_python())
            else:
                gemma_embedders[arm] = LlamaServerEmbedder(
                    args.base_url, dimensions=ARMS[arm].dimensions
                )
        for arm in arms:
            for case in cases:
                question = case["question"]
                if question in query_vectors[arm]:
                    continue
                if ARMS[arm].kind == "jina":
                    query_vectors[arm][question] = jina_query_embedder.embed(question)
                else:
                    query_vectors[arm][question] = gemma_embedders[arm].embed_query(question)
            log(f"[query] {arm}: {len(query_vectors[arm])} question vectors")
    except Exception:
        if jina_query_embedder is not None:
            jina_query_embedder.close()
        raise
    wall_times["queries"] = time.time() - started

    # Document vectors and in-memory indexes for the gemma arms.
    started = time.time()
    doc_indexes: Dict[str, Dict[str, faiss.Index]] = {arm: {} for arm in arms}
    for arm in arms:
        if ARMS[arm].kind != "gemma":
            continue
        for key, data in loaded.items():
            cache_path = out_dir / f"{key}__{arm}__{'img' if use_images else 'noimg'}.npy"
            matrix = compute_doc_vectors(
                data, ARMS[arm], gemma_embedders[arm], cache_path, use_images
            )
            doc_indexes[arm][key] = build_index(matrix, ARMS[arm].dimensions)
    wall_times["docs"] = time.time() - started

    # Retrieval and scoring.
    started = time.time()
    ranks: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {
        arm: {"dense": {}, "hybrid": {}} for arm in arms
    }
    per_case: List[dict] = []
    for case in cases:
        data = loaded[case["source"]]
        case_record = {
            "id": case["id"],
            "source": case["source"],
            "lang": case["lang"],
            "case_type": case["case_type"],
            "paper": case["paper"],
            "verified_pages": case["verified_pages"],
            "arms": {},
        }
        for arm in arms:
            if ARMS[arm].kind == "jina":
                index = data.index
            else:
                index = doc_indexes[arm][case["source"]]
            dense = search_index(index, query_vectors[arm][case["question"]], HYBRID_DENSE_K)
            hybrid = hybrid_rows(data, dense[:HYBRID_DENSE_K], case["question"])
            dense_rank = rank_first_relevant(case, dense[:DENSE_K], data)
            hybrid_rank = rank_first_relevant(case, hybrid[:DENSE_K], data)
            ranks[arm]["dense"][case["id"]] = dense_rank
            ranks[arm]["hybrid"][case["id"]] = hybrid_rank
            case_record["arms"][arm] = {
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
                "dense_top": [data.fragments[r].id for r in dense[:DENSE_K]],
                "hybrid_top": [data.fragments[r].id for r in hybrid[:DENSE_K]],
            }
        per_case.append(case_record)
    wall_times["retrieval"] = time.time() - started

    if jina_query_embedder is not None:
        jina_query_embedder.close()
    for embedder in gemma_embedders.values():
        embedder.close()

    # Paired comparisons: each gemma width against jina, and against each other.
    pairs: List[Tuple[str, str, str]] = []
    if "gemma512" in arms and "jina" in arms:
        pairs.append(("gemma512__vs__jina", "jina", "gemma512"))
    if "gemma768" in arms and "jina" in arms:
        pairs.append(("gemma768__vs__jina", "jina", "gemma768"))
    if "gemma768" in arms and "gemma512" in arms:
        pairs.append(("gemma768__vs__gemma512", "gemma512", "gemma768"))

    paired: Dict[str, Dict[str, dict]] = {"dense": {}, "hybrid": {}}
    for label, first, second in pairs:
        for mode in ("dense", "hybrid"):
            paired[mode][label] = paired_comparison(cases, ranks, first, second, mode)

    wall_times["total"] = time.time() - grand_start
    report = render_report(
        cases, arms, ranks, paired, image_stats, wall_times, use_images
    )

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    results = {
        "config": {
            "arms": arms,
            "stores": needed_stores,
            "use_images": use_images,
            "dense_k": DENSE_K,
            "hybrid_dense_k": HYBRID_DENSE_K,
            "hybrid_bm25_k": HYBRID_BM25_K,
            "recall_ks": list(RECALL_KS),
            "mrr_ks": list(MRR_KS),
        },
        "wall_times_s": wall_times,
        "image_matching": image_stats,
        "metrics": {
            arm: {
                mode: {
                    "overall": metric_block([ranks[arm][mode][c["id"]] for c in cases]),
                    "by_source": grouped_metrics(cases, ranks, arm, mode, "source"),
                    "by_lang": grouped_metrics(cases, ranks, arm, mode, "lang"),
                    "by_case_type": grouped_metrics(cases, ranks, arm, mode, "case_type"),
                }
                for mode in ("dense", "hybrid")
            }
            for arm in arms
        },
        "paired": paired,
        "cases": per_case,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(report)
    log(f"[main] wrote {out_dir / 'report.md'} and {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
