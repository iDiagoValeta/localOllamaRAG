"""
generate_diagram.py -- MonkeyGrab architecture diagram renderer.

Sends the Mermaid diagram to the Kroki.io rendering API via POST and saves
the result as a local image. No URL length limits; supports PNG, SVG and PDF.

Usage (desde la raíz del repositorio):
    python generate_diagram.py
    python generate_diagram.py --output docs/architecture.png
    python generate_diagram.py --format svg --output docs/architecture.svg

Dependencies:
    requests  (pip install requests)
"""


# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports
#  +-- 2. Diagram definition   (Mermaid flowchart source)
#
#  PIPELINE
#  +-- 3. Rendering            build_url, fetch_image, save
#
#  ENTRY
#  +-- 4. CLI                  parse_args, main
#
# ─────────────────────────────────────────────

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Error: 'requests' not installed.  Run:  pip install requests")


# ─────────────────────────────────────────────
# SECTION 2: DIAGRAM DEFINITION
# ─────────────────────────────────────────────

DIAGRAM = """\
flowchart TD
    %% ── ROW 1: CLIENTS ───────────────────────────────────
    WEB["React Web App"]
    CLI["Rich CLI"]

    %% ── ROW 2: API ────────────────────────────────────────
    API["Flask REST API  ·  web/app.py\\nHTTP / SSE Streaming"]

    %% ── ROW 3: TWO BRANCHES ──────────────────────────────
    subgraph IDX["  Indexing Pipeline  "]
        direction TB
        EXT["PDF Extraction\\npymupdf4llm · pypdf\\n+ vision captioning"]
        CHUNK["Chunking\\n1500 chars · 350 overlap"]
        EMB["Embedding\\nembeddinggemma · 768-d"]
        EXT --> CHUNK --> EMB
    end

    subgraph RET["  Hybrid Retrieval Pipeline  "]
        direction TB
        D1["① Query Decomposition\\ngemma3:4b  —  3 sub-queries"]
        D2["② Semantic Search  +  Lexical Search\\nChromaDB  ·  top-80 + top-40"]
        D3["③ RRF Fusion  +  Reranking\\n55% semantic · 45% lexical · Cross-Encoder"]
        D4["④ Context Expansion  +  Optimization\\nNeighbor chunks · artifact removal"]
        D1 --> D2 --> D3 --> D4
    end

    %% ── ROW 4: STORE / GENERATION ────────────────────────
    DB[("ChromaDB\\nPersistent Vector Store")]
    GEN["Generation\\nQwen3-14B FineTuned  ·  Q4_K_M\\ntemp 0.15  ·  streaming"]

    %% ── ROW 5: MODELS ────────────────────────────────────
    subgraph OLL["  Ollama — Local Inference  "]
        direction LR
        M1["embeddinggemma\\nEmbedding"]
        M2["gemma3:4b\\nOrchestration"]
        M3["BAAI/bge-reranker\\nCross-Encoder"]
        M4["Qwen3-14B FT\\nGenerator"]
    end

    %% ── EDGES ────────────────────────────────────────────
    WEB & CLI -->|"query / PDF upload"| API

    API -->|"PDF files"| IDX
    EMB -->|"store vectors"| DB

    API -->|"user question"| RET
    D2 <-->|"vector + keyword search"| DB
    D4 -->|"ranked context"| GEN
    GEN -->|"answer + sources"| API
    API -->|"SSE tokens"| WEB

    M1 -. embeddings .-> EMB
    M2 -. "decomp + retrieval" .-> D1
    M3 -. reranking .-> D3
    M4 -. generation .-> GEN

    %% ── STYLE ────────────────────────────────────────────
    classDef client  fill:#4A90D9,stroke:#2C5F8A,color:#fff,font-weight:bold
    classDef api     fill:#5BAD6F,stroke:#3A7A4A,color:#fff,font-weight:bold
    classDef idx     fill:#E8A838,stroke:#B07820,color:#fff
    classDef ret     fill:#8B6BB1,stroke:#5E4080,color:#fff
    classDef gen     fill:#D45F5F,stroke:#9A3535,color:#fff,font-weight:bold
    classDef db      fill:#2D7D9A,stroke:#1A5570,color:#fff
    classDef model   fill:#3A3A3A,stroke:#111,color:#eee

    class WEB,CLI client
    class API api
    class EXT,CHUNK,EMB idx
    class D1,D2,D3,D4 ret
    class GEN gen
    class DB db
    class M1,M2,M3,M4 model
"""


# ─────────────────────────────────────────────
# SECTION 3: RENDERING  (Kroki.io POST API)
# ─────────────────────────────────────────────

KROKI_BASE = "https://kroki.io"

FORMATS = ("png", "svg", "pdf")


def build_url(fmt: str) -> str:
    """Return the Kroki POST endpoint for the requested format.

    Args:
        fmt: Output format ('png', 'svg', or 'pdf').

    Returns:
        Full endpoint URL.
    """
    return f"{KROKI_BASE}/mermaid/{fmt}"


def fetch_image(diagram: str, fmt: str, timeout: int = 60) -> bytes:
    """POST the diagram source to Kroki and return the rendered bytes.

    Args:
        diagram: Raw Mermaid diagram text.
        fmt:     Output format.
        timeout: HTTP timeout in seconds.

    Returns:
        Raw response bytes (the rendered image/document).

    Raises:
        SystemExit: On HTTP error or connection failure.
    """
    url = build_url(fmt)
    print(f"  POST {url}")
    try:
        resp = requests.post(
            url,
            data=diagram.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        sys.exit("Error: Could not connect to kroki.io. Check your internet connection.")
    except requests.exceptions.Timeout:
        sys.exit(f"Error: Request timed out after {timeout}s.")

    if resp.status_code != 200:
        detail = resp.text.strip()[:300] if resp.text else "(no detail)"
        sys.exit(
            f"Error: kroki.io returned HTTP {resp.status_code}.\n"
            f"Detail: {detail}\n"
            f"Tip: Validate the diagram at https://kroki.io"
        )
    return resp.content


def save(data: bytes, path: Path) -> None:
    """Write bytes to disk, creating parent directories as needed.

    Args:
        data: Raw file content.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    size_kb = len(data) / 1024
    print(f"  Saved : {path.resolve()}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────
# SECTION 4: CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the MonkeyGrab architecture diagram to an image file via Kroki.io.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output file path. "
            "Defaults to 'docs/monkeygrab_architecture.<format>'."
        ),
    )
    parser.add_argument(
        "--format", "-f",
        choices=FORMATS,
        default="png",
        dest="fmt",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent.parent / "docs" / f"monkeygrab_architecture.{args.fmt}"
    )

    print(f"\nMonkeyGrab -- Diagram Renderer")
    print(f"  Format : {args.fmt.upper()}")
    print(f"  Output : {output_path}")
    print(f"  Engine : kroki.io (Mermaid)")
    print()

    data = fetch_image(DIAGRAM, args.fmt)
    save(data, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
