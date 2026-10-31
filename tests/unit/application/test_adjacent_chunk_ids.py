"""The previous page's tail as ``adjacent_chunk_ids`` builds it (issue #397).

``Answer`` passes the previous page's chunk count; with it, the backward
neighbours of a page's first chunk are that page's last chunks, contiguous with
the anchor, and they come before the forward ones.
"""

from monkeygrab.application.text_chunking import adjacent_chunk_ids
from monkeygrab.domain.chunk_metadata import ChunkMetadata


def test_first_chunk_of_a_page_takes_the_previous_page_tail():
    meta = ChunkMetadata(source="paperA.pdf", page=2, chunk=0, total_chunks_in_page=6)

    assert adjacent_chunk_ids(meta, n_neighbors=1, previous_page_total=6) == [
        "paperA.pdf_pag1_chunk5",
        "paperA.pdf_pag2_chunk1",
    ]
    assert adjacent_chunk_ids(meta, n_neighbors=2, previous_page_total=6) == [
        "paperA.pdf_pag1_chunk4",
        "paperA.pdf_pag1_chunk5",
        "paperA.pdf_pag2_chunk1",
        "paperA.pdf_pag2_chunk2",
    ]
