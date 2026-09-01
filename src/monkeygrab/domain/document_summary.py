"""DocumentSummary -- a structured summary of retrieved evidence.

Issue #140. ``Answer`` produces prose a person reads once; this is an artifact
a UI renders in parts and a reader navigates. Keeping them separate types is
the point: a summary section knows which fragments it came from, and a caller
that wants to show "where does this claim come from" needs that link to exist
in the data, not to be recovered by matching strings after the fact.

Zero infrastructure imports, like everything else under ``domain/``.
"""

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class SummarySection:
    """One section of a summary, and the evidence it was written from.

    Attributes:
        heading: Short title for the section. Not a document heading copied
            verbatim -- the generator writes it, because a summary's structure
            need not mirror the source's.
        body: The section's prose.
        source_pages: Pages of the fragments this section was written from,
            in ascending order, deduplicated. Pages rather than fragment ids
            because a page is what a reader can open and check, which is the
            whole purpose of carrying them.
    """

    heading: str
    body: str
    source_pages: Tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class DocumentSummary:
    """An ordered set of summary sections over one retrieval.

    Attributes:
        sections: In the order the generator produced them, which is the
            order a reader should read them.
        source_document: The document the evidence came from, or "" when the
            fragments span more than one -- an honest empty rather than the
            first one, which would read as a claim about the whole summary.
    """

    sections: Tuple[SummarySection, ...]
    source_document: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.sections


@dataclasses.dataclass(frozen=True)
class OutlineNode:
    """One heading in a document outline, with its children.

    A tree rather than a flat list carrying depth numbers: the nesting IS the
    artifact, and a flat representation makes every consumer rebuild it, each
    in its own slightly different way.

    Attributes:
        title: The heading text.
        children: Sub-headings, in reading order. A leaf has none.
    """

    title: str
    children: Tuple["OutlineNode", ...] = ()

    @property
    def depth(self) -> int:
        """How many levels this node spans, itself included."""
        return 1 + max((child.depth for child in self.children), default=0)


@dataclasses.dataclass(frozen=True)
class DocumentOutline:
    """The heading tree of one retrieval.

    Attributes:
        nodes: Top-level headings, in reading order.
        source_document: As in ``DocumentSummary`` -- "" when the evidence
            spans more than one document, rather than naming one of them.
    """

    nodes: Tuple[OutlineNode, ...]
    source_document: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.nodes
