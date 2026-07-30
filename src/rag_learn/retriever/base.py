"""Retriever contract shared by all adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    text: str  # chunk content
    source_file: str  # e.g. "18-graphrag.md"
    chunk_index: int  # index within source file
    score: float  # relevance score; semantics depend on retriever
    # Vector retrievers report L2/cosine distance (lower = more similar).
    # BM25 and Hybrid (RRF) retrievers report a higher-is-better score.


@runtime_checkable
class BaseRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...

    def ensure_indexed(self, docs_dir: str) -> None: ...
