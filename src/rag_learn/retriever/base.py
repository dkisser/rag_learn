"""Retriever contract shared by all adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    text: str  # chunk content
    source_file: str  # e.g. "18-graphrag.md"
    chunk_index: int  # index within source file
    score: float  # L2 distance; lower = more similar


@runtime_checkable
class BaseRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...

    def ensure_indexed(self, docs_dir: str) -> None: ...
