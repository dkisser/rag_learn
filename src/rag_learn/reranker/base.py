"""Reranker contract shared by all implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_learn.retriever.base import Hit


@runtime_checkable
class Reranker(Protocol):
    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Return a new list of hits sorted by descending relevance.

        The returned list may be truncated by the caller to the final k.
        """
        ...
