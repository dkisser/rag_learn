"""Tests for the build_retriever factory."""

from __future__ import annotations

from pathlib import Path

from rag_learn.retriever.chroma_impl import ChromaRetriever
from rag_learn.retriever.factory import build_retriever
from rag_learn.retriever.hybrid_impl import HybridRetriever


def test_returns_chroma_when_hybrid_disabled(tmp_path: Path) -> None:
    retriever = build_retriever(tmp_path / "chroma", collection_name="xxx", hybrid_enabled=False)
    assert isinstance(retriever, ChromaRetriever)
    assert not isinstance(retriever, HybridRetriever)


def test_returns_hybrid_when_enabled(tmp_path: Path) -> None:
    retriever = build_retriever(
        tmp_path / "chroma",
        collection_name="xxx",
        hybrid_enabled=True,
        hybrid_rrf_k=42,
    )
    assert isinstance(retriever, HybridRetriever)
    # rrf_k is forwarded to HybridRetriever.
    assert retriever._rrf_k == 42  # noqa: SLF001 — test inspects internal


def test_hybrid_default_rrf_k(tmp_path: Path) -> None:
    retriever = build_retriever(
        tmp_path / "chroma",
        collection_name="xxx",
        hybrid_enabled=True,
    )
    assert retriever._rrf_k == 60  # noqa: SLF001 — default value
