"""Tests for the build_retriever factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    assert retriever._rrf_k == 42  # noqa: SLF001 — 测试检查内部配置


def test_hybrid_default_rrf_k(tmp_path: Path) -> None:
    retriever = build_retriever(
        tmp_path / "chroma",
        collection_name="xxx",
        hybrid_enabled=True,
    )
    assert retriever._rrf_k == 60  # noqa: SLF001 — default value


def test_chroma_max_distance_is_forwarded_to_plain_retriever(tmp_path: Path) -> None:
    fake = MagicMock()
    with patch("rag_learn.retriever.factory.ChromaRetriever", return_value=fake) as factory:
        result = build_retriever(
            tmp_path / "chroma",
            collection_name="xxx",
            chroma_max_distance=0.75,
        )

    assert result is fake
    factory.assert_called_once_with(
        persist_dir=tmp_path / "chroma",
        collection_name="xxx",
        max_distance=0.75,
    )


def test_chroma_max_distance_is_forwarded_to_hybrid_vector_side(tmp_path: Path) -> None:
    fake = MagicMock()
    with patch("rag_learn.retriever.hybrid_impl.ChromaRetriever", return_value=fake) as factory:
        result = build_retriever(
            tmp_path / "chroma",
            collection_name="xxx",
            hybrid_enabled=True,
            chroma_max_distance=0.75,
        )

    assert isinstance(result, HybridRetriever)
    assert result._vector is fake  # noqa: SLF001 — test inspects wiring
    factory.assert_called_once_with(
        persist_dir=tmp_path / "chroma",
        collection_name="xxx",
        max_distance=0.75,
    )


def test_hybrid_factory_forwards_both_options(tmp_path: Path) -> None:
    fake = MagicMock()
    with patch("rag_learn.retriever.factory.HybridRetriever", return_value=fake) as factory:
        result = build_retriever(
            tmp_path / "chroma",
            collection_name="xxx",
            hybrid_enabled=True,
            hybrid_rrf_k=42,
            chroma_max_distance=0.75,
        )

    assert result is fake
    factory.assert_called_once_with(
        persist_dir=tmp_path / "chroma",
        collection_name="xxx",
        rrf_k=42,
        max_distance=0.75,
    )


def test_default_distance_filter_is_disabled_for_direct_factory_calls(tmp_path: Path) -> None:
    fake = MagicMock()
    with patch("rag_learn.retriever.factory.ChromaRetriever", return_value=fake) as factory:
        build_retriever(tmp_path / "chroma", collection_name="xxx")

    factory.assert_called_once_with(
        persist_dir=tmp_path / "chroma",
        collection_name="xxx",
        max_distance=None,
    )


def test_factory_accepts_explicit_none(tmp_path: Path) -> None:
    retriever = build_retriever(
        tmp_path / "chroma",
        collection_name="xxx",
        chroma_max_distance=None,
    )
    assert isinstance(retriever, ChromaRetriever)
