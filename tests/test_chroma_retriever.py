from __future__ import annotations

from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.chroma_impl import ChromaRetriever


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    p = tmp_path / "chroma"
    p.mkdir()
    return p


def test_chroma_retriever_ensure_indexed_then_search(chroma_dir: Path, fixtures_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    r.ensure_indexed(str(fixtures_dir))
    hits = r.search("alpha", k=3)
    assert isinstance(hits, list)
    assert all(hasattr(h, "text") for h in hits)
    assert all(hasattr(h, "score") for h in hits)
    assert all(h.score >= 0 for h in hits)
    assert len(hits) <= 3


def test_chroma_retriever_is_base_retriever(chroma_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    assert isinstance(r, BaseRetriever)


def test_chroma_retriever_is_idempotent(chroma_dir: Path, fixtures_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    r.ensure_indexed(str(fixtures_dir))
    first_count = r.search("alpha", k=5)
    r.ensure_indexed(str(fixtures_dir))
    second_count = r.search("alpha", k=5)
    assert len(first_count) == len(second_count)


def test_chroma_retriever_second_collection_reuses_persisted(chroma_dir: Path, fixtures_dir: Path):
    ChromaRetriever(persist_dir=chroma_dir).ensure_indexed(str(fixtures_dir))
    fresh = ChromaRetriever(persist_dir=chroma_dir)
    hits = fresh.search("alpha", k=5)
    assert hits, "second client should see already-indexed data without re-ingesting"
