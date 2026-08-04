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


class _FakeCollection:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.last_n_results: int | None = None

    def query(self, *, n_results: int, **_: object) -> dict[str, object]:
        self.last_n_results = n_results
        return self._result


def _stub_retriever(
    distances: list[float], max_distance: float | None
) -> tuple[ChromaRetriever, _FakeCollection]:
    result: dict[str, object] = {
        "documents": [[f"doc-{i}" for i in range(len(distances))]],
        "metadatas": [
            [{"source_file": f"source-{i}.md", "chunk_index": i} for i in range(len(distances))]
        ],
        "distances": [distances],
    }
    collection = _FakeCollection(result)
    retriever = object.__new__(ChromaRetriever)
    retriever._collection = collection  # noqa: SLF001 — 隔离的查询桩
    retriever._max_distance = max_distance  # noqa: SLF001 — 隔离的查询桩
    return retriever, collection


def test_chroma_search_filters_far_distances_and_overfetches() -> None:
    retriever, collection = _stub_retriever([0.2, 1.0, 1.0001], max_distance=1.0)

    hits = retriever.search("q", k=2)

    assert [hit.score for hit in hits] == [0.2, 1.0]
    assert collection.last_n_results == 8


def test_chroma_search_returns_empty_when_all_distances_exceed_threshold() -> None:
    retriever, _ = _stub_retriever([1.01, 1.2], max_distance=1.0)

    assert retriever.search("q", k=2) == []


def test_chroma_search_zero_distance_keeps_exact_match_only() -> None:
    retriever, _ = _stub_retriever([0.0, 0.0001], max_distance=0.0)

    hits = retriever.search("q", k=2)

    assert [hit.score for hit in hits] == [0.0]


def test_chroma_search_without_threshold_preserves_requested_k() -> None:
    retriever, collection = _stub_retriever([0.2, 1.2], max_distance=None)

    hits = retriever.search("q", k=1)

    assert len(hits) == 1
    assert collection.last_n_results == 1
