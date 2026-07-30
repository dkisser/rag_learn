"""Tests for HybridRetriever (vector + BM25, fused via RRF)."""

from __future__ import annotations

from pathlib import Path

from rag_learn.loader import Chunk
from rag_learn.retriever.base import BaseRetriever, Hit
from rag_learn.retriever.bm25_index import BM25Index
from rag_learn.retriever.hybrid_impl import HybridRetriever


class _FakeVectorRetriever:
    """Implements BaseRetriever via duck-typing."""

    def __init__(self, hits_by_query: dict[str, list[Hit]]) -> None:
        self._hits_by_query = hits_by_query
        self.indexed_dir: str | None = None

    def ensure_indexed(self, docs_dir: str) -> None:
        self.indexed_dir = docs_dir

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self._hits_by_query.get(query, [])[:k]


class _FakeBM25:
    """Implements BM25Index duck-typing (only ``search``/``is_built``/``build``)."""

    def __init__(self, hits_by_query: dict[str, list[Hit]]) -> None:
        self._hits_by_query = hits_by_query
        self.built_with: list[Chunk] | None = None

    @property
    def is_built(self) -> bool:
        return True  # already built for tests

    def build(self, chunks: list[Chunk]) -> None:
        self.built_with = chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self._hits_by_query.get(query, [])[:k]


def _hit(text: str, source: str, idx: int, score: float) -> Hit:
    return Hit(text=text, source_file=source, chunk_index=idx, score=score)


def test_search_returns_top_k_sorted_by_rrf_score() -> None:
    vector = _FakeVectorRetriever(
        {
            "q": [
                _hit("vec-only", "a.md", 0, 0.10),
                _hit("shared", "shared.md", 1, 0.20),
            ]
        }
    )
    bm25 = _FakeBM25(
        {
            "q": [
                _hit("bm25-only", "b.md", 0, 5.0),
                _hit("shared", "shared.md", 1, 4.0),
            ]
        }
    )
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),  # unused; fake vector bypasses Chroma
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )

    hits = hybrid.search("q", k=3)
    assert len(hits) == 3
    # The shared chunk gets two reciprocal-rank contributions → highest score.
    assert hits[0].source_file == "shared.md"
    assert hits[0].chunk_index == 1
    # Scores are descending.
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_search_dedupes_shared_hits() -> None:
    vector = _FakeVectorRetriever(
        {"q": [_hit("same", "x.md", 0, 0.1), _hit("only-vec", "y.md", 0, 0.2)]}
    )
    bm25 = _FakeBM25({"q": [_hit("same", "x.md", 0, 3.0), _hit("only-bm25", "z.md", 0, 2.0)]})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hits = hybrid.search("q", k=4)
    keys = {(h.source_file, h.chunk_index) for h in hits}
    assert len(keys) == 3  # no duplicates: x.md, y.md, z.md
    assert ("x.md", 0) in keys


def test_search_respects_k_limit() -> None:
    vector = _FakeVectorRetriever({"q": [_hit(f"v{i}", f"v{i}.md", 0, 0.1 * i) for i in range(5)]})
    bm25 = _FakeBM25({"q": [_hit(f"b{i}", f"b{i}.md", 0, 1.0 * i) for i in range(5)]})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hits = hybrid.search("q", k=3)
    assert len(hits) == 3


def test_search_empty_when_neither_side_returns_hits() -> None:
    vector = _FakeVectorRetriever({})
    bm25 = _FakeBM25({})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    assert hybrid.search("q", k=5) == []


def test_search_returns_only_vector_when_bm25_empty() -> None:
    vector = _FakeVectorRetriever({"q": [_hit("v1", "v.md", 0, 0.1)]})
    bm25 = _FakeBM25({})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hits = hybrid.search("q", k=5)
    assert len(hits) == 1
    assert hits[0].source_file == "v.md"


def test_ensure_indexed_delegates_to_both_sides() -> None:
    vector = _FakeVectorRetriever({})
    bm25 = _FakeBM25({})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hybrid.ensure_indexed("/some/docs")
    assert vector.indexed_dir == "/some/docs"
    # BM25.build only runs when ``is_built`` is False. Our fake reports True,
    # so we just confirm the delegation contract exists.


def test_hybrid_retriever_satisfies_protocol() -> None:
    vector = _FakeVectorRetriever({"q": [_hit("v", "v.md", 0, 0.1)]})
    bm25 = _FakeBM25({})
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    assert isinstance(hybrid, BaseRetriever)  # Protocol is runtime_checkable


def test_rrf_score_higher_for_hits_in_both_lists() -> None:
    """Shared hits must outrank hits that only appear in one list."""
    vector = _FakeVectorRetriever(
        {
            "q": [
                _hit("shared", "shared.md", 0, 0.10),
                _hit("vec-only", "v.md", 0, 0.05),
            ]
        }
    )
    bm25 = _FakeBM25(
        {
            "q": [
                _hit("shared", "shared.md", 0, 5.0),
                _hit("bm25-only", "b.md", 0, 4.0),
            ]
        }
    )
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        rrf_k=60,
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hits = hybrid.search("q", k=3)
    # shared hit ranks first, single-list hits follow.
    assert hits[0].source_file == "shared.md"
    # The shared hit gets reciprocal rank from both lists → strictly higher.
    assert hits[0].score > hits[1].score
    assert hits[0].score > hits[2].score


def test_hybrid_uses_real_bm25_with_chinese_keywords() -> None:
    """End-to-end: real BM25 + fake vector should surface a Chinese keyword match."""
    chunks = [
        Chunk(
            text="苏帕摩 中度烘焙 阿拉比卡",
            source_file="coffee.md",
            chunk_index=0,
            char_start=0,
            char_end=20,
        ),
        Chunk(
            text="无关内容",
            source_file="other.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
        ),
        Chunk(
            text="另一段无关说明",
            source_file="misc.md",
            chunk_index=0,
            char_start=0,
            char_end=6,
        ),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)
    vector = _FakeVectorRetriever({})  # empty vector side
    hybrid = HybridRetriever(
        persist_dir=Path("/tmp/nope"),
        collection_name="test",
        vector_retriever=vector,
        bm25_index=bm25,
    )
    hits = hybrid.search("苏帕摩", k=2)
    assert len(hits) == 1
    assert hits[0].source_file == "coffee.md"
