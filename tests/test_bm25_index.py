"""Tests for the BM25 keyword index."""

from __future__ import annotations

from rag_learn.loader import Chunk
from rag_learn.retriever.base import Hit
from rag_learn.retriever.bm25_index import BM25Index


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="苏帕摩是中度烘焙的阿拉比卡咖啡豆。",
            source_file="coffee.md",
            chunk_index=0,
            char_start=0,
            char_end=20,
        ),
        Chunk(
            text="本店承诺七日内新鲜烘焙。",
            source_file="promise.md",
            chunk_index=0,
            char_start=0,
            char_end=15,
        ),
        Chunk(
            text="养豆建议：收到后静置三到七天风味更佳。",
            source_file="tips.md",
            chunk_index=0,
            char_start=0,
            char_end=20,
        ),
    ]


def test_search_returns_empty_when_not_built() -> None:
    idx = BM25Index()
    assert idx.search("苏帕摩") == []


def test_search_returns_empty_for_empty_corpus() -> None:
    idx = BM25Index()
    idx.build([])
    assert idx.search("anything") == []


def test_search_finds_keyword_match_and_orders_by_score() -> None:
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("苏帕摩 风味", k=3)
    assert len(hits) >= 1
    # Top hit should be the chunk that mentions 苏帕摩.
    assert hits[0].source_file == "coffee.md"
    # Scores are descending.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_empty_when_no_term_overlap() -> None:
    idx = BM25Index()
    idx.build(_chunks())
    # A query whose tokens don't appear in any chunk yields zero scores.
    hits = idx.search("xyz123 qqq 不存在词")
    assert hits == []


def test_search_respects_k_limit() -> None:
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("烘焙", k=2)
    assert len(hits) <= 2


def test_build_replaces_existing_index() -> None:
    idx = BM25Index()
    idx.build(_chunks())
    # Rebuild with a fresh corpus that does NOT mention 苏帕摩.
    new_chunks = [
        Chunk(
            text="全新内容",
            source_file="new.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
        ),
        Chunk(
            text="另一个无关的内容",
            source_file="other.md",
            chunk_index=0,
            char_start=0,
            char_end=8,
        ),
        Chunk(
            text="额外的说明文字",
            source_file="misc.md",
            chunk_index=0,
            char_start=0,
            char_end=7,
        ),
    ]
    idx.build(new_chunks)
    # Old-corpus term should no longer hit anything.
    assert idx.search("苏帕摩") == []
    # New corpus should be searchable for its own content.
    assert len(idx.search("全新")) == 1
    assert idx.search("全新")[0].source_file == "new.md"


def test_search_returns_hits_with_expected_fields() -> None:
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("苏帕摩", k=1)
    assert len(hits) == 1
    hit: Hit = hits[0]
    assert hit.source_file == "coffee.md"
    assert hit.chunk_index == 0
    assert hit.text.startswith("苏帕摩")
    assert hit.score > 0
