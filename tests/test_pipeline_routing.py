"""Integration tests for intent-aware routing in pipeline.answer_stream."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config
from rag_learn.pipeline import answer_stream
from rag_learn.retriever import Hit


def _make_config(
    *,
    intent_enabled: bool = False,
    decompose_enabled: bool = False,
    catalog_recall_k: int = 20,
    rerank_enabled: bool = False,
    rerank_factor: int = 4,
) -> Config:
    base = Path(__file__).parent.parent / "src"
    return Config(
        deepseek_api_key="k",
        llm_model="m",
        deepseek_base_url="u",
        retrieve_k=2,
        chunk_size=800,
        chunk_overlap=50,
        repo_root=base,
        docs_dir=base / "docs" / "rag_doc",
        data_dir=base / "data",
        chroma_dir=base / "data" / "chroma",
        milvus_path=base / "data" / "milvus.db",
        rerank_enabled=rerank_enabled,
        rerank_model="BAAI/bge-reranker-base",
        rerank_factor=rerank_factor,
        rerank_k=None,
        rerank_batch_size=8,
        rerank_device=None,
        hybrid_enabled=False,
        hybrid_rrf_k=60,
        intent_enabled=intent_enabled,
        intent_timeout_s=2.0,
        decompose_enabled=decompose_enabled,
        decompose_timeout_s=2.0,
        decompose_max=8,
        catalog_recall_k=catalog_recall_k,
    )


# ---- Per-sub-query fake retriever ----


class _PerQueryFakeRetriever:
    """Returns hits keyed on the query string. Records every call's k."""

    def __init__(self, hits_by_query: dict[str, list[Hit]]) -> None:
        self._hits_by_query = hits_by_query
        self.calls: list[tuple[str, int]] = []

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.calls.append((query, k))
        return list(self._hits_by_query.get(query, []))


class _FakeReranker:
    """If `.rank` is called, mark it so the test can assert it was NOT used."""

    def __init__(self) -> None:
        self.rank_calls = 0

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        self.rank_calls += 1
        return list(hits)


class _ScriptedRoutingLLM:
    """Routes different system prompts to different canned replies."""

    def __init__(self, intent_reply: str, decompose_reply: str) -> None:
        self.intent_reply = intent_reply
        self.decompose_reply = decompose_reply
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        # Use distinct substrings for each prompt to avoid ambiguity.
        # classify_intent prompt contains "classify a user question"; decompose
        # prompt contains "small catalog". Both prompts contain "RAG system"
        # so we MUST check the more specific phrases first.
        if "decompose" in system.lower() or "small catalog" in system:
            return iter([self.decompose_reply])
        if "classify" in system or "EXACTLY ONE WORD" in system:
            return iter([self.intent_reply])
        # the final answer-construction prompt
        return iter(["final-answer"])


def _make_catalog(tmp_path: Path) -> Catalog:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nhello")
    col = Collection(
        name="abc",
        display_name="Test",
        docs_dir=docs,
        description="测试用集合",
        retriever_factory=lambda _p, _n: _NoopRetriever(),
    )
    return Catalog(collections=(col,))


class _NoopRetriever:
    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


# ---- The 8 new tests ----


def test_intent_disabled_skips_classification(tmp_path: Path):
    """When intent_enabled=False, the LLM's classify intent is never invoked."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["x"]')
    r = _PerQueryFakeRetriever(
        {"anything": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=False, decompose_enabled=True)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog)
    _ = "".join(out["abc"][0])
    # No routing prompt was sent to the LLM.
    classify_calls = [c for c in llm.calls if "classify" in c[0]]
    assert classify_calls == []


def test_specific_intent_uses_original_path(tmp_path: Path):
    """intent=='specific' → no fan-out, original single-query path."""
    llm = _ScriptedRoutingLLM(intent_reply="specific", decompose_reply='["ignored"]')
    r = _PerQueryFakeRetriever(
        {"推荐": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True)

    md: dict = {}
    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog, metadata=md)
    _ = "".join(out["abc"][0])
    # Original path: only the original query was searched.
    assert [q for q, _ in r.calls] == ["推荐"]
    # routing metadata was NOT populated with sub-queries.
    assert md.get("sub_queries") == [] or "sub_queries" not in md


def test_all_intent_triggers_fanout_and_merge(tmp_path: Path):
    """intent=='all' with [sub1, sub2] → both sub-queries searched, merged."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1", "sub2"]')
    hits_sub1 = [
        Hit(text="a1", source_file="x.md", chunk_index=0, score=0.1),
        Hit(text="a2", source_file="x.md", chunk_index=1, score=0.2),
    ]
    hits_sub2 = [
        Hit(text="b1", source_file="y.md", chunk_index=0, score=0.3),
        Hit(text="b2", source_file="y.md", chunk_index=1, score=0.4),
    ]
    r = _PerQueryFakeRetriever({"sub1": hits_sub1, "sub2": hits_sub2})
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=10)

    md: dict = {}
    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog, metadata=md)
    _ = "".join(out["abc"][0])
    # Both sub-queries were searched; original query was NOT used as a sub-query.
    searched = {q for q, _ in r.calls}
    assert searched == {"sub1", "sub2"}
    final_hits = out["abc"][1]
    # All four unique chunks landed in the merged set.
    keys = {(h.source_file, h.chunk_index) for h in final_hits}
    assert keys == {("x.md", 0), ("x.md", 1), ("y.md", 0), ("y.md", 1)}


def test_all_intent_skips_reranker(tmp_path: Path):
    """Even when a reranker is passed, the catalog branch never calls it."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1"]')
    r = _PerQueryFakeRetriever(
        {"sub1": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    reranker = _FakeReranker()
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=5)

    out = answer_stream(
        {"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog, reranker=reranker
    )
    _ = "".join(out["abc"][0])
    assert reranker.rank_calls == 0


def test_all_intent_caps_at_catalog_recall_k(tmp_path: Path):
    """Merged size is capped at Config.catalog_recall_k even when more lands."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1"]')
    r = _PerQueryFakeRetriever(
        {
            "sub1": [
                Hit(text=f"t{i}", source_file=f"f{i}.md", chunk_index=0, score=0.0)
                for i in range(20)
            ]
        }
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=7)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog)
    _ = "".join(out["abc"][0])
    assert len(out["abc"][1]) == 7


def test_all_intent_records_metadata(tmp_path: Path):
    """RAGEvent carries routing fields; caller's dict is NOT mutated."""
    from rag_learn.eval.tracing import ListEmitter

    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1", "sub2"]')
    r = _PerQueryFakeRetriever(
        {
            "sub1": [Hit(text="a", source_file="a.md", chunk_index=0, score=0.0)],
            "sub2": [Hit(text="b", source_file="b.md", chunk_index=0, score=0.0)],
        }
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=10)

    emitter = ListEmitter()
    md: dict = {}
    out = answer_stream(
        {"abc": r},
        llm,
        "推荐",
        k=2,
        config=config,
        catalog=catalog,
        metadata=md,
        emitter=emitter,
    )
    _ = "".join(out["abc"][0])
    out["abc"][2]("")  # triggers emitter.emit(RAGEvent)

    # Caller's dict must NOT be mutated.
    assert md == {}
    # The emitted event metadata must carry the routing fields.
    assert len(emitter.events) == 1
    event = emitter.events[0]
    assert event.metadata["intent"] == "all"
    assert event.metadata["sub_queries"] == ["sub1", "sub2"]
    assert event.metadata["target_collections"] == ["abc"]
    assert event.metadata["merged_k"] == 2


def test_empty_decompose_falls_back_to_original_question(tmp_path: Path):
    """decompose_query returns [] → fan-out uses [question] (no crash)."""
    llm = _ScriptedRoutingLLM(
        intent_reply="all",
        decompose_reply="  ",  # empty / whitespace only
    )
    r = _PerQueryFakeRetriever(
        {"推荐": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=5)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog)
    _ = "".join(out["abc"][0])
    # The original question is the only thing that got searched.
    assert [q for q, _ in r.calls] == ["推荐"]


def test_catalog_recall_mode_uses_catalog_system_prompt(tmp_path: Path):
    """The final ``llm.stream`` call's system message is the catalog prompt."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1"]')
    r = _PerQueryFakeRetriever(
        {"sub1": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True, catalog_recall_k=5)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog)
    _ = "".join(out["abc"][0])
    # Find the streaming call (not the classify nor decompose ones).
    stream_calls = [
        c
        for c in llm.calls
        if c[0] != "" and "classify" not in c[0] and "small catalog" not in c[0]
    ]
    assert stream_calls, "expected at least one streaming call"
    sys_msg = stream_calls[0][0]
    assert "覆盖整个目录" in sys_msg or "catalog" in sys_msg.lower()
