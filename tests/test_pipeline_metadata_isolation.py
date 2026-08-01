"""Regression tests for cross-call metadata isolation in pipeline.answer_stream.

The user's dict passed as ``metadata`` MUST NOT be mutated by the pipeline.
If it were, downstream code that re-uses the dict across calls would see
stale ``intent`` / ``sub_queries`` / ``target_collections`` bleed from the
previous call.

These tests use a single ``metadata`` dict across multiple ``answer_stream``
invocations to assert the dict returned to the caller is independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config
from rag_learn.pipeline import answer_stream
from rag_learn.retriever import Hit


def _make_config(intent_enabled: bool = True, decompose_enabled: bool = True) -> Config:
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
        rerank_enabled=False,
        rerank_model="BAAI/bge-reranker-base",
        rerank_factor=4,
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
        catalog_sub_k=10,
        catalog_recall_k=10,
    )


class _PerQueryFakeRetriever:
    def __init__(self, hits_by_query: dict[str, list[Hit]]) -> None:
        self._hits_by_query = hits_by_query
        self.calls: list[tuple[str, int]] = []

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.calls.append((query, k))
        return list(self._hits_by_query.get(query, []))


class _ScriptedRoutingLLM:
    def __init__(self, intent_reply: str, decompose_reply: str) -> None:
        self.intent_reply = intent_reply
        self.decompose_reply = decompose_reply

    def stream(self, system: str, user: str) -> Iterator[str]:
        if "small catalog" in system:
            return iter([self.decompose_reply])
        if "classify" in system or "EXACTLY ONE WORD" in system:
            return iter([self.intent_reply])
        return iter(["answer"])


class _NoopRetriever:
    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


def _make_catalog(tmp_path: Path) -> Catalog:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A")
    return Catalog(
        collections=(
            Collection(
                name="abc",
                display_name="T",
                docs_dir=docs,
                description="d",
                retriever_factory=lambda _p, _n: _NoopRetriever(),
            ),
        )
    )


def test_metadata_dict_is_not_mutated_by_catalog_branch(tmp_path: Path):
    """When the catalog branch fires, the caller's metadata dict is unchanged."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["x"]')
    r = _PerQueryFakeRetriever({"x": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]})
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True)

    md = {"llm_model": "m", "rerank_enabled": False}
    snapshot_before = dict(md)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog, metadata=md)
    _ = "".join(out["abc"][0])

    # The caller's dict must be unchanged.
    assert md == snapshot_before, (
        f"callers' metadata dict was mutated: before={snapshot_before} after={md}"
    )


def test_metadata_dict_is_not_mutated_by_specific_intent(tmp_path: Path):
    """Specific-intent path also must not mutate the caller's dict."""
    llm = _ScriptedRoutingLLM(intent_reply="specific", decompose_reply='["ignored"]')
    r = _PerQueryFakeRetriever(
        {"推荐": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True)

    md = {"llm_model": "m"}
    snapshot_before = dict(md)

    out = answer_stream({"abc": r}, llm, "推荐", k=2, config=config, catalog=catalog, metadata=md)
    _ = "".join(out["abc"][0])

    assert md == snapshot_before


def test_routing_field_does_not_leak_between_same_dict_calls(tmp_path: Path):
    """A dict reused across calls must not accumulate stale routing fields."""
    llm_all = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["y"]')
    llm_specific = _ScriptedRoutingLLM(intent_reply="specific", decompose_reply='["ignore"]')
    catalog = _make_catalog(tmp_path)
    config = _make_config(intent_enabled=True, decompose_enabled=True)

    md = {"llm_model": "m"}

    # Call 1: catalog branch with sub_queries=["y"].
    r1 = _PerQueryFakeRetriever(
        {"y": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    out1 = answer_stream(
        {"abc": r1}, llm_all, "推荐", k=2, config=config, catalog=catalog, metadata=md
    )
    _ = "".join(out1["abc"][0])

    # Call 2: reuses the same dict; intent should be "specific", no sub_queries.
    r2 = _PerQueryFakeRetriever(
        {"推荐": [Hit(text="t", source_file="a.md", chunk_index=0, score=0.0)]}
    )
    out2 = answer_stream(
        {"abc": r2}, llm_specific, "推荐", k=2, config=config, catalog=catalog, metadata=md
    )
    _ = "".join(out2["abc"][0])

    # Caller's dict should still look like the original (no leaked routing fields).
    assert md == {"llm_model": "m"}, f"callers' dict was mutated across calls: {md}"
