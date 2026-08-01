"""Tests for the ``routing_sink`` out-parameter of ``pipeline.answer_stream``.

The caller's ``metadata`` dict is deliberately treated as read-only (see
``test_pipeline_metadata_isolation``) because ``eval.runner`` shares one
dict across concurrently-processed rows. Routing decisions therefore have
to reach the caller some other way: ``answer_stream`` pushes an immutable
:class:`RoutingInfo` into the optional ``routing_sink`` callback, once per
call, before it returns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config
from rag_learn.pipeline import answer_stream
from rag_learn.retriever import Hit
from rag_learn.routing import RoutingInfo


class _PerQueryFakeRetriever:
    def __init__(self, hits_by_query: dict[str, list[Hit]]) -> None:
        self._hits_by_query = hits_by_query

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
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


def _hit(name: str = "a.md", idx: int = 0) -> Hit:
    return Hit(text="t", source_file=name, chunk_index=idx, score=0.0)


def test_sink_receives_catalog_branch_routing(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    """intent=='all' + decompose → sink gets intent/sub_queries/merged_k."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1", "sub2"]')
    r = _PerQueryFakeRetriever({"sub1": [_hit("a.md", 0)], "sub2": [_hit("b.md", 1)]})
    catalog = _make_catalog(tmp_path)
    seen: list[RoutingInfo] = []

    out = answer_stream(
        {"abc": r},
        llm,
        "推荐",
        k=2,
        config=make_routing_config(),
        catalog=catalog,
        routing_sink=seen.append,
    )
    _ = "".join(out["abc"][0])

    assert len(seen) == 1
    info = seen[0]
    assert info.intent == "all"
    assert info.sub_queries == ("sub1", "sub2")
    assert info.target_collections == ("abc",)
    assert info.merged_k == 2


def test_sink_receives_specific_intent(tmp_path: Path, make_routing_config: Callable[..., Config]):
    """intent=='specific' → sink still fires, with no sub-queries."""
    llm = _ScriptedRoutingLLM(intent_reply="specific", decompose_reply='["ignored"]')
    r = _PerQueryFakeRetriever({"推荐": [_hit()]})
    catalog = _make_catalog(tmp_path)
    seen: list[RoutingInfo] = []

    out = answer_stream(
        {"abc": r},
        llm,
        "推荐",
        k=2,
        config=make_routing_config(),
        catalog=catalog,
        routing_sink=seen.append,
    )
    _ = "".join(out["abc"][0])

    assert len(seen) == 1
    assert seen[0].intent == "specific"
    assert seen[0].sub_queries == ()
    assert seen[0].merged_k == 1


def test_sink_not_called_when_intent_disabled(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    """No classifier ran → nothing to report; the sink stays untouched."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["x"]')
    r = _PerQueryFakeRetriever({"推荐": [_hit()]})
    catalog = _make_catalog(tmp_path)
    seen: list[RoutingInfo] = []

    out = answer_stream(
        {"abc": r},
        llm,
        "推荐",
        k=2,
        config=make_routing_config(intent_enabled=False),
        catalog=catalog,
        routing_sink=seen.append,
    )
    _ = "".join(out["abc"][0])

    assert seen == []


def test_sink_is_optional_and_metadata_still_not_mutated(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    """Omitting the sink keeps the old signature working; dict stays clean."""
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1"]')
    r = _PerQueryFakeRetriever({"sub1": [_hit()]})
    catalog = _make_catalog(tmp_path)
    md = {"llm_model": "m"}

    out = answer_stream(
        {"abc": r}, llm, "推荐", k=2, config=make_routing_config(), catalog=catalog, metadata=md
    )
    _ = "".join(out["abc"][0])

    assert md == {"llm_model": "m"}


def test_routing_info_is_immutable():
    """RoutingInfo is a frozen value object — callers cannot corrupt it."""
    import dataclasses

    import pytest

    info = RoutingInfo(intent="all", sub_queries=("a",), target_collections=("c",), merged_k=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.intent = "specific"  # ty: ignore[invalid-assignment]
