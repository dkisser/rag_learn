"""Unit tests for the routing caption rendered under the answer box.

Regression guard: the caption used to read ``answer_stream``'s ``metadata``
argument, assuming it was populated in place. Once the pipeline started
copying that dict (required — ``eval.runner`` shares one dict across
concurrent rows), the caption silently rendered "intent: specific ·
sub-queries: 0" for every question, even when the catalog branch had run.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_learn.app import _format_routing
from rag_learn.config import Config
from rag_learn.routing import RoutingInfo


def test_routing_caption_disabled(make_routing_config: Callable[..., Config]):
    caption = _format_routing(make_routing_config(intent_enabled=False), None)
    assert "routing 已关闭" in caption


def test_routing_caption_none_when_classifier_did_not_run(
    make_routing_config: Callable[..., Config],
):
    caption = _format_routing(make_routing_config(), None)
    assert "未产生 routing" in caption


def test_routing_caption_renders_catalog_branch(make_routing_config: Callable[..., Config]):
    info = RoutingInfo(
        intent="all",
        sub_queries=("a", "b", "c"),
        target_collections=("shanzhongshi",),
        merged_k=20,
    )
    caption = _format_routing(make_routing_config(), info)
    assert "intent: `all`" in caption
    assert "sub-queries: `3`" in caption
    assert "target: `[shanzhongshi]`" in caption
    assert "unique hits: `20`" in caption


def test_routing_caption_renders_specific_branch(make_routing_config: Callable[..., Config]):
    info = RoutingInfo(intent="specific", target_collections=("rag_doc",), merged_k=5)
    caption = _format_routing(make_routing_config(), info)
    assert "intent: `specific`" in caption
    assert "sub-queries: `0`" in caption
    assert "unique hits: `5`" in caption
