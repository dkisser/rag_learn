"""Unit tests for routing.classify_intent and routing.decompose_query."""

from __future__ import annotations

import time
from collections.abc import Iterator

from rag_learn.routing import (
    DECOMPOSE_MAX,
    classify_intent,
    decompose_query,
)


class _ScriptedFakeLLM:
    """LLM fake that returns a canned reply; optionally records calls."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        # Yield in 2 chars / call to mimic real streaming tokens.
        out = self._reply
        for i in range(0, len(out), 2):
            yield out[i : i + 2]


class _BlockingLLM:
    """LLM fake that never yields. Used to assert timeout fallback."""

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls += 1
        # Sleep much longer than the caller's timeout to force a timeout.
        time.sleep(5.0)
        yield "never"


# ---- classify_intent ----


def test_classify_returns_all_on_clean_label():
    llm = _ScriptedFakeLLM("all")
    assert classify_intent(llm, "推荐几款豆子") == "all"
    assert len(llm.calls) == 1


def test_classify_returns_specific_on_clean_label():
    llm = _ScriptedFakeLLM("specific")
    assert classify_intent(llm, "苏帕摩是什么风味") == "specific"


def test_classify_tolerates_chatty_prefix():
    llm = _ScriptedFakeLLM("Sure! all\n")
    assert classify_intent(llm, "?") == "all"


def test_classify_falls_back_on_garbage():
    llm = _ScriptedFakeLLM("asdfqwer")
    assert classify_intent(llm, "?") == "specific"


def test_classify_respects_timeout_fallback():
    """Blocking LLM must not hang class inference; always returns "specific"."""
    llm = _BlockingLLM()
    started = time.perf_counter()
    label = classify_intent(llm, "?", timeout_s=0.1)
    elapsed = time.perf_counter() - started
    assert label == "specific"
    assert elapsed < 1.0  # well under the 5 s sleep


# ---- decompose_query ----


def test_decompose_parses_json_array():
    llm = _ScriptedFakeLLM('["a", "b", "c"]')
    out = decompose_query(llm, "推荐", catalog_summary="(catalog)")
    assert out == ["a", "b", "c"]


def test_decompose_parses_bullet_list_when_no_json():
    llm = _ScriptedFakeLLM("- a\n- b\n- c")
    out = decompose_query(llm, "推荐", catalog_summary="(catalog)")
    assert out == ["a", "b", "c"]


def test_decompose_caps_at_max():
    items = ", ".join(f'"sub{i}"' for i in range(10))
    llm = _ScriptedFakeLLM(f"[{items}]")
    out = decompose_query(llm, "q", catalog_summary="(c)", max_sub_queries=3)
    assert len(out) == 3


def test_decompose_falls_back_on_empty():
    llm = _ScriptedFakeLLM("")
    out = decompose_query(llm, "原问题", catalog_summary="(c)")
    assert out == ["原问题"]


def test_decompose_dedups_preserving_order():
    llm = _ScriptedFakeLLM('["  a  ", "a", "b"]')
    out = decompose_query(llm, "q", catalog_summary="(c)")
    assert out == ["a", "b"]


def test_decompose_respects_timeout_fallback():
    """Blocking LLM must time out and fall back to [question]."""
    llm = _BlockingLLM()
    started = time.perf_counter()
    out = decompose_query(llm, "原问题", catalog_summary="(c)", timeout_s=0.1)
    elapsed = time.perf_counter() - started
    assert out == ["原问题"]
    assert elapsed < 1.0


def test_decompose_default_max_is_module_constant():
    """Sanity: the module-level default matches the documented cap."""
    assert DECOMPOSE_MAX == 8
