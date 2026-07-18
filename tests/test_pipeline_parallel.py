from __future__ import annotations

from collections.abc import Iterator

from rag_learn.pipeline import StreamPerf, answer_stream
from rag_learn.retriever.base import Hit


class _FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.search_calls = 0

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.search_calls += 1
        return self._hits


class _FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        yield from self._tokens


def test_answer_stream_returns_both_sides():
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits), "milvus": _FakeRetriever(hits)}
    llm = _FakeLLM(["hi", "world"])
    out = answer_stream(retrievers, llm, "Q?")
    assert set(out.keys()) == {"chroma", "milvus"}
    for stream, h, perf_fn in out.values():
        assert isinstance(stream, Iterator)
        assert h == hits
        assert callable(perf_fn)
    # Drain to populate perf
    for stream, _, perf_fn in out.values():
        list(stream)
        assert isinstance(perf_fn(), StreamPerf)


def test_answer_stream_collects_tokens_in_order():
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits), "milvus": _FakeRetriever(hits)}
    llm = _FakeLLM(["a", "b", "c"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        assert list(stream) == ["a", "b", "c"]


def test_answer_stream_calls_each_retriever_and_each_llm():
    retrievers = {"chroma": _FakeRetriever([]), "milvus": _FakeRetriever([])}
    llm = _FakeLLM(["ok"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        list(stream)
    assert retrievers["chroma"].search_calls == 1
    assert retrievers["milvus"].search_calls == 1
    assert len(llm.calls) == 2


def test_answer_stream_empty_hits_still_yields_tokens():
    retrievers = {"chroma": _FakeRetriever([]), "milvus": _FakeRetriever([])}
    llm = _FakeLLM(["empty"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        assert list(stream) == ["empty"]
