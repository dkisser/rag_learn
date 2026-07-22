from collections.abc import Iterator

from rag_learn.eval.tracing import ListEmitter, RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.pipeline import answer_stream, build_prompt
from rag_learn.retriever import Hit


def _hits() -> list[Hit]:
    return [
        Hit(text="alpha content", source_file="a.md", chunk_index=0, score=0.10),
        Hit(text="beta content", source_file="b.md", chunk_index=3, score=0.20),
    ]


def test_build_prompt_returns_system_and_user():
    sys_msg, user_msg = build_prompt(_hits(), "什么是 RAG？")
    assert isinstance(sys_msg, str) and sys_msg
    assert isinstance(user_msg, str) and user_msg


def test_build_prompt_includes_question():
    _, user_msg = build_prompt(_hits(), "什么是 RAG？")
    assert "什么是 RAG？" in user_msg


def test_build_prompt_lists_each_chunk_with_source():
    sys_msg, user_msg = build_prompt(_hits(), "Q")
    assert "[1] (来源: a.md) alpha content" in user_msg
    assert "[2] (来源: b.md) beta content" in user_msg
    assert "alpha content" in sys_msg or "上下文" in sys_msg or "回答" in sys_msg


def test_build_prompt_truncates_long_chunks():
    long = "x" * 5000
    h = [Hit(text=long, source_file="long.md", chunk_index=0, score=0.0)]
    _, user_msg = build_prompt(h, "Q")
    # 5000 chars truncated to CHUNK_DISPLAY_CHARS (600)
    assert user_msg.count("x") == 600


def test_build_prompt_empty_hits_has_empty_prompt_branch():
    sys_msg, _ = build_prompt([], "Q")
    assert "RAG 助手" in sys_msg


def test_build_prompt_numbering_starts_at_one():
    _, user_msg = build_prompt(_hits(), "Q")
    assert "[1]" in user_msg
    assert "[2]" in user_msg
    assert "[0]" not in user_msg and "[3]" not in user_msg


class _FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self._hits


class _FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield from self._tokens


def test_answer_stream_emits_event_when_perf_fn_called_with_answer() -> None:
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits)}
    llm = _FakeLLM(["hello", "world"])
    emitter = ListEmitter()

    out = answer_stream(
        retrievers,
        llm,
        "Q?",
        k=5,
        emitter=emitter,
        metadata={"llm_model": "dummy"},
    )

    stream, retrieved_hits, perf_fn = out["chroma"]
    answer = "".join(stream)
    perf = perf_fn(answer)

    assert isinstance(perf, StreamPerf)
    assert len(emitter.events) == 1
    event = emitter.events[0]
    assert isinstance(event, RAGEvent)
    assert event.collection == "chroma"
    assert event.question == "Q?"
    assert event.answer == answer
    assert event.hits == tuple(retrieved_hits)
    assert event.metadata == {"k": 5, "llm_model": "dummy"}


def test_answer_stream_no_emitter_does_not_record() -> None:
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits)}
    llm = _FakeLLM(["ok"])

    out = answer_stream(retrievers, llm, "Q?")
    stream, _, perf_fn = out["chroma"]
    answer = "".join(stream)
    perf = perf_fn(answer)

    assert isinstance(perf, StreamPerf)
