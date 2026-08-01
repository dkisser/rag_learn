from collections.abc import Iterator
from pathlib import Path

from rag_learn.config import Config
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
        self.last_k: int | None = None

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.last_k = k
        return self._hits


class _FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return sorted(
            hits,
            key=lambda h: self._scores.get(h.text, 0.0),
            reverse=True,
        )


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


def test_answer_stream_reranker_reorders_and_truncates_to_k() -> None:
    hits = [
        Hit(text="irrelevant", source_file="generic.md", chunk_index=0, score=0.1),
        Hit(text="relevant", source_file="target.md", chunk_index=1, score=0.5),
        Hit(text="also relevant", source_file="target.md", chunk_index=2, score=0.4),
    ]
    retriever = _FakeRetriever(hits)
    llm = _FakeLLM(["ok"])
    reranker = _FakeReranker({"relevant": 1.0, "also relevant": 0.8, "irrelevant": 0.0})

    out = answer_stream(
        {"chroma": retriever},
        llm,
        "Q?",
        k=1,
        reranker=reranker,
    )
    _, final_hits, _ = out["chroma"]
    assert len(final_hits) == 1
    assert final_hits[0].text == "relevant"


def test_answer_stream_with_config_over_fetches_candidates() -> None:
    hits = [
        Hit(text="a", source_file="a.md", chunk_index=0, score=0.1),
        Hit(text="b", source_file="b.md", chunk_index=1, score=0.2),
        Hit(text="c", source_file="c.md", chunk_index=2, score=0.3),
        Hit(text="d", source_file="d.md", chunk_index=3, score=0.4),
    ]
    retriever = _FakeRetriever(hits)
    llm = _FakeLLM(["ok"])
    config = Config(
        deepseek_api_key="k",
        llm_model="m",
        deepseek_base_url="u",
        retrieve_k=2,
        chunk_size=800,
        chunk_overlap=50,
        repo_root=Path(__file__).parent.parent / "src",
        docs_dir=Path(__file__).parent.parent / "docs" / "rag_doc",
        data_dir=Path(__file__).parent.parent / "data",
        chroma_dir=Path(__file__).parent.parent / "data" / "chroma",
        milvus_path=Path(__file__).parent.parent / "data" / "milvus.db",
        rerank_enabled=True,
        rerank_model="BAAI/bge-reranker-base",
        rerank_factor=2,
        rerank_k=None,
        rerank_batch_size=8,
        rerank_device=None,
        hybrid_enabled=False,
        hybrid_rrf_k=60,
        intent_enabled=False,
        intent_timeout_s=8.0,
        decompose_enabled=False,
        decompose_timeout_s=15.0,
        decompose_max=8,
        catalog_sub_k=20,
        catalog_recall_k=20,
    )

    out = answer_stream(
        {"chroma": retriever},
        llm,
        "Q?",
        k=2,
        config=config,
    )
    _ = "".join(out["chroma"][0])
    assert retriever.last_k == 4  # k * factor
    assert len(out["chroma"][1]) == 2  # but prompt only sees final_k


def test_answer_stream_no_reranker_uses_vector_order() -> None:
    hits = [
        Hit(text="first", source_file="a.md", chunk_index=0, score=0.1),
        Hit(text="second", source_file="b.md", chunk_index=1, score=0.2),
    ]
    retriever = _FakeRetriever(hits)
    llm = _FakeLLM(["ok"])

    out = answer_stream({"chroma": retriever}, llm, "Q?", k=2)
    _, final_hits, _ = out["chroma"]
    assert [h.text for h in final_hits] == ["first", "second"]
    assert retriever.last_k == 2
