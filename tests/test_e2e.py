"""End-to-end smoke: real 25 markdown docs go through the Chroma pipeline with a mocked LLM.

Exercises the full pipeline (load_config → ChromaRetriever →
answer_stream → DeepSeekLLM). The LLM is replaced with a fake that yields one
OpenAI-shaped chunk so we can assert the streaming contract end-to-end without
calling DeepSeek.

NOTE: The brief specified exercising MilvusRetriever in parallel with Chroma,
but the in-tree Milvus adapter (milvus-lite 3.1.0 + pymilvus 2.6.17) hangs on
all subsequent gRPC calls after a 433-row insert — search(), has_collection(),
release_collection(), even load_collection() all deadlock. Verified locally
with minimal repro (see Task 12 report). Milvus coverage is already exercised
end-to-end by tests/test_milvus_retriever.py against the small fixture; the
real-25-docs path is exercised here via Chroma only.
"""

from __future__ import annotations

from pathlib import Path

from rag_learn.config import load_config
from rag_learn.llm import DeepSeekLLM
from rag_learn.pipeline import answer_stream
from rag_learn.retriever.chroma_impl import ChromaRetriever

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "rag_doc"


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = type("Delta", (), {"content": content})()


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)] if content is not None else []


class _FakeStream:
    """Counts how many tokens were requested and emits a canned answer."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str):
        # Yield an OpenAI-shaped chunk so DeepSeekLLM.stream can decode it.
        self.calls += 1
        yield _FakeChunk("TEST ANSWER")


def test_e2e_full_pipeline_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = load_config()
    chroma_p = tmp_path / "chroma"
    chroma_p.mkdir()

    chroma = ChromaRetriever(persist_dir=chroma_p)
    chroma.ensure_indexed(str(DOCS_DIR))

    retrievers = {"chroma": chroma}

    fake = _FakeStream()

    # Wrap fake so DeepSeekLLM.stream uses it.
    class _FakeChatCompletions:
        def create(self, **kwargs):
            return fake(kwargs["messages"][1]["content"], "")

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    llm = DeepSeekLLM(api_key="k", model=cfg.llm_model, client=_FakeClient())

    out = answer_stream(retrievers, llm, "什么是 RAG？", k=cfg.retrieve_k)

    for name in ("chroma",):
        stream, hits, perf_fn = out[name]
        # Stream must be iterable.
        tokens = list(stream)
        assert tokens == ["TEST ANSWER"]
        # Hits must come from real docs in docs/rag_doc.
        assert hits
        for h in hits:
            assert (DOCS_DIR / h.source_file).exists(), h.source_file
        # Perf must be populated.
        perf = perf_fn()
        assert perf.total_ms >= 0
        assert perf.retrieve_ms >= 0
        assert perf.first_token_ms >= 0
        assert perf.finished_at
