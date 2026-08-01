"""End-to-end smoke: multi-collection catalog flows through the pipeline with a mocked LLM.

Exercises the full pipeline (load_config → Catalog → StubRetriever →
answer_stream → DeepSeekLLM). The LLM is replaced with a fake that yields one
OpenAI-shaped chunk so we can assert the streaming contract end-to-end without
calling DeepSeek.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_learn.app import build_app
from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config, load_config
from rag_learn.llm import DeepSeekLLM
from rag_learn.pipeline import answer_stream
from rag_learn.retriever.base import Hit


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

    def __call__(self, system: str, user: str) -> Iterator[_FakeChunk]:
        # Yield an OpenAI-shaped chunk so DeepSeekLLM.stream can decode it.
        self.calls += 1
        yield _FakeChunk("TEST ANSWER")


class _AltStub:
    """Returns collection-specific hits so we can prove selection matters."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.collection_name = collection_name

    def ensure_indexed(self, docs_dir: str) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [
            Hit(
                text=f"hit-from-{self.collection_name}-for-{query}",
                source_file=f"{self.collection_name}.md",
                chunk_index=0,
                score=0.0,
            )
        ]


def _make_config(tmp_path: Path) -> Config:
    return Config(
        deepseek_api_key="dummy",
        llm_model="dummy",
        deepseek_base_url="https://example.invalid",
        retrieve_k=5,
        chunk_size=800,
        chunk_overlap=50,
        repo_root=tmp_path,
        docs_dir=tmp_path / "docs",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        milvus_path=tmp_path / "data" / "milvus.db",
        rerank_enabled=False,
        rerank_model="BAAI/bge-reranker-base",
        rerank_factor=4,
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


def _stub_llm() -> Any:
    """Fake DeepSeekLLM whose .stream yields a single token."""

    class _StubLLM:
        def stream(self, system: str, user: str) -> Iterator[str]:
            yield "ok"

    return _StubLLM()


def _two_collection_catalog(tmp_path: Path) -> Catalog:
    docs_a = tmp_path / "docs_a"
    docs_b = tmp_path / "docs_b"
    docs_a.mkdir()
    docs_b.mkdir()
    (docs_a / "x.md").write_text("# X\n\nhi")
    (docs_b / "y.md").write_text("# Y\n\nyo")
    return Catalog(
        collections=(
            Collection(
                name="aaa",
                display_name="甲",
                docs_dir=docs_a,
                retriever_factory=lambda d, n: _AltStub(d, n),
            ),
            Collection(
                name="bbb",
                display_name="乙",
                docs_dir=docs_b,
                retriever_factory=lambda d, n: _AltStub(d, n),
            ),
        )
    )


def test_e2e_full_pipeline_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = load_config()
    catalog = _two_collection_catalog(tmp_path)

    retrievers = {name: catalog.get(name).retriever for name in catalog.names()}

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

    for name in catalog.names():
        stream, hits, perf_fn = out[name]
        # Stream must be iterable.
        tokens = list(stream)
        assert tokens == ["TEST ANSWER"]
        # Hits must come from the selected collection.
        assert hits
        for h in hits:
            assert name in h.source_file, h.source_file
        # Perf must be populated.
        perf = perf_fn("TEST ANSWER")
        assert perf.total_ms >= 0
        assert perf.retrieve_ms >= 0
        assert perf.first_token_ms >= 0
        assert perf.finished_at


def test_e2e_build_app_renders_two_collections(tmp_path: Path):
    catalog = _two_collection_catalog(tmp_path)
    config = _make_config(tmp_path)
    app = build_app(catalog=catalog, llm=_stub_llm(), config=config)
    rendered = str(app.config)
    assert "甲" in rendered
    assert "乙" in rendered


def test_e2e_build_app_collection_selection_changes_chunks(tmp_path: Path):
    """Selecting a different collection must drive retrieval to that side."""
    catalog = _two_collection_catalog(tmp_path)
    config = _make_config(tmp_path)
    app = build_app(catalog=catalog, llm=_stub_llm(), config=config)

    # The first registered event is submit.click(...).
    submit_fn = app.fns[0].fn
    for slug in ("aaa", "bbb"):
        outputs = submit_fn(slug, f"question for {slug}")
        assert isinstance(outputs, list) and len(outputs) == 6
        chunks_md = outputs[4]
        assert f"{slug}.md" in str(chunks_md), (
            f"selection {slug!r} should drive retrieval from that collection"
        )


def test_e2e_build_app_empty_question_clears_output(tmp_path: Path):
    """Submitting an empty question returns empty outputs without raising."""
    catalog = _two_collection_catalog(tmp_path)
    config = _make_config(tmp_path)
    app = build_app(catalog=catalog, llm=_stub_llm(), config=config)

    submit_fn = app.fns[0].fn
    outputs = submit_fn("aaa", "   ")
    assert isinstance(outputs, list) and len(outputs) == 6
    assert outputs[1] == ""  # desc_md cleared
    assert outputs[3] == []  # bot empty


def test_e2e_build_app_unknown_collection_shows_warning(tmp_path: Path):
    """An unknown collection slug shows a warning and leaves chunks empty."""
    catalog = _two_collection_catalog(tmp_path)
    config = _make_config(tmp_path)
    app = build_app(catalog=catalog, llm=_stub_llm(), config=config)

    submit_fn = app.fns[0].fn
    outputs = submit_fn("no-such-slug", "hello")
    assert isinstance(outputs, list) and len(outputs) == 6
    assert "未知集合" in str(outputs[3])  # bot warning
    assert outputs[4] == "_（无召回）_"  # chunks empty
