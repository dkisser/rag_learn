from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from rag_learn.app import _migrate_legacy_chroma, build_app
from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config
from rag_learn.eval.tracing import event_from_dict
from rag_learn.retriever.base import BaseRetriever, Hit


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
    )


# ---- _migrate_legacy_chroma ----


def test_migrate_noop_when_target_exists(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    target = config.chroma_dir / "rag_doc"
    target.mkdir()
    marker = config.chroma_dir / ".migrated"
    # Drop a fake legacy file to prove it isn't touched
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")

    _migrate_legacy_chroma(config)

    assert (config.chroma_dir / "chroma.sqlite3").exists()  # untouched
    assert not marker.exists()


def test_migrate_moves_sqlite_and_uuid_dirs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")
    uuid_dir = config.chroma_dir / "01234567-89ab-cdef-0123-456789abcdef"
    uuid_dir.mkdir()
    (uuid_dir / "index.bin").write_bytes(b"\x00" * 4)

    _migrate_legacy_chroma(config)

    target = config.chroma_dir / "rag_doc"
    assert target.is_dir()
    assert (target / "chroma.sqlite3").read_text() == "legacy"
    assert (target / "01234567-89ab-cdef-0123-456789abcdef" / "index.bin").exists()
    assert not (config.chroma_dir / "chroma.sqlite3").exists()
    assert not uuid_dir.exists()
    assert (config.chroma_dir / ".migrated").exists()


def test_migrate_idempotent_via_marker(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")
    (config.chroma_dir / ".migrated").write_text("prior run")

    _migrate_legacy_chroma(config)

    target = config.chroma_dir / "rag_doc"
    assert not target.exists()  # not migrated because marker says done
    assert (config.chroma_dir / "chroma.sqlite3").exists()  # untouched


def test_migrate_noop_when_nothing_to_migrate(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)

    _migrate_legacy_chroma(config)

    assert not (config.chroma_dir / "rag_doc").exists()
    assert not (config.chroma_dir / ".migrated").exists()


def test_migrate_fail_open_on_io_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """I/O failure during migration must not crash startup."""
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("rag_learn.app.shutil.move", _boom)

    _migrate_legacy_chroma(config)

    # Marker should be written so the failing migration isn't retried every launch.
    assert (config.chroma_dir / ".migrated").exists()
    assert (config.chroma_dir / "chroma.sqlite3").exists()  # untouched after failure


# ---- build_app(catalog=...) ----


class StubRetriever:
    """Satisfies BaseRetriever Protocol without touching Chroma."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.queries: list[str] = []

    def ensure_indexed(self, docs_dir: str) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return [
            Hit(
                text=f"hit-for-{self.collection_name}",
                source_file=f"{self.collection_name}.md",
                chunk_index=0,
                score=0.1,
            )
        ]


@pytest.fixture
def stub_catalog(tmp_path: Path) -> Catalog:
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
                display_name="甲集",
                docs_dir=docs_a,
                retriever_factory=lambda d, n: StubRetriever(d, n),
            ),
            Collection(
                name="bbb",
                display_name="乙集",
                docs_dir=docs_b,
                retriever_factory=lambda d, n: StubRetriever(d, n),
            ),
        )
    )


def _stub_llm() -> Any:
    """Fake DeepSeekLLM whose .stream yields a single token."""

    class _StubLLM:
        def stream(self, system: str, user: str) -> Iterator[str]:
            yield "ok"

    return _StubLLM()


def test_build_app_with_catalog_builds_dropdown(stub_catalog: Catalog, tmp_path: Path):
    config = _make_config(tmp_path)
    app = build_app(catalog=stub_catalog, llm=_stub_llm(), config=config)
    # Gradio Blocks exposes its component tree; we check the rendered text via
    # its .config dict representation. Asserting "甲集" / "乙集" appear means
    # the Dropdown choices wired up.
    rendered = str(app.config)
    assert "甲集" in rendered
    assert "乙集" in rendered
    assert "知识库" in rendered  # the Dropdown label


def test_build_app_warns_on_failed_collections(stub_catalog: Catalog, tmp_path: Path):
    config = _make_config(tmp_path)
    app = build_app(
        catalog=stub_catalog,
        llm=_stub_llm(),
        config=config,
        warnings=[("aaa", "boom")],
    )
    rendered = str(app.config)
    assert "启动期集合 ingest 失败" in rendered
    assert "aaa" in rendered
    assert "boom" in rendered


def test_build_app_logs_event_to_jsonl(stub_catalog: Catalog, tmp_path: Path):
    config = _make_config(tmp_path)
    app = build_app(catalog=stub_catalog, llm=_stub_llm(), config=config)

    submit_fn = app.fns[0].fn
    outputs = submit_fn("aaa", "hello")
    assert isinstance(outputs, list) and len(outputs) == 5

    files = list(tmp_path.glob("data/rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        data = json.loads(f.readline())
    event = event_from_dict(data)
    assert event.collection == "aaa"
    assert event.question == "hello"
    assert event.answer == "ok"


# ---- launch() behavior (no real Gradio launch) ----


def test_launch_filters_failed_collections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If one collection's retriever factory raises, build_app must not see it."""
    from rag_learn import app as app_module

    good_dir = tmp_path / "docs_good"
    good_dir.mkdir()
    (good_dir / "x.md").write_text("# X\n\nhi")

    def boom_factory(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    good = Collection(
        name="good",
        display_name="好集",
        docs_dir=good_dir,
        retriever_factory=lambda d, n: StubRetriever(d, n),
    )
    bad = Collection(
        name="bad",
        display_name="坏集",
        docs_dir=good_dir,  # re-use; factory never gets there
        retriever_factory=boom_factory,
    )
    catalog = Catalog(collections=(good, bad))

    # Stub out Gradio launch so this test doesn't bind a port.
    launched = {"called": False}

    def fake_launch(self, *args, **kwargs):
        launched["called"] = True

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    monkeypatch.setattr(gr.Blocks, "queue", lambda self: self)

    # Stub the LLM and config so launch() doesn't hit the network or read .env.
    fake_llm = _stub_llm()

    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    built = {}

    def fake_build_app(catalog, llm, config, warnings=None, reranker=None):  # type: ignore[no-untyped-def]
        built["catalog_names"] = catalog.names()
        built["warnings"] = warnings or []
        built["reranker"] = reranker
        return gr.Blocks()  # empty Blocks is fine for this test

    monkeypatch.setattr(app_module, "build_app", fake_build_app)
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "DeepSeekLLM", lambda **_kw: fake_llm)
    monkeypatch.setattr(app_module, "build_catalog", lambda: catalog)

    app_module.launch()

    assert launched["called"], "Gradio launch() should have been called"
    assert built["catalog_names"] == ["good"], "failed collection must be filtered out"
    assert len(built["warnings"]) == 1
    assert built["warnings"][0][0] == "bad"


def test_launch_exits_when_all_collections_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from rag_learn import app as app_module

    def boom_factory(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    good_dir = tmp_path / "docs"
    good_dir.mkdir()
    (good_dir / "x.md").write_text("# X\n\nhi")
    catalog = Catalog(
        collections=(
            Collection(
                name="bad1",
                display_name="坏1",
                docs_dir=good_dir,
                retriever_factory=boom_factory,
            ),
            Collection(
                name="bad2",
                display_name="坏2",
                docs_dir=good_dir,
                retriever_factory=boom_factory,
            ),
        )
    )

    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "DeepSeekLLM", lambda **_kw: _stub_llm())
    monkeypatch.setattr(app_module, "build_catalog", lambda: catalog)

    with pytest.raises(SystemExit, match="所有 collection ingest 失败"):
        app_module.launch()
