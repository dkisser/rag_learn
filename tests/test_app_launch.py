from __future__ import annotations

from types import SimpleNamespace

from rag_learn.app import (
    _flatten_output_targets,
    _flatten_output_values,
    build_app,
)
from rag_learn.config import Config
from rag_learn.llm import DeepSeekLLM
from rag_learn.retriever.chroma_impl import ChromaRetriever
from rag_learn.retriever.milvus_impl import MilvusRetriever


def _cfg(monkeypatch) -> Config:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from rag_learn.config import load_config

    cfg = load_config()
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_build_app_constructs_without_launching(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"
    chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(monkeypatch)
    chroma = ChromaRetriever(persist_dir=cfg.chroma_dir)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())  # not actually called

    app = build_app(retrievers={"chroma": chroma, "milvus": milvus}, llm=llm, config=cfg)
    assert app is not None


def test_build_app_returns_gradio_blocks(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"
    chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(monkeypatch)
    chroma = ChromaRetriever(persist_dir=cfg.chroma_dir)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())

    import gradio as gr

    app = build_app(retrievers={"chroma": chroma, "milvus": milvus}, llm=llm, config=cfg)
    assert isinstance(app, gr.Blocks)


def test_build_app_with_warnings_constructs(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"
    chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(monkeypatch)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())

    app = build_app(
        retrievers={"milvus": milvus},
        llm=llm,
        config=cfg,
        warnings=[("chroma", "model download failed")],
    )
    assert app is not None


# ---------------------------------------------------------------------------
# _flatten_output_targets / _flatten_output_values
#
# The bug: submit.click(..., outputs=[]) made Gradio drop every per-component
# .value mutation from on_submit, so the UI never updated. These helpers
# pin down the contract that on_submit returns exactly the values for the
# declared output targets.
# ---------------------------------------------------------------------------


def test_flatten_output_targets_chroma_only_has_four_components():
    panels = {"chroma": {"bot": "b", "chunks": "c", "perf": "p"}}
    assert _flatten_output_targets("q", panels, ["chroma"]) == ["q", "b", "c", "p"]


def test_flatten_output_targets_two_retrievers_seven_components():
    panels = {
        "chroma": {"bot": "b1", "chunks": "c1", "perf": "p1"},
        "milvus": {"bot": "b2", "chunks": "c2", "perf": "p2"},
    }
    assert _flatten_output_targets("q", panels, ["chroma", "milvus"]) == [
        "q",
        "b1",
        "c1",
        "p1",
        "b2",
        "c2",
        "p2",
    ]


def test_flatten_output_values_matches_targets_in_order():
    panels = {
        "chroma": {
            "bot": SimpleNamespace(value=[{"role": "assistant", "content": "hi"}]),
            "chunks": SimpleNamespace(value="**chunks**"),
            "perf": SimpleNamespace(value="42ms"),
        },
    }
    assert _flatten_output_values("", panels, ["chroma"]) == [
        "",
        [{"role": "assistant", "content": "hi"}],
        "**chunks**",
        "42ms",
    ]


def test_flatten_output_values_clear_question_with_empty_string():
    panels = {
        "chroma": {
            "bot": SimpleNamespace(value=[]),
            "chunks": SimpleNamespace(value=""),
            "perf": SimpleNamespace(value=""),
        }
    }
    assert _flatten_output_values("", panels, ["chroma"])[0] == ""
