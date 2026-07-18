from __future__ import annotations

from rag_learn.app import build_app
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
