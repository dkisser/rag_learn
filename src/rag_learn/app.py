"""Gradio UI: side-by-side streams with per-side chunks panels + perf metrics."""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from rag_learn.collections import Catalog, CollectionNotFoundError
from rag_learn.config import Config
from rag_learn.pipeline import StreamPerf, answer_stream
from rag_learn.retriever import Hit
from rag_learn.retriever.base import BaseRetriever

# from rag_learn.retriever.milvus_impl import MilvusRetriever

logger = logging.getLogger(__name__)


def _format_chunks(hits: list[Hit]) -> str:
    if not hits:
        return "_（无召回）_"
    lines = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.text[:200] + "…") if len(h.text) > 200 else h.text
        lines.append(
            f"**[{i}]** `{h.source_file}#{h.chunk_index}` (dist={h.score:.4f})\n\n{snippet}\n\n---"
        )
    return "\n\n".join(lines)


def _format_perf(perf: StreamPerf | None) -> str:
    if perf is None:
        return "检索 · ms · 首个 token · ms · 总 · ms · 完成于 …"
    return (
        f"检索 {perf.retrieve_ms:.0f}ms · "
        f"首个 token {perf.first_token_ms:.0f}ms · "
        f"总 {perf.total_ms:.0f}ms · "
        f"完成于 {perf.finished_at}"
    )


def _drain_to_chatbot(stream: Iterator[str]) -> str:
    """Consume an answer_stream's iterator and return the joined text."""
    return "".join(list(stream)) or "_（无输出）_"


def build_app(
    catalog: Catalog,
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks:
    """Construct the Gradio UI but do not launch it."""
    choices = catalog.display_choices()
    default_slug = choices[0][1] if choices else None

    with gr.Blocks(title="RAG 多集合问答") as app:
        if warnings:
            warn_md = "\n".join(f"- **{name}**: {msg}" for name, msg in warnings)
            gr.Markdown(f"⚠ **启动期集合 ingest 失败**：\n\n{warn_md}")

        gr.Markdown(
            f"# RAG 多集合问答\n\n"
            f"模型：`{config.llm_model}` · Top-k: `{config.retrieve_k}` · "
            f"Chunk: `{config.chunk_size}` chars\n\n"
            "选择知识库 → 输入问题 → 流式生成回答。"
        )
        with gr.Row():
            collection_dd = gr.Dropdown(
                choices=choices,
                label="知识库",
                value=default_slug,
            )
            question = gr.Textbox(
                label="问题",
                placeholder="例如：什么是 GraphRAG？",
                lines=2,
                scale=3,
            )
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            clear = gr.Button("清空")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 回答")
                bot = gr.Chatbot(label="答案", height=400, type="messages")
                with gr.Accordion("检索到的 chunks", open=False):
                    chunks_md = gr.Markdown("_提交问题后展示_")
                perf_md = gr.Markdown(_format_perf(None))

        def on_submit(collection_slug: str, q: str) -> list[Any]:
            empty_outputs: list[Any] = [
                gr.update(value=""),
                [],
                "_（无召回）_",
                _format_perf(None),
            ]
            if not q.strip():
                return empty_outputs
            try:
                collection = catalog.get(collection_slug)
            except CollectionNotFoundError:
                logger.warning("Unknown collection slug: %r", collection_slug)
                bot.value = [{"role": "assistant", "content": f"⚠ 未知集合：{collection_slug}"}]
                return [gr.update(value=""), bot.value, "_（无召回）_", _format_perf(None)]

            retriever = collection.retriever
            try:
                outputs = answer_stream(
                    {collection_slug: retriever},
                    llm,
                    q,
                    k=config.retrieve_k,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.exception("answer_stream failed")
                bot.value = [{"role": "assistant", "content": f"⚠ 流水线失败：{exc}"}]
                return [gr.update(value=""), bot.value, "_（无召回）_", _format_perf(None)]

            bot.value = bot.value + [{"role": "user", "content": q}]
            stream_iter, hits, perf_fn = outputs[collection_slug]
            chunks_md.value = _format_chunks(hits)
            try:
                answer_text = _drain_to_chatbot(stream_iter)
            except Exception as exc:  # noqa: BLE001 — spec §7 RetrievalError
                logger.exception("retrieval / LLM stream failed for side=%s", collection_slug)
                bot.value = bot.value + [{"role": "assistant", "content": f"⚠ 检索失败：{exc}"}]
                perf_md.value = _format_perf(None)
                return [gr.update(value=""), bot.value, chunks_md.value, perf_md.value]

            perf = perf_fn()
            logger.info(
                "[%s] %-12s retrieve=%dms first_token=%dms total=%dms",
                perf.finished_at,
                collection_slug,
                int(perf.retrieve_ms),
                int(perf.first_token_ms),
                int(perf.total_ms),
            )
            bot.value = bot.value + [{"role": "assistant", "content": answer_text}]
            perf_md.value = _format_perf(perf)
            return [gr.update(value=""), bot.value, chunks_md.value, perf_md.value]

        submit.click(
            on_submit,
            inputs=[collection_dd, question],
            outputs=[question, bot, chunks_md, perf_md],
        )

        def on_clear() -> Any:
            bot.value = []
            chunks_md.value = "_提交问题后展示_"
            perf_md.value = _format_perf(None)
            return gr.update(value="")

        clear.click(on_clear, inputs=[], outputs=[question])

    return app


def launch() -> None:
    """Production entry: load config, build real retrievers + LLM, ingest, serve."""
    from rag_learn.config import ConfigError, load_config
    from rag_learn.llm import DeepSeekLLM
    from rag_learn.retriever.chroma_impl import ChromaRetriever

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"启动失败: {exc}") from exc

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    llm = DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )

    retrievers: dict[str, BaseRetriever] = {}
    warnings: list[tuple[str, str]] = []

    # Per-retriever ingestion is fail-open (see spec §5.5).
    for name, factory in [
        ("chroma", lambda: ChromaRetriever(persist_dir=config.chroma_dir)),
        # ("milvus", lambda: MilvusRetriever(db_path=config.milvus_path, dim=384)),
    ]:
        try:
            r = factory()
            r.ensure_indexed(str(config.docs_dir))
            retrievers[name] = r
            logger.info("[%s] %s ready", _ts(), name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] %s ingestion failed: %s", _ts(), name, exc)
            warnings.append((name, str(exc)))

    if not retrievers:
        raise SystemExit("两个 retriever 都没准备好，无法启动")

    app = build_app(retrievers=retrievers, llm=llm, config=config, warnings=warnings)
    # Disable Gradio's analytics daemon: it spawns background threads that
    # call uuid4()/os.urandom during startup. On macOS ARM with milvus-lite's
    # gRPC server already holding threads + file descriptors from a fork,
    # those concurrent os.urandom() calls segfault the interpreter before
    # launch() can bind 127.0.0.1:7860. Gradio 5.50 has no launch() kwarg
    # for this, so we set the env var it consults internally.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
    )


def _ts() -> str:
    import time

    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


# ---- Legacy migration ----


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _migrate_legacy_chroma(config: Config) -> None:
    """一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。

    触发条件：data/chroma/rag_doc/ 不存在 AND 没有 .migrated 标记 AND
    chroma.sqlite3 或 UUID 子目录存在。
    幂等：迁移完成后写 data/chroma/.migrated。
    """
    target = config.chroma_dir / "rag_doc"
    marker = config.chroma_dir / ".migrated"
    if target.exists() or marker.exists():
        return
    if not config.chroma_dir.exists():
        return

    legacy: list[Path] = [
        *config.chroma_dir.glob("chroma.sqlite3"),
        *(p for p in config.chroma_dir.iterdir() if p.is_dir() and _UUID_RE.match(p.name)),
    ]
    if not legacy:
        return

    target.mkdir(parents=True, exist_ok=True)
    for src in legacy:
        shutil.move(str(src), str(target / src.name))
    marker.write_text("migrated\n", encoding="utf-8")
    logger.info("Migrated legacy Chroma data: %d entries -> %s", len(legacy), target)
