"""Gradio UI: collection dropdown + single answer panel with chunks and perf metrics."""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from rag_learn.collections import Catalog, CollectionNotFoundError, build_catalog
from rag_learn.config import Config, ConfigError, load_config
from rag_learn.eval import JSONLEmitter
from rag_learn.llm import DeepSeekLLM
from rag_learn.perf import StreamPerf
from rag_learn.pipeline import answer_stream
from rag_learn.reranker import build_reranker
from rag_learn.retriever import Hit

# from rag_learn.retriever.milvus_impl import MilvusRetriever

logger = logging.getLogger(__name__)


def _format_chunks(hits: list[Hit]) -> str:
    if not hits:
        return "_（无召回）_"
    lines = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.text[:200] + "…") if len(h.text) > 200 else h.text
        lines.append(
            f"**[{i}]** `{h.source_file}#{h.chunk_index}` (score={h.score:.4f})\n\n{snippet}\n\n---"
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


def _format_routing(config: Config, metadata: dict[str, Any]) -> str:
    """Render a one-line caption summarizing routing decisions.

    Reads the metadata dict that ``answer_stream`` populated in-place with
    ``intent`` / ``sub_queries`` / ``target_collections`` / ``merged_k``.
    Falls back to "routing disabled" when the intent classifier is off.
    """
    if not config.intent_enabled:
        return "_（routing 已关闭 — 设 `INTENT_ENABLED=true` 开启）_"
    intent = metadata.get("intent", "specific")
    sub_q = metadata.get("sub_queries", [])
    target = metadata.get("target_collections", [])
    merged = metadata.get("merged_k", "?")
    return (
        f"intent: `{intent}` · sub-queries: `{len(sub_q)}` · "
        f"target: `[{', '.join(target)}]` · unique hits: `{merged}`"
    )


def _drain_to_chatbot(stream: Iterator[str]) -> str:
    """Consume an answer_stream's iterator and return the joined text."""
    return "".join(list(stream)) or "_（无输出）_"


def build_app(
    catalog: Catalog,
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
    reranker: Any = None,
) -> gr.Blocks:
    """Construct the Gradio UI but do not launch it.

    Args:
        catalog: The collection catalog to render in the dropdown.
        llm: LLM instance compatible with answer_stream.
        config: Application config for display metadata.
        warnings: Optional list of (collection_name, error_message) for
            collections that failed ingest during startup.
        reranker: Optional reranker to refine retrieval results.
    """
    emitter = JSONLEmitter(config.data_dir)

    choices = catalog.display_choices()
    default_slug = choices[0][1] if choices else None

    with gr.Blocks(title="RAG 多集合问答") as app:
        if warnings:
            warn_md = "\n".join(f"- **{name}**: {msg}" for name, msg in warnings)
            gr.Markdown(f"⚠ **启动期集合 ingest 失败**：\n\n{warn_md}")

        rerank_status = f"Rerank: `{config.rerank_model}`" if reranker else "Rerank: off"
        hybrid_status = (
            f"Hybrid: on (RRF k={config.hybrid_rrf_k})" if config.hybrid_enabled else "Hybrid: off"
        )
        gr.Markdown(
            f"# RAG 多集合问答\n\n"
            f"模型：`{config.llm_model}` · Top-k: `{config.retrieve_k}` · "
            f"Chunk: `{config.chunk_size}` chars · {rerank_status} · {hybrid_status}\n\n"
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
        desc_md = gr.Markdown()

        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            clear = gr.Button("清空")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 回答")
                routing_md = gr.Markdown("_（routing 信息提交后展示）_")
                bot = gr.Chatbot(label="答案", height=400, type="messages")
                with gr.Accordion("检索到的 chunks", open=False):
                    chunks_md = gr.Markdown("_提交问题后展示_")
                perf_md = gr.Markdown(_format_perf(None))

        def on_submit(collection_slug: str, q: str) -> list[Any]:
            empty_outputs: list[Any] = [
                gr.update(value=""),
                "",
                "_（routing 信息提交后展示）_",
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
                return [
                    gr.update(value=""),
                    "",
                    "_（routing 信息提交后展示）_",
                    bot.value,
                    "_（无召回）_",
                    _format_perf(None),
                ]

            retriever = collection.retriever
            metadata: dict[str, Any] = {
                "llm_model": config.llm_model,
                "rerank_enabled": config.rerank_enabled,
                "rerank_model": config.rerank_model if config.rerank_enabled else None,
                "hybrid_enabled": config.hybrid_enabled,
                "hybrid_rrf_k": config.hybrid_rrf_k if config.hybrid_enabled else None,
            }
            try:
                outputs = answer_stream(
                    {collection_slug: retriever},
                    llm,
                    q,
                    k=config.retrieve_k,
                    emitter=emitter,
                    metadata=metadata,
                    reranker=reranker,
                    config=config,
                    catalog=catalog,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.exception("answer_stream failed")
                bot.value = [{"role": "assistant", "content": f"⚠ 流水线失败：{exc}"}]
                return [
                    gr.update(value=""),
                    collection.description,
                    "_（routing 信息提交后展示）_",
                    bot.value,
                    "_（无召回）_",
                    _format_perf(None),
                ]

            bot.value = bot.value + [{"role": "user", "content": q}]
            stream_iter, hits, perf_fn = outputs[collection_slug]
            chunks_md.value = _format_chunks(hits)
            # Render the routing caption from the metadata dict that
            # ``answer_stream`` populated in-place.
            routing_md.value = _format_routing(config, metadata)
            try:
                answer_text = _drain_to_chatbot(stream_iter)
            except Exception as exc:  # noqa: BLE001 — spec §7 RetrievalError
                logger.exception("retrieval / LLM stream failed for side=%s", collection_slug)
                bot.value = bot.value + [{"role": "assistant", "content": f"⚠ 检索失败：{exc}"}]
                perf_md.value = _format_perf(None)
                return [
                    gr.update(value=""),
                    collection.description,
                    routing_md.value,
                    bot.value,
                    chunks_md.value,
                    perf_md.value,
                ]

            perf = perf_fn(answer_text)
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
            return [
                gr.update(value=""),
                collection.description,
                routing_md.value,
                bot.value,
                chunks_md.value,
                perf_md.value,
            ]

        submit.click(
            on_submit,
            inputs=[collection_dd, question],
            outputs=[question, desc_md, routing_md, bot, chunks_md, perf_md],
        )

        def on_clear() -> Any:
            bot.value = []
            chunks_md.value = "_提交问题后展示_"
            perf_md.value = _format_perf(None)
            desc_md.value = ""
            routing_md.value = "_（routing 信息提交后展示）_"
            return gr.update(value="")

        clear.click(on_clear, inputs=[], outputs=[question])

    return app


def launch() -> None:
    """Production entry: load config, build catalog + LLM, migrate, ingest, serve."""

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"启动失败: {exc}") from exc

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_chroma(config)

    llm = DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )

    catalog = build_catalog(
        hybrid_enabled=config.hybrid_enabled,
        hybrid_rrf_k=config.hybrid_rrf_k,
    )
    logger.info("Catalog: %d collections %s", len(catalog.collections), catalog.names())
    raw_warnings = catalog.ensure_all_indexed()
    failed_names = {name for name, _ in raw_warnings}
    for c in catalog.collections:
        if c.name not in failed_names:
            logger.info("Collection ready: %s", c.name)
    for name, msg in raw_warnings:
        logger.warning("Collection ingest failed: %s - %s", name, msg)
    working = Catalog(
        collections=tuple(c for c in catalog.collections if c.name not in failed_names)
    )
    if not working.names():
        raise SystemExit("所有 collection ingest 失败，无法启动")

    reranker = build_reranker(config)
    if reranker:
        logger.info("Reranker enabled: %s", config.rerank_model)

    app = build_app(
        catalog=working,
        llm=llm,
        config=config,
        warnings=raw_warnings,
        reranker=reranker,
    )
    # Disable Gradio's analytics daemon — see CLAUDE.md macOS ARM note.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app.queue().launch(server_name="127.0.0.1", server_port=7860)


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
    try:
        for src in legacy:
            shutil.move(str(src), str(target / src.name))
        marker.write_text("migrated\n", encoding="utf-8")
        logger.info("Migrated legacy Chroma data: %d entries -> %s", len(legacy), target)
    except Exception as exc:  # noqa: BLE001 — migration failure is fail-open per spec §7
        logger.warning("Legacy Chroma migration failed: %s; continuing startup", exc)
        try:
            marker.write_text("failed\n", encoding="utf-8")
        except Exception:  # noqa: BLE001 — best-effort marker to avoid retry
            pass
