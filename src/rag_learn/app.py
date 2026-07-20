"""Gradio UI: side-by-side streams with per-side chunks panels + perf metrics."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import gradio as gr

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


def _flatten_output_targets(
    question: Any,
    panels: dict[str, dict[str, Any]],
    retriever_names: list[str],
) -> list[Any]:
    """Flat list of components that on_submit must return values for, in order.

    Order: question first (so it clears on submit), then per-retriever
    (bot, chunks, perf). Mirrors _flatten_output_values.

    A previous version declared `outputs=[]` on submit.click and relied on
    direct .value mutation. Gradio only re-renders declared outputs, so
    those mutations never reached the UI — submit appeared to do nothing.
    """
    targets: list[Any] = [question]
    for name in retriever_names:
        targets.extend(
            [
                panels[name]["bot"],
                panels[name]["chunks"],
                panels[name]["perf"],
            ]
        )
    return targets


def _flatten_output_values(
    question_value: str,
    panels: dict[str, dict[str, Any]],
    retriever_names: list[str],
) -> list[Any]:
    """Parallel to _flatten_output_targets: current .value of each component."""
    values: list[Any] = [question_value]
    for name in retriever_names:
        values.extend(
            [
                panels[name]["bot"].value,
                panels[name]["chunks"].value,
                panels[name]["perf"].value,
            ]
        )
    return values


def build_app(
    retrievers: dict[str, BaseRetriever],
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks:
    """Construct the Gradio UI but do not launch it.

    `warnings` is a list of (side_name, error_message) for retrievers whose
    ingestion failed at startup (spec §5.5). When non-empty, a red banner
    is rendered above the panels; failed sides are excluded from `retrievers`
    upstream by `launch()`.
    """
    retriever_names = list(retrievers.keys())

    with gr.Blocks(title="RAG Compare: Chroma vs Milvus") as app:
        if warnings:
            warn_md = "\n".join(f"- **{name}**: {msg}" for name, msg in warnings)
            gr.Markdown(f"⚠ **启动期侧 ingest 失败**（spec §5.5 fail-open）：\n\n{warn_md}")

        gr.Markdown(
            f"# RAG 多 Retriever 对比\n\n"
            f"模型：`{config.llm_model}` · Top-k: `{config.retrieve_k}` · "
            f"Chunk: `{config.chunk_size}` chars\n\n"
            "输入问题 → 两侧并行检索 + 双侧流式生成 → 并排展示。"
        )
        with gr.Row():
            question = gr.Textbox(
                label="问题",
                placeholder="例如：什么是 GraphRAG？",
                lines=2,
            )
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            clear = gr.Button("清空")

        with gr.Row():
            panels: dict[str, dict[str, Any]] = {}
            for name in retriever_names:
                with gr.Column():
                    gr.Markdown(f"## {name.upper()}")
                    # type='messages' picks the openai-style dict format
                    # (role/content) — matches what on_submit appends below.
                    # The default 'tuples' format is deprecated in 5.x and
                    # emits a UserWarning at every Chatbot() construction.
                    bot = gr.Chatbot(
                        label=f"{name} 答案",
                        height=400,
                        type="messages",
                    )
                    with gr.Accordion("检索到的 chunks", open=False):
                        chunks_md = gr.Markdown("_提交问题后展示_")
                    perf_md = gr.Markdown(_format_perf(None))
                    panels[name] = {
                        "bot": bot,
                        "chunks": chunks_md,
                        "perf": perf_md,
                    }

        def on_submit(q: str) -> list[Any]:
            if not q.strip():
                return _flatten_output_values("", panels, retriever_names)
            try:
                outputs = answer_stream(retrievers, llm, q, k=config.retrieve_k)
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.exception("answer_stream failed")
                # Show a single banner-style error in the first chatbot.
                first = retriever_names[0]
                panels[first]["bot"].value = [
                    {"role": "assistant", "content": f"⚠ 流水线失败：{exc}"}
                ]
                return _flatten_output_values("", panels, retriever_names)

            # Display user question in each chatbot as history seed.
            for name in retriever_names:
                bot = panels[name]["bot"]
                bot.value = bot.value + [{"role": "user", "content": q}]

            # Known limitation: this handler batch-consumes each stream iterator
            # via _drain_to_chatbot before updating the UI. The user sees nothing
            # until BOTH sides complete. True per-token streaming (spec §5.2
            # "threaded streams") would require converting on_submit to a generator
            # that yields per-token updates via Gradio's response stream API.
            # TODO: convert to incremental streaming once the demo UX matters.
            # Tracked for the final-review follow-up list.
            # Per-side stream + chunk + perf updates.
            for name in retriever_names:
                stream_iter, hits, perf_fn = outputs[name]
                panels[name]["chunks"].value = _format_chunks(hits)
                try:
                    answer_text = _drain_to_chatbot(stream_iter)
                except Exception as exc:  # noqa: BLE001 — spec §7 RetrievalError
                    logger.exception("retrieval / LLM stream failed for side=%s", name)
                    panels[name]["bot"].value = panels[name]["bot"].value + [
                        {"role": "assistant", "content": f"⚠ 检索失败：{exc}"}
                    ]
                    panels[name]["perf"].value = _format_perf(None)
                    continue
                perf = perf_fn()
                logger.info(
                    "[%s] %-7s retrieve=%dms first_token=%dms total=%dms",
                    perf.finished_at,
                    name,
                    int(perf.retrieve_ms),
                    int(perf.first_token_ms),
                    int(perf.total_ms),
                )
                bot = panels[name]["bot"]
                bot.value = bot.value + [{"role": "assistant", "content": answer_text}]
                panels[name]["perf"].value = _format_perf(perf)

            return _flatten_output_values("", panels, retriever_names)

        click_outputs = _flatten_output_targets(question, panels, retriever_names)
        submit.click(on_submit, inputs=[question], outputs=click_outputs)

        def on_clear():
            for name in retriever_names:
                panels[name]["bot"].value = []
                panels[name]["chunks"].value = "_提交问题后展示_"
                panels[name]["perf"].value = _format_perf(None)
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
