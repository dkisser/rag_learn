"""RAG pipeline prompt construction, parallel retrieval, and streaming perf.

`answer_stream` supports two calling modes:

  1. Single-collection mode (new collection picker UI):
       answer_stream({slug: retriever}, llm, q)

  2. Multi-retriever parallel compare (legacy Chroma vs Milvus demo):
       answer_stream({"chroma": c, "milvus": m}, llm, q)

Both modes share the same internals: parallel retrieve → per-side
build_prompt → per-side streamed generation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rag_learn.config import CHUNK_DISPLAY_CHARS
from rag_learn.eval.tracing import RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit

if TYPE_CHECKING:
    from rag_learn.config import Config
    from rag_learn.eval.tracing import MetricsEmitter
    from rag_learn.llm import DeepSeekLLM
    from rag_learn.reranker.base import Reranker
    from rag_learn.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

# SYSTEM_PROMPT = (
#     "你是一个 RAG 助手。仅基于下方提供的「上下文」回答用户问题。"
#     "如果上下文不足以回答，直接说「未找到相关上下文」。"
#     "不要使用先验知识或编造内容。"
# )


# EMPTY_HITS_SYSTEM_PROMPT = (
#     "你是一个 RAG 助手。当前没有检索到任何相关上下文，"
#     "请直接告诉用户「未找到相关上下文」，不要使用先验知识或编造内容。"
# )
SYSTEM_PROMPT = (
    "你是一个 RAG 助手。逐条阅读「上下文」回答用户问题。"
    "如果遇到跟商品售卖相关的不确定信息，可以尝试回答，但必须强调一切以人工客服回答为准，建议用户转人工去咨询"
)

EMPTY_HITS_SYSTEM_PROMPT = "你是一个 RAG 助手。"


def build_prompt(chunks: list[Hit], question: str) -> tuple[str, str]:
    """Return ``(system_msg, user_msg)`` with display-safe chunk lengths."""
    if not chunks:
        return EMPTY_HITS_SYSTEM_PROMPT, f"问题：{question}\n回答："

    lines = ["上下文："]
    for i, hit in enumerate(chunks, start=1):
        text = hit.text
        if len(text) > CHUNK_DISPLAY_CHARS:
            text = text[:CHUNK_DISPLAY_CHARS]
        lines.append(f"[{i}] (来源: {hit.source_file}) {text}")
    user_msg = "\n".join(lines) + f"\n\n问题：{question}\n回答："
    return SYSTEM_PROMPT, user_msg


def _now_hms_ms() -> str:
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return time.strftime("%H:%M:%S", t) + f".{ms:03d}"


def _make_perf(
    retrieve_ms: float, started: float, first_token_at: float, end_at: float
) -> StreamPerf:
    return StreamPerf(
        retrieve_ms=retrieve_ms,
        first_token_ms=(first_token_at - started) * 1000.0,
        total_ms=(end_at - started) * 1000.0,
        finished_at=_now_hms_ms(),
    )


def _candidate_k(final_k: int, config: Config | None) -> int:
    """Compute how many candidates to fetch before optional reranking."""
    if config is None or not config.rerank_enabled:
        return final_k
    if config.rerank_k is not None:
        return max(final_k, config.rerank_k)
    return max(final_k, final_k * config.rerank_factor)


def _retrieve(
    retrievers: dict[str, BaseRetriever],
    question: str,
    final_k: int,
    candidate_k: int,
    reranker: Reranker | None = None,
) -> dict[str, list[Hit]]:
    """Run all retrievers in parallel (threads); return their Hits per side.

    Each retriever fetches ``candidate_k`` hits. If a reranker is provided, the
    candidates are re-scored and truncated back to ``final_k`` before being
    returned and fed into the prompt.
    """

    def _one(name: str) -> tuple[str, list[Hit]]:
        hits = retrievers[name].search(question, k=candidate_k)
        if reranker is not None:
            hits = reranker.rank(question, hits)
        return name, hits[:final_k]

    results: dict[str, list[Hit]] = {}
    with ThreadPoolExecutor(max_workers=max(2, len(retrievers))) as ex:
        futures = [ex.submit(_one, name) for name in retrievers]
        for fut in futures:
            name, hits = fut.result()
            results[name] = hits
    return results


def answer_stream(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
    emitter: MetricsEmitter | None = None,
    metadata: dict[str, Any] | None = None,
    reranker: Reranker | None = None,
    config: Config | None = None,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]]:
    """Parallel retrieve → build prompt per side → stream tokens per side.

    Returns ``{name: (token_iterator, hits, perf_fn)}``. The ``perf_fn``
    callable accepts the fully drained answer text, optionally emits a
    ``RAGEvent`` if an emitter was provided, and returns the populated
    :class:`StreamPerf`. It MUST be invoked AFTER the token iterator is
    fully drained by the caller.
    """
    event_metadata = metadata or {}
    retrieve_started = time.perf_counter()
    candidate_k = _candidate_k(k, config)
    hits_by_side = _retrieve(
        retrievers, question, final_k=k, candidate_k=candidate_k, reranker=reranker
    )
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0

    def _side(
        name: str,
        hits: list[Hit],
    ) -> tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]:
        sys_msg, user_msg = build_prompt(hits, question)
        prompt_text = f"{sys_msg}\n\n{user_msg}"
        started = time.perf_counter()
        out_perf_holder: list[StreamPerf] = []

        class _TimedIter:
            """Iterator that records first_token / end time as it streams."""

            def __init__(self) -> None:
                self._gen: Iterator[str] = iter(llm.stream(sys_msg, user_msg))
                self.first_token_at: float | None = None
                self.end_at: float | None = None
                self._done = False

            def __iter__(self) -> _TimedIter:
                return self

            def __next__(self) -> str:
                if self._done:
                    raise StopIteration
                try:
                    tok = next(self._gen)
                except StopIteration:
                    self._done = True
                    self.end_at = time.perf_counter()
                    first = self.first_token_at if self.first_token_at is not None else self.end_at
                    out_perf_holder.append(_make_perf(retrieve_ms, started, first, self.end_at))
                    raise
                if self.first_token_at is None:
                    self.first_token_at = time.perf_counter()
                return tok

        it = _TimedIter()

        def get_perf(answer: str) -> StreamPerf:
            perf = out_perf_holder[0]
            if emitter is not None:
                event: RAGEvent = RAGEvent(
                    trace_id=str(uuid.uuid4()),
                    timestamp=datetime.now(UTC).isoformat(),
                    collection=name,
                    question=question,
                    hits=tuple(hits),
                    prompt=prompt_text,
                    answer=answer,
                    perf=perf,
                    ground_truth=None,
                    metadata={**event_metadata, "k": k},
                )
                emitter.emit(event)
            return perf

        return it, hits, get_perf

    return {name: _side(name, hits_by_side[name]) for name in hits_by_side}
