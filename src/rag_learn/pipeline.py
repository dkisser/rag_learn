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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_learn.config import CHUNK_DISPLAY_CHARS
from rag_learn.retriever import Hit

if TYPE_CHECKING:
    from rag_learn.llm import DeepSeekLLM
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
SYSTEM_PROMPT = "你是一个 RAG 助手。尽量基于下方提供的「上下文」回答用户问题。"

EMPTY_HITS_SYSTEM_PROMPT = "你是一个 RAG 助手。"


@dataclass(frozen=True)
class StreamPerf:
    retrieve_ms: float
    first_token_ms: float
    total_ms: float
    finished_at: str  # HH:MM:SS.mmm


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


def _retrieve(retrievers: dict[str, BaseRetriever], question: str, k: int) -> dict[str, list[Hit]]:
    """Run all retrievers in parallel (threads); return their Hits per side."""

    def _one(name: str) -> tuple[str, list[Hit]]:
        return name, retrievers[name].search(question, k=k)

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
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[], StreamPerf]]]:
    """Parallel retrieve → build prompt per side → stream tokens per side.

    Returns ``{name: (token_iterator, hits, perf_fn)}``. The ``perf_fn``
    callable returns the populated :class:`StreamPerf` and MUST be invoked
    AFTER the token iterator is fully drained by the caller.
    """
    retrieve_started = time.perf_counter()
    hits_by_side = _retrieve(retrievers, question, k)
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0

    def _side(
        hits: list[Hit],
    ) -> tuple[Iterator[str], list[Hit], Callable[[], StreamPerf]]:
        sys_msg, user_msg = build_prompt(hits, question)
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

        def get_perf() -> StreamPerf:
            return out_perf_holder[0]

        return it, hits, get_perf

    return {name: _side(hits_by_side[name]) for name in hits_by_side}
