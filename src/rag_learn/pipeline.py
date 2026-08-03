"""RAG pipeline prompt construction, parallel retrieval, and streaming perf.

`answer_stream` supports two calling modes:

  1. Single-collection mode (new collection picker UI):
       answer_stream({slug: retriever}, llm, q)

  2. Multi-retriever parallel compare (legacy Chroma vs Milvus demo):
       answer_stream({"chroma": c, "milvus": m}, llm, q)

Both modes share the same internals: parallel retrieve → per-side
build_prompt → per-side streamed generation.

When the optional ``Catalog`` and ``config.intent_enabled`` are provided,
the pipeline first classifies the user's intent. If the intent is
``"all"`` AND ``config.decompose_enabled`` is true, the question is
split into sub-queries (via ``routing.decompose_query``) and each
sub-query is retrieved independently; the merged hits are
round-robin dedup'd and fed into the catalog-recall prompt. The
reranker is intentionally skipped in this branch — diversity beats
relevance for "compare all" / "recommend" queries.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from rag_learn.config import CHUNK_DISPLAY_CHARS
from rag_learn.eval.tracing import RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit
from rag_learn.routing import INTENT_LABELS, RoutingInfo

if TYPE_CHECKING:
    from rag_learn.collections import Catalog
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
    "你是一个电商客服。你的工作是：逐条阅读「上下文」，使用和蔼语气来回答用户问题。"
    "如果上下文中已有相关内容，根据上下文去回答。"
    "而上下文中没有且和商品售卖相关的问题，可以尝试根据经验回答，但必须强调一切以人工客服回答为准，建议用户转人工去咨询。"
    "如果遇到和咖啡冲煮技巧、咖啡品类挑选等偏主观的经验性问题。你需要在你的回答中补充‘通常来说’，好让用户明白你给的是建议而不是绝对的答案。"
)

CATALOG_RECALL_SYSTEM_PROMPT = (
    "你正在回答一个要求覆盖整个目录的问题。"
    "下方片段可能横跨多个不同条目(不同豆子、不同冲煮法、不同政策)。"
    "请基于所有相关片段给出结构化答案:同类项归并、显式列出每一项、"
    "宁可详尽也不简略;若某条与问题无关,直接忽略;不得编造目录中不存在的条目。"
)

EMPTY_HITS_SYSTEM_PROMPT = "你是一个 RAG 助手。"

PromptMode = Literal["normal", "catalog_recall"]


def build_prompt(
    chunks: list[Hit],
    question: str,
    *,
    mode: PromptMode = "normal",
) -> tuple[str, str]:
    """Return ``(system_msg, user_msg)`` with display-safe chunk lengths."""
    system = CATALOG_RECALL_SYSTEM_PROMPT if mode == "catalog_recall" else SYSTEM_PROMPT
    if not chunks:
        return EMPTY_HITS_SYSTEM_PROMPT, f"问题：{question}\n回答："

    lines = ["上下文："]
    for i, hit in enumerate(chunks, start=1):
        text = hit.text
        if len(text) > CHUNK_DISPLAY_CHARS:
            text = text[:CHUNK_DISPLAY_CHARS]
        lines.append(f"[{i}] (来源: {hit.source_file}) {text}")
    user_msg = "\n".join(lines) + f"\n\n问题：{question}\n回答："
    return system, user_msg


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


def _build_catalog_summary(catalog: Catalog, only: Iterable[str] = ()) -> str:
    """Render a one-line-per-collection string for the decomposer prompt.

    ``only`` scopes the summary to the collections actually being searched
    (``retrievers`` keys). Describing collections the fan-out will never
    touch makes the decomposer emit sub-queries aimed at the wrong corpus.
    Keys that match no collection — the legacy ``chroma``/``milvus``
    compare mode — fall back to the whole catalog rather than an empty
    summary.
    """
    wanted = set(only)
    collections = list(catalog.iter_collections())
    scoped = [c for c in collections if c.name in wanted] or collections
    lines = [f"- {c.display_name}: {c.description}" for c in scoped]
    return "\n".join(lines) if lines else "(empty catalog)"


def _flat_retrieve(retrievers: dict[str, BaseRetriever], sub_query: str, k: int) -> list[Hit]:
    """Fan-out one sub-query across every retriever; no reranker, take top-k each."""
    out: list[Hit] = []
    for r in retrievers.values():
        out.extend(r.search(sub_query, k=k))
    return out


def _merge_dedup(per_sub: list[list[Hit]], final_k: int) -> list[Hit]:
    """Round-robin merge across sub-query hit lists, dedup on (file, chunk_index)."""
    seen: set[tuple[str, int]] = set()
    merged: list[Hit] = []
    max_len = max((len(h) for h in per_sub), default=0)
    for i in range(max_len):
        for hits in per_sub:
            if i < len(hits):
                h = hits[i]
                key = (h.source_file, h.chunk_index)
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
                    if len(merged) >= final_k:
                        return merged
    return merged


def _answer_catalog_recall(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    intent: str,
    sub_queries: list[str],
    *,
    sub_k: int,
    final_k: int,
    emitter: MetricsEmitter | None,
    metadata: dict[str, Any],
    retrieve_ms: float,
    result_key: str,
    routing_sink: Callable[[RoutingInfo], None] | None = None,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]]:
    """Fan-out retrieve → round-robin merge → stream via catalog system prompt.

    ``sub_k`` bounds how many candidates EACH sub-query pulls from EACH
    retriever; ``final_k`` bounds how many survive the merge and reach the
    prompt. Keeping them separate lets the fan-out stay broad without
    inflating the prompt (they used to be the same config value).
    """
    sub_queries = sub_queries or [question]
    with ThreadPoolExecutor(max_workers=min(8, len(sub_queries))) as ex:
        futures = [ex.submit(_flat_retrieve, retrievers, sq, sub_k) for sq in sub_queries]
        # Collect in SUBMIT order, not completion order: the round-robin
        # merge below is order-sensitive, so `as_completed` would make the
        # final chunk set depend on thread scheduling (non-reproducible).
        per_sub: list[list[Hit]] = [fut.result() for fut in futures]

    merged = _merge_dedup(per_sub, final_k=final_k)
    sys_msg, user_msg = build_prompt(merged, question, mode="catalog_recall")

    info = RoutingInfo(
        intent="all" if intent == "all" else "specific",
        sub_queries=tuple(sub_queries),
        target_collections=tuple(retrievers.keys()),
        merged_k=len(merged),
    )
    metadata.update(info.as_metadata())
    if routing_sink is not None:
        routing_sink(info)
    logger.info(
        "routing: intent=%s sub_queries=%d merged_k=%d",
        intent,
        len(sub_queries),
        len(merged),
    )

    started = time.perf_counter()
    out_perf_holder: list[StreamPerf] = []

    class _TimedIter:
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
    prompt_text = f"{sys_msg}\n\n{user_msg}"

    def get_perf(answer: str) -> StreamPerf:
        perf = out_perf_holder[0]
        if emitter is not None:
            event: RAGEvent = RAGEvent(
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(UTC).isoformat(),
                collection=result_key,
                question=question,
                hits=tuple(merged),
                prompt=prompt_text,
                answer=answer,
                perf=perf,
                ground_truth=None,
                metadata={**metadata, "k": final_k},
            )
            emitter.emit(event)
        return perf

    return {result_key: (it, merged, get_perf)}


def answer_stream(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
    emitter: MetricsEmitter | None = None,
    metadata: dict[str, Any] | None = None,
    reranker: Reranker | None = None,
    config: Config | None = None,
    catalog: Catalog | None = None,
    routing_sink: Callable[[RoutingInfo], None] | None = None,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]]:
    """Parallel retrieve → build prompt per side → stream tokens per side.

    Returns ``{name: (token_iterator, hits, perf_fn)}``. The ``perf_fn``
    callable accepts the fully drained answer text, optionally emits a
    ``RAGEvent`` if an emitter was provided, and returns the populated
    :class:`StreamPerf`. It MUST be invoked AFTER the token iterator is
    fully drained by the caller.

    When ``config.intent_enabled`` is True and ``catalog`` is provided,
    the user's question is first classified via ``routing.classify_intent``.
    If the intent is ``"all"`` AND ``config.decompose_enabled`` is True,
    the question is split into sub-queries via ``routing.decompose_query``,
    each sub-query is retrieved independently, and the merged hits are
    fed into the catalog-recall system prompt. The reranker is NOT
    invoked in this branch.

    ``metadata`` is READ-ONLY input: it is shallow-copied before any
    routing field is written, because ``eval.runner`` shares one dict
    across concurrently-processed rows. Callers that need to display the
    routing decisions (the UI does) pass ``routing_sink`` — a callback
    invoked at most once per call with an immutable :class:`RoutingInfo`.
    It is NOT invoked when the intent classifier is disabled.
    """
    # IMPORTANT: shallow-copy the caller's dict so routing metadata writes
    # (intent / sub_queries / target_collections / merged_k) never mutate
    # the caller's object. The original dict is treated as read-only input.
    event_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
    cfg = config
    intent: INTENT_LABELS | None = None

    # Catalog-coverage branch: classify + (optional) decompose + fan-out.
    if cfg is not None and cfg.intent_enabled and catalog is not None:
        from rag_learn.routing import classify_intent, decompose_query

        intent = classify_intent(llm, question, timeout_s=cfg.intent_timeout_s)
        if intent == "all" and cfg.decompose_enabled:
            sub_queries = decompose_query(
                llm,
                question,
                # Scope the summary to what will actually be searched.
                _build_catalog_summary(catalog, only=retrievers.keys()),
                max_sub_queries=cfg.decompose_max,
                timeout_s=cfg.decompose_timeout_s,
            )
            retrieve_started = time.perf_counter()
            # Use the original collection slug as the result key so callers
            # can still index `out[slug]` regardless of routing branch.
            result_key = next(iter(retrievers.keys())) if retrievers else "_catalog"
            result = _answer_catalog_recall(
                retrievers,
                llm,
                question,
                intent,
                sub_queries,
                sub_k=cfg.catalog_sub_k,
                final_k=cfg.catalog_recall_k,
                emitter=emitter,
                metadata=event_metadata,
                retrieve_ms=(time.perf_counter() - retrieve_started) * 1000.0,
                result_key=result_key,
                routing_sink=routing_sink,
            )
            return result

    # Original single-query path.
    retrieve_started = time.perf_counter()
    candidate_k = _candidate_k(k, cfg)
    hits_by_side = _retrieve(
        retrievers, question, final_k=k, candidate_k=candidate_k, reranker=reranker
    )
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0

    # Report the classification even when it did NOT trigger the catalog
    # branch, so the caller can tell "classified as specific" apart from
    # "classifier never ran".
    if intent is not None:
        specific_info = RoutingInfo(
            intent=intent,
            sub_queries=(),
            target_collections=tuple(retrievers.keys()),
            merged_k=sum(len(h) for h in hits_by_side.values()),
        )
        event_metadata.update(specific_info.as_metadata())
        if routing_sink is not None:
            routing_sink(specific_info)

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
