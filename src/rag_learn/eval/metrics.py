"""Retrieval and generation evaluation metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from rag_learn.eval.tracing import RAGEvent
from rag_learn.retriever import Hit


def retrieval_recall_at_k(hits: Sequence[Hit], source_files: tuple[str, ...], k: int) -> float:
    if not source_files:
        return 0.0
    retrieved = {h.source_file for h in hits[:k]}
    relevant = set(source_files)
    return len(retrieved & relevant) / len(relevant)


def retrieval_precision_at_k(hits: Sequence[Hit], source_files: tuple[str, ...], k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    relevant = set(source_files)
    return sum(1 for h in top if h.source_file in relevant) / len(top)


def retrieval_mrr(hits: Sequence[Hit], source_files: tuple[str, ...]) -> float:
    relevant = set(source_files)
    for rank, h in enumerate(hits, start=1):
        if h.source_file in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_ndcg_at_k(hits: Sequence[Hit], source_files: tuple[str, ...], k: int) -> float:
    relevant = set(source_files)
    dcg = 0.0
    for i, h in enumerate(hits[:k], start=1):
        rel = 1.0 if h.source_file in relevant else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_rels = [1.0] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def answer_f1(answer: str, ground_truth: str) -> float:
    pred = _tokens(answer)
    true = _tokens(ground_truth)
    if not pred or not true:
        return 0.0
    common = pred & true
    precision = len(common) / len(pred)
    recall = len(common) / len(true)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _extract_score(text: str, scale: tuple[int, int] = (1, 5)) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    raw = float(match.group(1))
    lo, hi = scale
    clamped = max(lo, min(hi, raw))
    return clamped / hi


def _context_text(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))


def context_relevance(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    system = (
        "You are an evaluator. Rate how relevant the retrieved context is to the question. "
        "Output only a number from 1 to 5, where 1 = completely irrelevant, 5 = highly relevant."
    )
    user = (
        f"Question: {event.question}\n\n"
        f"Context:\n{_context_text(event.hits)}\n\n"
        "Relevance score (1-5):"
    )
    return _extract_score(judge_fn(system, user))


def faithfulness(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    system = (
        "You are an evaluator. Rate whether the answer is fully supported by the context "
        "and contains no hallucinated information. Output only a number from 1 to 5, "
        "where 1 = not faithful, 5 = fully faithful."
    )
    user = (
        f"Context:\n{_context_text(event.hits)}\n\n"
        f"Question: {event.question}\n\n"
        f"Answer: {event.answer}\n\n"
        "Faithfulness score (1-5):"
    )
    return _extract_score(judge_fn(system, user))


def answer_relevance(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    system = (
        "You are an evaluator. Rate how well the answer addresses the question. "
        "Output only a number from 1 to 5, where 1 = does not address, 5 = fully addresses."
    )
    user = f"Question: {event.question}\n\nAnswer: {event.answer}\n\nAnswer relevance score (1-5):"
    return _extract_score(judge_fn(system, user))


def overall_usefulness(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    system = (
        "You are an evaluator. Rate the overall usefulness of the answer. "
        "Output only a number from 1 to 5, where 1 = not useful, 5 = very useful."
    )
    user = (
        f"Question: {event.question}\n\nAnswer: {event.answer}\n\nOverall usefulness score (1-5):"
    )
    return _extract_score(judge_fn(system, user))


def answer_llm_correctness(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    if event.ground_truth is None or event.ground_truth.answer is None:
        return None
    system = (
        "You are an evaluator. Compare the generated answer to the ground truth answer. "
        "Output only a number from 1 to 5, where 1 = incorrect, 5 = correct."
    )
    user = (
        f"Question: {event.question}\n\n"
        f"Ground truth: {event.ground_truth.answer}\n\n"
        f"Generated answer: {event.answer}\n\n"
        "Correctness score (1-5):"
    )
    return _extract_score(judge_fn(system, user))
