"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from rag_learn.eval.metrics import (
    answer_f1,
    answer_llm_correctness,
    answer_relevance,
    context_relevance,
    faithfulness,
    overall_usefulness,
    retrieval_mrr,
    retrieval_ndcg_at_k,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from rag_learn.eval.tracing import GroundTruth, RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit


def _hit(source_file: str) -> Hit:
    return Hit(text="x", source_file=source_file, chunk_index=0, score=0.0)


def test_retrieval_recall_at_k() -> None:
    hits = [_hit("a.md"), _hit("b.md"), _hit("c.md")]
    assert retrieval_recall_at_k(hits, ("a.md", "d.md"), 3) == 0.5
    assert retrieval_recall_at_k(hits, ("a.md", "b.md"), 2) == 1.0
    assert retrieval_recall_at_k(hits, (), 3) == 0.0


def test_retrieval_precision_at_k() -> None:
    hits = [_hit("a.md"), _hit("b.md"), _hit("c.md")]
    assert retrieval_precision_at_k(hits, ("a.md", "c.md"), 3) == 2 / 3
    assert retrieval_precision_at_k(hits, ("a.md",), 1) == 1.0


def test_retrieval_precision_at_k_empty_hits() -> None:
    assert retrieval_precision_at_k([], ("a.md",), 3) == 0.0


def test_retrieval_mrr() -> None:
    hits = [_hit("x.md"), _hit("a.md"), _hit("b.md")]
    assert retrieval_mrr(hits, ("a.md",)) == 0.5
    assert retrieval_mrr([_hit("a.md")], ("a.md",)) == 1.0
    assert retrieval_mrr([_hit("a.md")], ("c.md",)) == 0.0


def test_retrieval_ndcg_at_k() -> None:
    hits = [_hit("a.md"), _hit("b.md"), _hit("c.md")]
    relevant = ("a.md", "c.md")
    # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309... = 1.6309...
    assert retrieval_ndcg_at_k(hits, relevant, 3) == pytest.approx(1.5 / (1 + 1 / 1.5849625))


def test_retrieval_ndcg_at_k_empty_relevant() -> None:
    assert retrieval_ndcg_at_k([_hit("a.md")], (), 3) == 0.0


def _make_event(
    question: str = "q",
    answer: str = "a",
    hits: tuple[Hit, ...] = (),
    ground_truth: GroundTruth | None = None,
) -> RAGEvent:
    return RAGEvent(
        trace_id="t1",
        timestamp="2026-07-22T10:00:00+00:00",
        collection="c",
        question=question,
        hits=hits,
        prompt="p",
        answer=answer,
        perf=StreamPerf(1.0, 2.0, 3.0, "10:00:01"),
        ground_truth=ground_truth,
        metadata={},
    )


def test_answer_f1() -> None:
    assert answer_f1("the quick brown fox", "the quick brown fox") == 1.0
    assert answer_f1("quick brown", "the quick brown fox") == pytest.approx(
        2 * (2 / 2) * (2 / 4) / ((2 / 2) + (2 / 4))
    )


def test_context_relevance_extracts_score() -> None:
    event = _make_event(question="what is RAG?", hits=(_hit("a.md"),))

    def judge(system: str, user: str) -> str:
        return "Score: 4"

    assert context_relevance(event, judge) == 0.8


def test_faithfulness_extracts_score() -> None:
    event = _make_event(question="q", answer="a", hits=(_hit("a.md"),))

    def judge(system: str, user: str) -> str:
        return "5"

    assert faithfulness(event, judge) == 1.0


def test_answer_relevance_returns_none_when_no_score() -> None:
    event = _make_event(question="q", answer="a")

    def judge(system: str, user: str) -> str:
        return "no number here"

    assert answer_relevance(event, judge) is None


def test_answer_llm_correctness_with_ground_truth() -> None:
    event = _make_event(
        question="q",
        answer="a",
        ground_truth=GroundTruth(answer="ground truth answer"),
    )

    def judge(system: str, user: str) -> str:
        return "Score: 3"

    assert answer_llm_correctness(event, judge) == 0.6


def test_overall_usefulness_no_ground_truth_needed() -> None:
    event = _make_event(question="q", answer="a")

    def judge(system: str, user: str) -> str:
        return "4"

    assert overall_usefulness(event, judge) == 0.8


def test_answer_f1_empty_input_returns_zero() -> None:
    assert answer_f1("", "ground truth") == 0.0
    assert answer_f1("answer", "") == 0.0


def test_answer_llm_correctness_returns_none_without_ground_truth() -> None:
    event = _make_event(question="q", answer="a")

    def judge(system: str, user: str) -> str:
        return "5"

    assert answer_llm_correctness(event, judge) is None
