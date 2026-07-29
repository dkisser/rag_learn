"""Tests for the batch evaluation CLI."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_learn.eval import batch as batch_module
from rag_learn.eval.batch import main
from rag_learn.eval.tracing import GroundTruth, JSONLEmitter, RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit


def _make_event(
    trace_id: str,
    collection: str = "rag_doc",
    ground_truth: GroundTruth | None = None,
) -> RAGEvent:
    return RAGEvent(
        trace_id=trace_id,
        timestamp="2026-07-22T10:00:00+00:00",
        collection=collection,
        question="q",
        hits=(Hit(text="chunk", source_file="a.md", chunk_index=0, score=0.1),),
        prompt="p",
        answer="a",
        perf=StreamPerf(1.0, 2.0, 3.0, "10:00:01"),
        ground_truth=ground_truth,
        metadata={},
    )


def test_batch_dry_run_reports_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))
    emitter.emit(_make_event("t2", ground_truth=GroundTruth(source_files=("a.md",))))

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 2
    assert report["with_ground_truth"] == 1
    assert report["without_ground_truth"] == 1
    assert len(report["details"]) == 2


def test_batch_skips_corrupted_lines_and_counts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))

    files = list(tmp_path.glob("rag_events_*.jsonl"))
    with open(files[0], "a", encoding="utf-8") as f:
        f.write("this is not json\n")

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1
    assert report["skipped_corrupted_lines"] == 1


def test_batch_dedupes_by_trace_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("same-id"))
    emitter.emit(_make_event("same-id"))

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1


def test_batch_dry_run_does_not_require_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1


def test_batch_judge_failure_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))

    def failing_judge(system: str, user: str) -> str:
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: failing_judge)

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output)])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1
    details = report["details"][0]
    assert details["metrics"]["context_relevance"] is None
    assert details["metrics"]["faithfulness"] is None


def test_batch_skips_invalid_ground_truth_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1", ground_truth=GroundTruth(source_files=())))

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    details = report["details"][0]
    assert "retrieval_recall@5" not in details["metrics"]


def test_batch_computes_answer_f1_when_source_files_empty_but_answer_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty source_files means 'rely on model knowledge' — answer_f1 still applies."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(
        _make_event(
            "t1",
            ground_truth=GroundTruth(answer="ground truth", source_files=()),
        )
    )

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    details = report["details"][0]
    assert "retrieval_recall@5" not in details["metrics"]
    assert "retrieval_precision@5" not in details["metrics"]
    assert "retrieval_mrr" not in details["metrics"]
    assert "retrieval_ndcg@5" not in details["metrics"]
    assert "answer_f1" in details["metrics"]
    assert 0.0 <= details["metrics"]["answer_f1"] <= 1.0


def test_batch_details_include_question_answer_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1", ground_truth=GroundTruth(answer="gt", source_files=("a.md",))))

    output = tmp_path / "report.json"
    rc = main([str(tmp_path), "--output", str(output), "--dry-run"])
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    details = report["details"][0]
    assert details["question"] == "q"
    assert details["answer"] == "a"
    assert details["ground_truth"] == {
        "answer": "gt",
        "source_files": ["a.md"],
        "chunk_ids": [],
    }


def test_batch_judges_observe_concurrency_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent judge calls must not exceed the configured max_concurrency."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")

    # Three events × four unsupervised metrics each → 12 judge calls total.
    emitter = JSONLEmitter(tmp_path)
    for i in range(3):
        emitter.emit(_make_event(f"t{i}"))

    in_flight = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def tracking_judge(system: str, user: str) -> str:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            barrier.wait(timeout=2.0)
            time.sleep(0.05)
            return "4"
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: tracking_judge)

    output = tmp_path / "report.json"
    rc = main(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--max-concurrency",
            "2",
            "--max-retries",
            "1",
        ]
    )
    assert rc == 0
    assert peak <= 2, f"observed peak concurrency {peak} > 2"


def test_batch_retries_judge_on_rate_limit_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judge raising RateLimitError twice then returning a score must succeed."""
    from openai import RateLimitError

    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1", ground_truth=GroundTruth(answer="gt", source_files=("a.md",))))

    calls = {"n": 0}

    def flaky_judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429", response=MagicMock(), body={})
        return "4"

    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: flaky_judge)

    output = tmp_path / "report.json"
    rc = main(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--max-concurrency",
            "1",
            "--max-retries",
            "3",
        ]
    )
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    details = report["details"][0]
    assert details["metrics"]["context_relevance"] == 0.8
    assert details["metrics"]["answer_llm_correctness"] == 0.8


def test_batch_judge_exhausting_retries_records_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge that always 429s yields None for that metric, never raises."""
    from openai import RateLimitError

    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))

    def always_429(system: str, user: str) -> str:
        raise RateLimitError("429", response=MagicMock(), body={})

    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: always_429)

    output = tmp_path / "report.json"
    rc = main(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--max-concurrency",
            "1",
            "--max-retries",
            "2",
        ]
    )
    assert rc == 0

    with open(output, encoding="utf-8") as f:
        report = json.load(f)
    details = report["details"][0]
    assert details["metrics"]["context_relevance"] is None
    assert details["metrics"]["faithfulness"] is None
