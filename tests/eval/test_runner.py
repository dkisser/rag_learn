"""Tests for rag_learn.eval.runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rag_learn.eval import batch as batch_module
from rag_learn.eval.runner import run_qa_csv


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "answer", "source_files", "chunk_ids", "collection"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _find_events_file(output_path: Path) -> Path | None:
    return output_path if output_path.is_file() and output_path.stat().st_size > 0 else None


def test_run_qa_csv_produces_events_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")

    class FakeRetriever:
        pass

    class FakeCollection:
        retriever = FakeRetriever()

    class FakeCatalog:
        def get(self, name: str):
            return FakeCollection()

    monkeypatch.setattr("rag_learn.eval.runner._load_catalog", lambda: FakeCatalog())

    def fake_answer_stream(retrievers, llm, question, k=5, emitter=None, metadata=None):
        from rag_learn.eval.tracing import RAGEvent
        from rag_learn.perf import StreamPerf
        from rag_learn.retriever import Hit

        def get_perf(answer: str):
            perf = StreamPerf(
                retrieve_ms=1.0, first_token_ms=2.0, total_ms=3.0, finished_at="10:00:00"
            )
            if emitter is not None:
                emitter.emit(
                    RAGEvent(
                        trace_id="tid",
                        timestamp="2026-07-22T10:00:00+00:00",
                        collection="rag_doc",
                        question=question,
                        hits=(Hit(text="h", source_file="a.md", chunk_index=0, score=0.1),),
                        prompt="p",
                        answer=answer,
                        perf=perf,
                        ground_truth=None,
                        metadata=metadata or {},
                    )
                )
            return perf

        return {"rag_doc": (iter(["answer text"]), [], get_perf)}

    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", fake_answer_stream)
    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: lambda s, u: "3")

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "What is RAG?",
                "answer": " retrieval-augmented generation ",
                "source_files": "a.md",
                "chunk_ids": "",
                "collection": "rag_doc",
            }
        ],
    )
    output_events = tmp_path / "batch_events.jsonl"
    report_path = tmp_path / "report.json"

    rc = run_qa_csv(csv_path, None, output_events, report_path, judge_model="dummy")
    assert rc == 0

    events_file = _find_events_file(output_events)
    assert events_file is not None
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1
    details = report["details"][0]
    assert details["question"] == "What is RAG?"
    assert details["ground_truth"]["answer"] == "retrieval-augmented generation"
    assert details["ground_truth"]["source_files"] == ["a.md"]


def test_run_qa_csv_skips_invalid_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", lambda *a, **k: {})

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "rag_doc",
            },
            {
                "question": "q",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "",
            },
        ],
    )
    output_events = tmp_path / "batch_events.jsonl"
    report_path = tmp_path / "report.json"

    rc = run_qa_csv(csv_path, None, output_events, report_path)
    assert rc == 0
    events_file = _find_events_file(output_events)
    assert events_file is None or events_file.read_text(encoding="utf-8").strip() == ""


def _seed_events_file(events_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a JSONL file at the exact ``events_path`` (creates parents)."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return events_path


def test_run_qa_csv_skips_already_emitted_questions_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CSV re-run with resume=True skips (collection, question) already on disk."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")

    seed_event = {
        "trace_id": "seed-1",
        "timestamp": "2026-07-24T10:00:00+00:00",
        "collection": "rag_doc",
        "question": "Already done",
        "hits": [],
        "prompt": "",
        "answer": "prior answer",
        "perf": {
            "retrieve_ms": 1.0,
            "first_token_ms": 2.0,
            "total_ms": 3.0,
            "finished_at": "10:00:01",
        },
        "ground_truth": None,
        "metadata": {},
    }
    events_dir = tmp_path / "events"
    _seed_events_file(events_dir / "batch_events.jsonl", [seed_event])

    class FakeRetriever:
        pass

    class FakeCollection:
        retriever = FakeRetriever()

    class FakeCatalog:
        def get(self, name: str):
            return FakeCollection()

    monkeypatch.setattr("rag_learn.eval.runner._load_catalog", lambda: FakeCatalog())

    processed: list[str] = []

    def fake_answer_stream(retrievers, llm, question, k=5, emitter=None, metadata=None):
        processed.append(question)
        from rag_learn.eval.tracing import RAGEvent
        from rag_learn.perf import StreamPerf
        from rag_learn.retriever import Hit

        def get_perf(answer: str) -> StreamPerf:
            perf = StreamPerf(
                retrieve_ms=1.0, first_token_ms=2.0, total_ms=3.0, finished_at="10:00:00"
            )
            if emitter is not None:
                emitter.emit(
                    RAGEvent(
                        trace_id="new-tid",
                        timestamp="2026-07-24T10:00:00+00:00",
                        collection="rag_doc",
                        question=question,
                        hits=(Hit(text="h", source_file="a.md", chunk_index=0, score=0.1),),
                        prompt="p",
                        answer=answer,
                        perf=perf,
                        ground_truth=None,
                        metadata=metadata or {},
                    )
                )
            return perf

        return {"rag_doc": (iter(["answer text"]), [], get_perf)}

    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", fake_answer_stream)
    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: lambda s, u: "3")

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "Already done",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "rag_doc",
            },
            {
                "question": "New question",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "rag_doc",
            },
        ],
    )
    rc = run_qa_csv(
        csv_path,
        None,
        events_dir / "batch_events.jsonl",
        tmp_path / "report.json",
        resume=True,
    )
    assert rc == 0
    assert processed == ["New question"], (
        f"resume=True should skip the seed question; processed={processed}"
    )


def test_run_qa_csv_reprocesses_when_resume_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With resume=False, even already-emitted questions are processed again."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    seed_event = {
        "trace_id": "seed-1",
        "timestamp": "2026-07-24T10:00:00+00:00",
        "collection": "rag_doc",
        "question": "Q",
        "hits": [],
        "prompt": "",
        "answer": "prior",
        "perf": {
            "retrieve_ms": 1.0,
            "first_token_ms": 2.0,
            "total_ms": 3.0,
            "finished_at": "10:00:01",
        },
        "ground_truth": None,
        "metadata": {},
    }
    events_dir = tmp_path / "events"
    _seed_events_file(events_dir / "batch_events.jsonl", [seed_event])

    class FakeRetriever:
        pass

    class FakeCollection:
        retriever = FakeRetriever()

    class FakeCatalog:
        def get(self, name: str):
            return FakeCollection()

    monkeypatch.setattr("rag_learn.eval.runner._load_catalog", lambda: FakeCatalog())

    processed: list[str] = []

    def fake_answer_stream(retrievers, llm, question, k=5, emitter=None, metadata=None):
        processed.append(question)
        from rag_learn.eval.tracing import RAGEvent
        from rag_learn.perf import StreamPerf
        from rag_learn.retriever import Hit

        def get_perf(answer: str) -> StreamPerf:
            perf = StreamPerf(
                retrieve_ms=1.0, first_token_ms=2.0, total_ms=3.0, finished_at="10:00:00"
            )
            if emitter is not None:
                emitter.emit(
                    RAGEvent(
                        trace_id="new",
                        timestamp="2026-07-24T10:00:00+00:00",
                        collection="rag_doc",
                        question=question,
                        hits=(Hit(text="h", source_file="a.md", chunk_index=0, score=0.1),),
                        prompt="p",
                        answer=answer,
                        perf=perf,
                        ground_truth=None,
                        metadata=metadata or {},
                    )
                )
            return perf

        return {"rag_doc": (iter(["x"]), [], get_perf)}

    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", fake_answer_stream)
    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: lambda s, u: "3")

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "Q",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "rag_doc",
            }
        ],
    )
    rc = run_qa_csv(
        csv_path,
        None,
        events_dir / "batch_events.jsonl",
        tmp_path / "report.json",
        resume=False,
    )
    assert rc == 0
    assert processed == ["Q"]


def test_run_qa_csv_invokes_rate_limiter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_qa_csv wraps _process_row in limiter.call()."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")

    class FakeRetriever:
        pass

    class FakeCollection:
        retriever = FakeRetriever()

    class FakeCatalog:
        def get(self, name: str):
            return FakeCollection()

    monkeypatch.setattr("rag_learn.eval.runner._load_catalog", lambda: FakeCatalog())
    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", lambda *a, **k: {})

    captured: dict[str, object] = {}

    class FakeLimiter:
        def __init__(self, max_concurrency, rate_per_minute, max_retries):
            captured["init_args"] = (max_concurrency, rate_per_minute, max_retries)

        def call(self, fn, *args, **kwargs):
            captured.setdefault("calls", []).append(getattr(fn, "__name__", repr(fn)))
            return fn(*args, **kwargs)

    monkeypatch.setattr("rag_learn.eval.runner.RateLimiter", FakeLimiter)
    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: lambda s, u: "3")

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "q",
                "answer": "",
                "source_files": "",
                "chunk_ids": "",
                "collection": "rag_doc",
            }
        ],
    )
    rc = run_qa_csv(
        csv_path,
        None,
        tmp_path / "batch_events.jsonl",
        tmp_path / "report.json",
        max_concurrency=5,
        rate_per_minute=42.0,
        max_retries=4,
    )
    assert rc == 0
    assert captured["init_args"] == (5, 42.0, 4)
    assert "_process_row" in captured["calls"]
