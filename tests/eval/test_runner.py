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


def _find_events_file(output_dir: Path) -> Path | None:
    files = list(output_dir.glob("rag_events_*.jsonl"))
    return files[0] if files else None


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

    events_file = _find_events_file(output_events.parent)
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
    events_file = _find_events_file(output_events.parent)
    assert events_file is None or events_file.read_text(encoding="utf-8").strip() == ""
