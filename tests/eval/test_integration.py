"""End-to-end integration test for the batch evaluation CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rag_learn.eval import batch as batch_module
from rag_learn.eval.cli import main as cli_main


def test_cli_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lightweight end-to-end test using mocked answer_stream and judge."""
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
                retrieve_ms=1.0, first_token_ms=1.0, total_ms=1.0, finished_at="10:00:00"
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

        return {"rag_doc": (iter(["generated answer"]), [], get_perf)}

    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", fake_answer_stream)
    monkeypatch.setattr(batch_module, "_make_judge_fn", lambda _config, _model: lambda s, u: "4")

    csv_path = tmp_path / "qa.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "source_files",
                "chunk_ids",
                "collection",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question": "q1",
                "answer": "a1",
                "source_files": "a.md",
                "chunk_ids": "",
                "collection": "rag_doc",
            }
        )

    rc = cli_main(
        [
            "run",
            str(csv_path),
            "--output-events",
            str(tmp_path / "events.jsonl"),
            "--output-report",
            str(tmp_path / "report.json"),
        ]
    )
    assert rc == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["total_events"] == 1
    assert report["details"][0]["question"] == "q1"
