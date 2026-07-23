"""Tests for rag_learn.eval.sampler."""

from __future__ import annotations

import random
from pathlib import Path

from rag_learn.eval.sampler import sample_events, write_samples
from rag_learn.eval.tracing import JSONLEmitter, RAGEvent
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit


def _make_event(trace_id: str, collection: str = "rag_doc") -> RAGEvent:
    return RAGEvent(
        trace_id=trace_id,
        timestamp="2026-07-22T10:00:00+00:00",
        collection=collection,
        question=f"q-{trace_id}",
        hits=(Hit(text="chunk", source_file="a.md", chunk_index=0, score=0.1),),
        prompt="p",
        answer="a",
        perf=StreamPerf(1.0, 2.0, 3.0, "10:00:01"),
        ground_truth=None,
        metadata={},
    )


def test_sample_events_limits_per_collection(tmp_path: Path) -> None:
    random.seed(0)
    emitter = JSONLEmitter(tmp_path)
    for i in range(10):
        emitter.emit(_make_event(f"c1-{i}", "coll_a"))
    for i in range(3):
        emitter.emit(_make_event(f"c2-{i}", "coll_b"))

    rows = sample_events(tmp_path, samples_per_collection=5)

    by_collection = {"coll_a": [], "coll_b": []}
    for row in rows:
        by_collection[row["collection"]].append(row)
    assert len(by_collection["coll_a"]) == 5
    assert len(by_collection["coll_b"]) == 3


def test_sample_events_returns_csv_rows(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))

    rows = sample_events(tmp_path, samples_per_collection=5)
    assert len(rows) == 1
    assert rows[0]["question"] == "q-t1"
    assert rows[0]["collection"] == "rag_doc"
    assert rows[0]["answer"] == ""
    assert rows[0]["source_files"] == ""


def test_write_samples_creates_csv(tmp_path: Path) -> None:
    rows = [
        {"question": "q1", "answer": "", "source_files": "", "chunk_ids": "", "collection": "c1"}
    ]
    output = tmp_path / "samples.csv"
    write_samples(rows, output)

    text = output.read_text(encoding="utf-8")
    assert "question,answer,source_files,chunk_ids,collection" in text
    assert "q1,,,,c1" in text


def test_sample_events_skips_collections_with_no_events(tmp_path: Path) -> None:
    assert sample_events(tmp_path, samples_per_collection=5) == []
