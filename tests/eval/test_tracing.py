import json
from pathlib import Path

from rag_learn.eval import (
    GroundTruth,
    JSONLEmitter,
    ListEmitter,
    MetricsEmitter,
    NullEmitter,
    RAGEvent,
    event_from_dict,
)
from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit


def _make_event(
    trace_id: str = "t1",
    collection: str = "rag_doc",
    ground_truth: GroundTruth | None = None,
) -> RAGEvent:
    return RAGEvent(
        trace_id=trace_id,
        timestamp="2026-07-22T10:00:00+00:00",
        collection=collection,
        question="什么是 RAG？",
        hits=(Hit(text="chunk", source_file="a.md", chunk_index=0, score=0.1),),
        prompt="system\n\nuser",
        answer="RAG 是检索增强生成。",
        perf=StreamPerf(
            retrieve_ms=1.0,
            first_token_ms=2.0,
            total_ms=3.0,
            finished_at="10:00:01",
        ),
        ground_truth=ground_truth,
        metadata={"k": 5, "llm_model": "deepseek-v4-flash"},
    )


def test_event_round_trip_via_jsonl(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    event = _make_event(
        ground_truth=GroundTruth(
            answer="RAG 是检索增强生成。",
            source_files=("a.md",),
            chunk_ids=("a.md#0",),
        )
    )
    emitter.emit(event)

    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        data = json.loads(f.readline())

    restored = event_from_dict(data)
    assert restored.trace_id == event.trace_id
    assert restored.collection == event.collection
    assert restored.hits == event.hits
    assert restored.ground_truth == event.ground_truth
    assert restored.metadata == event.metadata


def test_event_round_trip_without_ground_truth(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    event = _make_event()
    emitter.emit(event)

    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        data = json.loads(f.readline())

    restored = event_from_dict(data)
    assert restored.ground_truth is None
    assert restored.trace_id == event.trace_id


def test_jsonl_emitter_appends_to_same_daily_file(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event(trace_id="t1"))
    emitter.emit(_make_event(trace_id="t2"))

    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert event_from_dict(json.loads(lines[0])).trace_id == "t1"
    assert event_from_dict(json.loads(lines[1])).trace_id == "t2"


def test_jsonl_emitter_creates_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "events"
    emitter = JSONLEmitter(nested)
    emitter.emit(_make_event())
    assert nested.exists()


def test_jsonl_emitter_is_safe_on_write_failure(tmp_path: Path, monkeypatch) -> None:
    emitter = JSONLEmitter(tmp_path)

    def _raise(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(emitter, "_path_for", _raise)
    emitter.emit(_make_event())  # should not raise


def test_jsonl_emitter_with_file_name_writes_literal_path(tmp_path: Path) -> None:
    """With file_name set, JSONLEmitter writes to that exact file regardless of timestamp."""
    out_file = tmp_path / "shanzhongshi_events.jsonl"
    emitter = JSONLEmitter(tmp_path, file_name="shanzhongshi_events.jsonl")

    emitter.emit(_make_event("t1"))
    emitter.emit(_make_event("t2"))

    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    # No daily-rotation file should appear when file_name is fixed.
    assert list(tmp_path.glob("rag_events_*.jsonl")) == []


def test_jsonl_emitter_file_name_creates_parent_dirs(tmp_path: Path) -> None:
    """file_name may include subdirectories; parent dirs are created."""
    out_file = tmp_path / "nested" / "deep" / "events.jsonl"
    emitter = JSONLEmitter(tmp_path, file_name="nested/deep/events.jsonl")

    emitter.emit(_make_event("t1"))

    assert out_file.is_file()


def test_jsonl_emitter_default_daily_rotation_still_works(tmp_path: Path) -> None:
    """Backwards compat: no file_name → keeps the rag_events_<date>.jsonl behavior."""
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event("t1"))
    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 1


def test_list_emitter_collects_events() -> None:
    emitter = ListEmitter()
    event = _make_event()
    emitter.emit(event)
    assert emitter.events == [event]


def test_null_emitter_does_nothing() -> None:
    emitter = NullEmitter()
    emitter.emit(_make_event())


def test_metrics_emitter_protocol_acceptance() -> None:
    def _use(emitter: MetricsEmitter, event: RAGEvent) -> None:
        emitter.emit(event)

    _use(ListEmitter(), _make_event())
    _use(NullEmitter(), _make_event())


def test_ground_truth_defaults() -> None:
    gt = GroundTruth()
    assert gt.answer is None
    assert gt.source_files == ()
    assert gt.chunk_ids == ()
