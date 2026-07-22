"""RAG event model, emitters, and JSONL serialization."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from rag_learn.perf import StreamPerf
from rag_learn.retriever import Hit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruth:
    answer: str | None = None
    source_files: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAGEvent:
    trace_id: str
    timestamp: str
    collection: str
    question: str
    hits: tuple[Hit, ...]
    prompt: str
    answer: str
    perf: StreamPerf
    ground_truth: GroundTruth | None
    metadata: Mapping[str, Any]


class MetricsEmitter(Protocol):
    def emit(self, event: RAGEvent) -> None: ...


class ListEmitter:
    def __init__(self) -> None:
        self.events: list[RAGEvent] = []

    def emit(self, event: RAGEvent) -> None:
        self.events.append(event)


class NullEmitter:
    def emit(self, event: RAGEvent) -> None:
        return


class JSONLEmitter:
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def emit(self, event: RAGEvent) -> None:
        try:
            path = self._path_for(event.timestamp)
            record = _event_to_dict(event)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to emit RAGEvent to JSONL: %s", exc)

    def _path_for(self, timestamp: str) -> Path:
        date = timestamp[:10]
        return self.dir_path / f"rag_events_{date}.jsonl"


def _event_to_dict(event: RAGEvent) -> dict[str, Any]:
    return asdict(event)


def event_from_dict(data: dict[str, Any]) -> RAGEvent:
    gt_data = data.get("ground_truth")
    ground_truth = None
    if gt_data is not None:
        ground_truth = GroundTruth(
            answer=gt_data.get("answer"),
            source_files=tuple(gt_data.get("source_files", [])),
            chunk_ids=tuple(gt_data.get("chunk_ids", [])),
        )
    return RAGEvent(
        trace_id=data["trace_id"],
        timestamp=data["timestamp"],
        collection=data["collection"],
        question=data["question"],
        hits=tuple(Hit(**h) for h in data["hits"]),
        prompt=data["prompt"],
        answer=data["answer"],
        perf=StreamPerf(**data["perf"]),
        ground_truth=ground_truth,
        metadata=data.get("metadata", {}),
    )
