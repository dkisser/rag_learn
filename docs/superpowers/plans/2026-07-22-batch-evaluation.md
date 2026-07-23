# Batch RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI with three subcommands (`sample`, `run`, `evaluate`) that lets users sample online RAG events into a CSV, label them with ground truth, run them through the RAG pipeline, and compute supervised + unsupervised metrics.

**Architecture:** A thin `cli.py` dispatches to three focused modules: `sampler.py` reads existing JSONL events and writes a CSV; `runner.py` reads a CSV, calls `answer_stream`, emits `RAGEvent`s with `ground_truth`, and invokes the evaluator; `batch.py` is extended only to include `question`, `answer`, and `ground_truth` in the report `details`. A tiny `_csv.py` helper keeps CSV row parsing/generation in one place.

**Tech Stack:** Python 3.12, `csv` module, `argparse`, existing `rag_learn.pipeline.answer_stream`, `rag_learn.eval.tracing.JSONLEmitter`, `rag_learn.eval.batch`, pytest.

## Global Constraints

- Maintain `>= 80%` test coverage for `src/rag_learn`.
- Use type annotations on all function signatures.
- Prefer immutable data; do not mutate existing objects.
- Files should stay focused (target `< 800` lines).
- Follow existing import order and ruff lint rules (`E`, `F`, `I`, `B`, `UP`).
- The typechecker is Astral `ty` (`ty check src`), not mypy.
- `DEEPSEEK_API_KEY` is required for non-dry-run evaluation; tests must monkeypatch or use `--dry-run`.
- Do not bump pinned dependency versions in `pyproject.toml`.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/rag_learn/eval/_csv.py` | Parse a CSV row into `(question, collection, GroundTruth)` and format a row dict for writing. |
| `src/rag_learn/eval/sampler.py` | Load `rag_events_*.jsonl`, group by `collection`, sample N per collection, write CSV via `_csv.py`. |
| `src/rag_learn/eval/runner.py` | Read CSV, resolve collection, call `answer_stream`, emit events, then call `batch.main` to produce the report. |
| `src/rag_learn/eval/batch.py` | Extend report `details` to include `question`, `answer`, `ground_truth`. |
| `src/rag_learn/eval/cli.py` | `argparse` entry point with `sample`, `run`, `evaluate` subcommands. |
| `src/rag_learn/eval/__init__.py` | Export public names (unchanged except no new exports needed). |
| `tests/eval/test_csv.py` | Unit tests for `_csv.py`. |
| `tests/eval/test_sampler.py` | Unit tests for `sampler.py`. |
| `tests/eval/test_runner.py` | Unit tests for `runner.py`. |
| `tests/eval/test_batch.py` | Extend existing tests to verify enriched `details`. |
| `tests/eval/test_cli.py` | Smoke tests for argument parsing and dispatch. |

---

### Task 1: CSV row helper (`_csv.py`)

**Files:**
- Create: `src/rag_learn/eval/_csv.py`
- Test: `tests/eval/test_csv.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CSV_COLUMNS: list[str]` — ordered list of column names.
  - `parse_csv_row(row: dict[str, str], default_collection: str | None = None) -> tuple[str | None, str | None, GroundTruth | None]` — returns `(question, collection, ground_truth)`; returns `None` values when row is invalid.
  - `format_csv_row(question: str, collection: str) -> dict[str, str]` — returns a row dict with empty optional fields.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_csv.py`:

```python
"""Tests for rag_learn.eval._csv."""

from __future__ import annotations

from rag_learn.eval._csv import format_csv_row, parse_csv_row


def test_parse_csv_row_minimal():
    question, collection, gt = parse_csv_row({"question": "q", "collection": "rag_doc"})
    assert question == "q"
    assert collection == "rag_doc"
    assert gt is not None
    assert gt.answer is None
    assert gt.source_files == ()
    assert gt.chunk_ids == ()


def test_parse_csv_row_uses_default_collection():
    question, collection, gt = parse_csv_row(
        {"question": "q"}, default_collection="rag_doc"
    )
    assert collection == "rag_doc"
    assert question == "q"


def test_parse_csv_row_csv_collection_beats_default():
    question, collection, gt = parse_csv_row(
        {"question": "q", "collection": "other"}, default_collection="rag_doc"
    )
    assert collection == "other"


def test_parse_csv_row_missing_question_returns_none():
    assert parse_csv_row({"collection": "rag_doc"}) == (None, None, None)


def test_parse_csv_row_missing_collection_returns_none():
    assert parse_csv_row({"question": "q"}) == (None, None, None)


def test_parse_csv_row_splits_semicolon_lists():
    question, collection, gt = parse_csv_row({
        "question": "q",
        "answer": "a",
        "source_files": "a.md; b.md ",
        "chunk_ids": "a.md#0; b.md#1 ",
        "collection": "rag_doc",
    })
    assert gt.answer == "a"
    assert gt.source_files == ("a.md", "b.md")
    assert gt.chunk_ids == ("a.md#0", "b.md#1")


def test_format_csv_row():
    row = format_csv_row("q", "rag_doc")
    assert row == {
        "question": "q",
        "answer": "",
        "source_files": "",
        "chunk_ids": "",
        "collection": "rag_doc",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_csv.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag_learn.eval._csv'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rag_learn/eval/_csv.py`:

```python
"""CSV row parsing and formatting for batch evaluation."""

from __future__ import annotations

from rag_learn.eval.tracing import GroundTruth

CSV_COLUMNS: list[str] = [
    "question",
    "answer",
    "source_files",
    "chunk_ids",
    "collection",
]


def _split_semicolon(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(";") if part.strip())


def parse_csv_row(
    row: dict[str, str],
    default_collection: str | None = None,
) -> tuple[str | None, str | None, GroundTruth | None]:
    """Parse a CSV row into question, collection, and optional ground truth.

    Returns (None, None, None) when the row is invalid (missing question or
    unresolved collection).
    """
    question = row.get("question", "").strip()
    if not question:
        return None, None, None

    collection = row.get("collection", "").strip() or default_collection
    if not collection:
        return None, None, None

    ground_truth = GroundTruth(
        answer=row.get("answer", "").strip() or None,
        source_files=_split_semicolon(row.get("source_files")),
        chunk_ids=_split_semicolon(row.get("chunk_ids")),
    )
    return question, collection, ground_truth


def format_csv_row(question: str, collection: str) -> dict[str, str]:
    """Return a row dict suitable for DictWriter, with empty optional fields."""
    return {
        "question": question,
        "answer": "",
        "source_files": "",
        "chunk_ids": "",
        "collection": collection,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_csv.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/_csv.py tests/eval/test_csv.py
git commit -m "feat(eval): add CSV row helper for batch evaluation"
```

---

### Task 2: Sampler (`sampler.py`)

**Files:**
- Create: `src/rag_learn/eval/sampler.py`
- Test: `tests/eval/test_sampler.py`

**Interfaces:**
- Consumes:
  - `RAGEvent` from `rag_learn.eval.tracing`
  - `event_from_dict` from `rag_learn.eval.tracing`
  - `format_csv_row`, `CSV_COLUMNS` from `rag_learn.eval._csv`
- Produces:
  - `sample_events(events_dir: Path, samples_per_collection: int) -> list[dict[str, str]]` — returns row dicts ready for CSV writing.
  - `write_samples(rows: list[dict[str, str]], output_path: Path) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_sampler.py`:

```python
"""Tests for rag_learn.eval.sampler."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    rows = [{"question": "q1", "answer": "", "source_files": "", "chunk_ids": "", "collection": "c1"}]
    output = tmp_path / "samples.csv"
    write_samples(rows, output)

    text = output.read_text(encoding="utf-8")
    assert "question,answer,source_files,chunk_ids,collection" in text
    assert "q1,,,,c1" in text


def test_sample_events_skips_collections_with_no_events(tmp_path: Path) -> None:
    assert sample_events(tmp_path, samples_per_collection=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_sampler.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag_learn.eval.sampler'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rag_learn/eval/sampler.py`:

```python
"""Sample online RAG events into a CSV for manual labeling."""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

from rag_learn.eval._csv import CSV_COLUMNS, format_csv_row
from rag_learn.eval.tracing import event_from_dict

logger = logging.getLogger(__name__)


def _load_events(events_dir: Path) -> list[dict[str, str]]:
    """Load all rag_events_*.jsonl files and return raw dicts."""
    records: list[dict[str, str]] = []
    for path in sorted(events_dir.glob("rag_events_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping corrupted line %s:%d: %s", path, line_number, exc)
    return records


def sample_events(events_dir: Path, samples_per_collection: int) -> list[dict[str, str]]:
    """Sample up to N events per collection and return CSV row dicts."""
    records = _load_events(events_dir)
    by_collection: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        collection = record.get("collection", "")
        if not collection:
            continue
        by_collection[collection].append(record)

    selected: list[dict[str, str]] = []
    for collection, items in by_collection.items():
        sample = random.sample(items, min(samples_per_collection, len(items)))
        for item in sample:
            selected.append(format_csv_row(question=item["question"], collection=collection))
    return selected


def write_samples(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write sample rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
```

Add missing `json` import at the top:

```python
import csv
import json
import logging
import random
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_sampler.py -v
```

Expected: all tests pass. Note: `test_sample_events_limits_per_collection` may be flaky due to randomness; if so, set a fixed seed in the test via `random.seed(0)` before calling `sample_events`.

Fix flakiness by updating `test_sample_events_limits_per_collection`:

```python
import random

def test_sample_events_limits_per_collection(tmp_path: Path) -> None:
    random.seed(0)
    ...
```

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/sampler.py tests/eval/test_sampler.py
git commit -m "feat(eval): add online event sampler"
```

---

### Task 3: Extend batch.py report details

**Files:**
- Modify: `src/rag_learn/eval/batch.py:188-190`
- Test: `tests/eval/test_batch.py`

**Interfaces:**
- Consumes: `RAGEvent`
- Produces: report `details` entries enriched with `question`, `answer`, `ground_truth`.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_batch.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_batch.py::test_batch_details_include_question_answer_ground_truth -v
```

Expected: `KeyError` on `details["question"]`.

- [ ] **Step 3: Write minimal implementation**

In `src/rag_learn/eval/batch.py`, replace lines 188-190:

```python
        details.append(
            {"trace_id": event.trace_id, "collection": event.collection, "metrics": event_metrics}
        )
```

with:

```python
        details.append(
            {
                "trace_id": event.trace_id,
                "collection": event.collection,
                "question": event.question,
                "answer": event.answer,
                "ground_truth": _ground_truth_to_dict(event.ground_truth),
                "metrics": event_metrics,
            }
        )
```

Add helper function near `_aggregate`:

```python
def _ground_truth_to_dict(gt: GroundTruth | None) -> dict[str, Any] | None:
    if gt is None:
        return None
    return {
        "answer": gt.answer,
        "source_files": list(gt.source_files),
        "chunk_ids": list(gt.chunk_ids),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_batch.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/batch.py tests/eval/test_batch.py
git commit -m "feat(eval): include question/answer/ground_truth in batch report details"
```

---

### Task 4: Runner (`runner.py`)

**Files:**
- Create: `src/rag_learn/eval/runner.py`
- Test: `tests/eval/test_runner.py`

**Interfaces:**
- Consumes:
  - `Catalog`, `Collection` from `rag_learn.collections`
  - `load_config` from `rag_learn.config`
  - `answer_stream` from `rag_learn.pipeline`
  - `JSONLEmitter`, `RAGEvent` from `rag_learn.eval.tracing`
  - `parse_csv_row` from `rag_learn.eval._csv`
  - `main` from `rag_learn.eval.batch`
- Produces:
  - `run_qa_csv(qa_csv: Path, default_collection: str | None, output_events: Path, output_report: Path, judge_model: str | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_runner.py`:

```python
"""Tests for rag_learn.eval.runner."""

from __future__ import annotations

import csv
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
            perf = StreamPerf(retrieve_ms=1.0, first_token_ms=2.0, total_ms=3.0, finished_at="10:00:00")
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
    events_path = tmp_path / "batch_events.jsonl"
    report_path = tmp_path / "report.json"

    rc = run_qa_csv(csv_path, None, events_path, report_path, judge_model="dummy")
    assert rc == 0

    assert events_path.is_file()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_events"] == 1
    details = report["details"][0]
    assert details["question"] == "What is RAG?"
    assert details["ground_truth"]["answer"] == " retrieval-augmented generation "
    assert details["ground_truth"]["source_files"] == ["a.md"]


def test_run_qa_csv_skips_invalid_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setattr("rag_learn.eval.runner.answer_stream", lambda *a, **k: {})

    csv_path = tmp_path / "qa.csv"
    _write_csv(
        csv_path,
        [
            {"question": "", "answer": "", "source_files": "", "chunk_ids": "", "collection": "rag_doc"},
            {"question": "q", "answer": "", "source_files": "", "chunk_ids": "", "collection": ""},
        ],
    )
    events_path = tmp_path / "batch_events.jsonl"
    report_path = tmp_path / "report.json"

    rc = run_qa_csv(csv_path, None, events_path, report_path)
    assert rc == 0
    assert not events_path.exists() or events_path.read_text(encoding="utf-8").strip() == ""
```

Add missing `json` import at the top:

```python
import csv
import json
from pathlib import Path
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag_learn.eval.runner'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rag_learn/eval/runner.py`:

```python
"""Run a prepared Q&A CSV through the RAG pipeline and evaluate it."""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_learn.collections import Catalog, build_catalog
from rag_learn.config import load_config
from rag_learn.eval._csv import parse_csv_row
from rag_learn.eval.batch import main as batch_main
from rag_learn.eval.tracing import JSONLEmitter, RAGEvent
from rag_learn.llm import DeepSeekLLM
from rag_learn.pipeline import answer_stream

logger = logging.getLogger(__name__)


def _load_catalog() -> Catalog:
    config = load_config()
    return build_catalog(config)


def _make_llm(config: Any) -> DeepSeekLLM:
    return DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )


def _process_row(
    row: dict[str, str],
    default_collection: str | None,
    catalog: Catalog,
    llm: DeepSeekLLM,
    emitter: JSONLEmitter,
    metadata: dict[str, Any],
    k: int,
) -> None:
    question, collection_slug, ground_truth = parse_csv_row(row, default_collection)
    if question is None or collection_slug is None:
        logger.warning("Skipping invalid CSV row: %r", row)
        return

    try:
        collection = catalog.get(collection_slug)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Collection %r not found: %s", collection_slug, exc)
        return

    try:
        result = answer_stream(
            {collection_slug: collection.retriever},
            llm,
            question,
            k=k,
            emitter=None,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("answer_stream failed for question %r: %s", question, exc)
        return

    side = result.get(collection_slug)
    if side is None:
        logger.error("No result for collection %r", collection_slug)
        return

    stream_iter, hits, perf_fn = side
    answer = "".join(list(stream_iter))
    perf = perf_fn(answer)

    event = RAGEvent(
        trace_id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        collection=collection_slug,
        question=question,
        hits=tuple(hits),
        prompt="",
        answer=answer,
        perf=perf,
        ground_truth=ground_truth,
        metadata={**metadata, "k": k},
    )
    emitter.emit(event)


def run_qa_csv(
    qa_csv: Path,
    default_collection: str | None,
    output_events: Path,
    output_report: Path,
    judge_model: str | None = None,
) -> int:
    """Read a Q&A CSV, run each question through RAG, emit events, and evaluate."""
    config = load_config()
    catalog = _load_catalog()
    llm = _make_llm(config)
    emitter = JSONLEmitter(output_events.parent)
    metadata = {"llm_model": config.llm_model}

    with open(qa_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _process_row(
                row,
                default_collection,
                catalog,
                llm,
                emitter,
                metadata,
                config.retrieve_k,
            )

    return batch_main(
        [
            str(output_events.parent),
            "--output",
            str(output_report),
            "--judge-model",
            judge_model or config.llm_model,
        ]
    )
```

Note: `output_events.parent` is passed to `JSONLEmitter` because the emitter chooses the filename from the timestamp. The actual `output_events` argument is only used to determine the directory; the filename will be `batch_events_YYYY-MM-DD.jsonl` inside that directory. This is acceptable per the spec, but document it in the CLI help.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_runner.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/runner.py tests/eval/test_runner.py
git commit -m "feat(eval): add batch Q&A runner"
```

---

### Task 5: CLI entry point (`cli.py`)

**Files:**
- Create: `src/rag_learn/eval/cli.py`
- Test: `tests/eval/test_cli.py`

**Interfaces:**
- Consumes:
  - `sample_events`, `write_samples` from `rag_learn.eval.sampler`
  - `run_qa_csv` from `rag_learn.eval.runner`
  - `main` from `rag_learn.eval.batch`
- Produces:
  - `main(argv: list[str] | None = None) -> int` — CLI entry point.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_cli.py`:

```python
"""Tests for rag_learn.eval.cli."""

from __future__ import annotations

from rag_learn.eval import cli


def test_cli_sample_dispatched(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli.sampler, "sample_events", lambda _dir, n: calls.append(("sample", n)) or [])
    monkeypatch.setattr(cli.sampler, "write_samples", lambda rows, path: calls.append(("write", path)))
    rc = cli.main(["sample", "data", "--samples-per-collection", "3", "--output", "out.csv"])
    assert rc == 0
    assert ("sample", 3) in calls
    assert any(c[0] == "write" and str(c[1]) == "out.csv" for c in calls)


def test_cli_run_dispatched(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli.runner, "run_qa_csv", lambda *args, **kwargs: calls.append(args) or 0)
    rc = cli.main([
        "run", "qa.csv",
        "--collection", "rag_doc",
        "--output-events", "data/events.jsonl",
        "--output-report", "data/report.json",
    ])
    assert rc == 0
    assert len(calls) == 1
    args = calls[0]
    assert str(args[0]) == "qa.csv"
    assert args[1] == "rag_doc"


def test_cli_evaluate_dispatched(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli.batch, "main", lambda argv: calls.append(argv) or 0)
    rc = cli.main(["evaluate", "data", "--output", "report.json", "--dry-run"])
    assert rc == 0
    assert len(calls) == 1
    assert "data" in calls[0]
    assert "--dry-run" in calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag_learn.eval.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rag_learn/eval/cli.py`:

```python
"""CLI entry point for batch RAG evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_learn.eval import batch, runner, sampler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch RAG evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Sample online events into a CSV")
    sample_parser.add_argument("events_dir", type=Path)
    sample_parser.add_argument("--samples-per-collection", type=int, default=5)
    sample_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="Run a Q&A CSV through RAG and evaluate")
    run_parser.add_argument("qa_csv", type=Path)
    run_parser.add_argument("--collection", default=None)
    run_parser.add_argument("--output-events", type=Path, required=True)
    run_parser.add_argument("--output-report", type=Path, required=True)
    run_parser.add_argument("--judge-model", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing events")
    eval_parser.add_argument("events_dir", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--judge-model", default=None)
    eval_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "sample":
        rows = sampler.sample_events(args.events_dir, args.samples_per_collection)
        sampler.write_samples(rows, args.output)
        return 0

    if args.command == "run":
        return runner.run_qa_csv(
            args.qa_csv,
            args.collection,
            args.output_events,
            args.output_report,
            judge_model=args.judge_model,
        )

    if args.command == "evaluate":
        batch_argv = [str(args.events_dir), "--output", str(args.output)]
        if args.judge_model:
            batch_argv.extend(["--judge-model", args.judge_model])
        if args.dry_run:
            batch_argv.append("--dry-run")
        return batch.main(batch_argv)

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/cli.py tests/eval/test_cli.py
git commit -m "feat(eval): add batch evaluation CLI entry point"
```

---

### Task 6: Integration test and final verification

**Files:**
- Test: `tests/eval/test_runner.py` (append) or create `tests/eval/test_integration.py`

- [ ] **Step 1: Add an end-to-end integration test**

Append to `tests/eval/test_runner.py` or create `tests/eval/test_integration.py`:

```python
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
            perf = StreamPerf(1.0, 1.0, 1.0, "10:00:00")
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
    monkeypatch.setattr("rag_learn.eval.batch._make_judge_fn", lambda _config, _model: lambda s, u: "4")

    from rag_learn.eval.cli import main as cli_main

    csv_path = tmp_path / "qa.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source_files", "chunk_ids", "collection"])
        writer.writeheader()
        writer.writerow({
            "question": "q1",
            "answer": "a1",
            "source_files": "a.md",
            "chunk_ids": "",
            "collection": "rag_doc",
        })

    rc = cli_main([
        "run", str(csv_path),
        "--output-events", str(tmp_path / "events.jsonl"),
        "--output-report", str(tmp_path / "report.json"),
    ])
    assert rc == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["total_events"] == 1
    assert report["details"][0]["question"] == "q1"
```

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest tests/eval/test_integration.py -v
```

Expected: pass.

- [ ] **Step 3: Run lint, typecheck, and full test suite**

```bash
make all
```

Expected: ruff passes, ty passes, pytest passes with `>= 80%` coverage.

If `test_milvus_retriever.py` fails due to the pre-existing NumPy environment issue, run:

```bash
uv run pytest --ignore=tests/test_milvus_retriever.py -q
```

and confirm coverage is still `>= 80%`.

- [ ] **Step 4: Update documentation**

Append a short usage section to the design doc or create a `docs/superpowers/usage/batch-evaluation.md` summarizing the CLI commands. At minimum, update `README.md` with the new commands if it has an evaluation section.

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_integration.py README.md
git commit -m "test(eval): add end-to-end batch evaluation test and docs"
```

---

## Self-Review

**Spec coverage:**
- CSV format with `;` separator → Task 1 `_csv.py`
- `sample` subcommand → Task 2 `sampler.py`
- `run` subcommand → Task 4 `runner.py` + Task 5 `cli.py`
- `evaluate` subcommand → Task 5 `cli.py` reuses `batch.main`
- Report includes `question`/`answer`/`ground_truth` → Task 3 `batch.py`
- Unified CSV template for sample and run → Task 1 `_csv.py` + Task 2 `sampler.py`
- Error handling (skip invalid rows, log error on answer_stream failure) → Task 4 `runner.py`
- Testing strategy → all tasks include tests

**Placeholder scan:**
- No TBD/TODO.
- All code blocks contain concrete implementations or concrete test code.
- All commands include expected outputs.

**Type consistency:**
- `parse_csv_row` returns `tuple[str | None, str | None, GroundTruth | None]` consistently.
- `run_qa_csv` signature matches CLI dispatch in Task 5.
- `batch.main` argv type is `list[str] | None` in all call sites.

**Open issue to resolve during implementation:**
- `runner.py` currently passes `output_events.parent` to `JSONLEmitter`, meaning the actual filename is `batch_events_YYYY-MM-DD.jsonl` inside that directory, not the exact `--output-events` path. If the user expects the exact path, adjust `JSONLEmitter` to accept an explicit filename or change the CLI argument name to `--output-events-dir`. Decide during Task 4 and update tests accordingly.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-22-batch-evaluation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach would you like?
