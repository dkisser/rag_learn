# RAG 埋点与效果评估实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `rag-learn` 链路上增加 `rag_learn/eval/` 评估包，实现可插拔 JSONL 埋点、检索指标、LLM-as-judge 指标和后台批量评估 CLI。

**Architecture:** 在 `pipeline.py` 的 `answer_stream` 中增加可选 `MetricsEmitter`，当 caller drain 完 token iterator 并传入最终 answer 时构造 `RAGEvent` 并 emit；`app.py` 实例化 `JSONLEmitter` 并传入；`rag_learn/eval/batch.py` 读取 JSONL，分流有/无 ground_truth 的事件并计算指标。

**Tech Stack:** Python 3.12, `dataclasses`, `pathlib`, `json`, `argparse`, pytest, ruff, ty. 不新增外部依赖。

## Global Constraints

- Python 版本：3.12
- 不新增 `pyproject.toml` 依赖；token 估算使用 4 字符 ≈ 1 token。
- 所有代码必须通过 `make all`（ruff lint、ty typecheck、pytest、覆盖率 ≥80%）。
- 遵循项目不可变数据原则：`@dataclass(frozen=True)`、`tuple` 代替可变列表。
- Emitter 失败必须 fail-open，不影响主链路返回回答。
- 测试使用 fake/mock，不调用真实 DeepSeek。
- 代码组织：`src/rag_learn/eval/` 包承载所有评估相关模块。

---

## File Structure

| 文件 | 责任 |
|------|------|
| `src/rag_learn/eval/__init__.py` | 导出公共 API：`GroundTruth`、`RAGEvent`、`MetricsEmitter`、`JSONLEmitter`、`ListEmitter`、`NullEmitter`、`event_from_dict`。 |
| `src/rag_learn/eval/tracing.py` | 定义事件模型、Emitter Protocol、`JSONLEmitter`（按天轮转 JSONL）、测试用 `ListEmitter` / `NullEmitter`、JSONL 反序列化。 |
| `src/rag_learn/eval/metrics.py` | 检索指标（Recall@K、Precision@K、MRR、NDCG@K）和 LLM-as-judge 指标（context_relevance、faithfulness、answer_relevance、overall_usefulness、answer_llm_correctness）。 |
| `src/rag_learn/eval/batch.py` | 批量评估 CLI：`python -m rag_learn.eval.batch data/ --output report.json --dry-run`。 |
| `src/rag_learn/pipeline.py` | 修改 `answer_stream`：新增 `emitter` 和 `metadata` 参数，`perf_fn` 签名改为 `Callable[[str], StreamPerf]`，drain 完 answer 后 emit `RAGEvent`。 |
| `src/rag_learn/app.py` | 修改 `build_app`：实例化 `JSONLEmitter(config.data_dir)`，调用 `answer_stream` 时传入 emitter 和 metadata。 |
| `tests/eval/test_tracing.py` | Emitter、序列化、反序列化、fail-open 测试。 |
| `tests/eval/test_metrics.py` | 检索指标和 judge 指标测试。 |
| `tests/eval/test_batch.py` | 批量评估 CLI 测试（dry-run、报告结构）。 |
| `tests/test_pipeline.py` / `tests/test_pipeline_parallel.py` / `tests/test_e2e.py` | 更新 `perf_fn()` → `perf_fn(answer)` 调用。 |
| `tests/test_app_launch.py` | 扩展：验证 `build_app` 正确传 emitter。 |

---

### Task 1: 创建 `rag_learn/eval` 包与事件模型

**Files:**
- Create: `src/rag_learn/eval/__init__.py`
- Create: `src/rag_learn/eval/tracing.py`
- Test: `tests/eval/test_tracing.py`

**Interfaces:**
- Produces: `GroundTruth`, `RAGEvent`, `MetricsEmitter`, `event_from_dict` 的签名与行为。
- Produces: `RAGEvent` 字段：`trace_id: str`, `timestamp: str`, `collection: str`, `question: str`, `hits: list[Hit]`, `prompt: str`, `answer: str`, `perf: StreamPerf`, `ground_truth: GroundTruth | None`, `metadata: dict[str, Any]`。

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_tracing.py
import json
from pathlib import Path

from rag_learn.eval.tracing import (
    GroundTruth,
    JSONLEmitter,
    RAGEvent,
    event_from_dict,
)
from rag_learn.pipeline import StreamPerf
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
        hits=[Hit(text="chunk", source_file="a.md", chunk_index=0, score=0.1)],
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_tracing.py::test_event_round_trip_via_jsonl -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_learn.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag_learn/eval/__init__.py
from rag_learn.eval.tracing import (
    GroundTruth,
    JSONLEmitter,
    ListEmitter,
    MetricsEmitter,
    NullEmitter,
    RAGEvent,
    event_from_dict,
)

__all__ = [
    "GroundTruth",
    "JSONLEmitter",
    "ListEmitter",
    "MetricsEmitter",
    "NullEmitter",
    "RAGEvent",
    "event_from_dict",
]
```

```python
# src/rag_learn/eval/tracing.py
"""RAG event model, emitters, and JSONL serialization."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from rag_learn.pipeline import StreamPerf
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
    hits: list[Hit]
    prompt: str
    answer: str
    perf: StreamPerf
    ground_truth: GroundTruth | None
    metadata: dict[str, Any]


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
        hits=[Hit(**h) for h in data["hits"]],
        prompt=data["prompt"],
        answer=data["answer"],
        perf=StreamPerf(**data["perf"]),
        ground_truth=ground_truth,
        metadata=data.get("metadata", {}),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_tracing.py::test_event_round_trip_via_jsonl -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/ tests/eval/test_tracing.py
git commit -m "feat(eval): add RAGEvent, emitters, and JSONL tracing"
```

---

### Task 2: Emitter 行为测试（ListEmitter、NullEmitter、JSONLEmitter fail-open）

**Files:**
- Modify: `src/rag_learn/eval/tracing.py`
- Test: `tests/eval/test_tracing.py`

**Interfaces:**
- Consumes: `RAGEvent`, `JSONLEmitter`, `ListEmitter`, `NullEmitter` from Task 1.
- Produces: `JSONLEmitter` 按天轮转、`ListEmitter` 收集事件、`NullEmitter` 不操作、emit 失败不抛异常。

- [ ] **Step 1: Write the failing tests**

追加到 `tests/eval/test_tracing.py`：

```python
def test_list_emitter_collects_events() -> None:
    from rag_learn.eval.tracing import ListEmitter

    emitter = ListEmitter()
    event = _make_event()
    emitter.emit(event)
    assert emitter.events == [event]


def test_null_emitter_does_nothing() -> None:
    from rag_learn.eval.tracing import NullEmitter

    emitter = NullEmitter()
    emitter.emit(_make_event())


def test_jsonl_emitter_appends_to_same_day(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    emitter.emit(_make_event(trace_id="t1"))
    emitter.emit(_make_event(trace_id="t2"))

    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        assert len(f.readlines()) == 2


def test_jsonl_emitter_fails_open(tmp_path: Path) -> None:
    emitter = JSONLEmitter(tmp_path)
    tmp_path.chmod(0o555)
    try:
        emitter.emit(_make_event())
    finally:
        tmp_path.chmod(0o755)
    files = list(tmp_path.glob("rag_events_*.jsonl"))
    assert len(files) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_tracing.py -v`

Expected: FAIL on `test_list_emitter_collects_events` / `test_null_emitter_does_nothing` / `test_jsonl_emitter_fails_open` because those classes may not be exported yet.

- [ ] **Step 3: Verify implementation covers tests**

确保 `src/rag_learn/eval/tracing.py` 已包含 `ListEmitter`、`NullEmitter`、按天轮转路径、`try/except` fail-open（Task 1 已实现）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_tracing.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_tracing.py
git commit -m "test(eval): cover emitter behaviors and fail-open"
```

---

### Task 3: 实现检索指标

**Files:**
- Create: `src/rag_learn/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `Hit` from `rag_learn.retriever`.
- Produces: `retrieval_recall_at_k(hits, source_files, k) -> float`, `retrieval_precision_at_k(...) -> float`, `retrieval_mrr(...) -> float`, `retrieval_ndcg_at_k(...) -> float`。

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
from rag_learn.eval.metrics import (
    retrieval_mrr,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
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


def test_retrieval_mrr() -> None:
    hits = [_hit("x.md"), _hit("a.md"), _hit("b.md")]
    assert retrieval_mrr(hits, ("a.md",)) == 0.5
    assert retrieval_mrr([_hit("a.md")], ("a.md",)) == 1.0
    assert retrieval_mrr([_hit("a.md")], ("c.md",)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_metrics.py::test_retrieval_recall_at_k -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag_learn/eval/metrics.py
"""Retrieval and generation evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from rag_learn.eval.tracing import GroundTruth, RAGEvent
from rag_learn.retriever import Hit


def retrieval_recall_at_k(hits: list[Hit], source_files: tuple[str, ...], k: int) -> float:
    if not source_files:
        return 0.0
    retrieved = {h.source_file for h in hits[:k]}
    relevant = set(source_files)
    return len(retrieved & relevant) / len(relevant)


def retrieval_precision_at_k(hits: list[Hit], source_files: tuple[str, ...], k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    relevant = set(source_files)
    return sum(1 for h in top if h.source_file in relevant) / len(top)


def retrieval_mrr(hits: list[Hit], source_files: tuple[str, ...]) -> float:
    relevant = set(source_files)
    for rank, h in enumerate(hits, start=1):
        if h.source_file in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_ndcg_at_k(hits: list[Hit], source_files: tuple[str, ...], k: int) -> float:
    relevant = set(source_files)
    dcg = 0.0
    for i, h in enumerate(hits[:k], start=1):
        rel = 1.0 if h.source_file in relevant else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_rels = [1.0] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_metrics.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): add retrieval metrics"
```

---

### Task 4: 实现答案 F1 与 LLM-as-judge 指标

**Files:**
- Modify: `src/rag_learn/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `RAGEvent`, `GroundTruth` from `tracing.py`.
- Produces: `answer_f1(answer, ground_truth) -> float`, `context_relevance(event, judge_fn) -> float | None`, `faithfulness(...)`, `answer_relevance(...)`, `overall_usefulness(...)`, `answer_llm_correctness(...)`。
- `judge_fn(system: str, user: str) -> str` 返回原始文本，指标函数负责提取 1-5 分并归一化到 0-1。

- [ ] **Step 1: Write the failing tests**

追加到 `tests/eval/test_metrics.py`：

```python
import pytest

from rag_learn.eval.metrics import (
    answer_f1,
    answer_llm_correctness,
    context_relevance,
    faithfulness,
    answer_relevance,
    overall_usefulness,
)
from rag_learn.eval.tracing import GroundTruth, RAGEvent
from rag_learn.pipeline import StreamPerf
from rag_learn.retriever import Hit


def _make_event(
    question: str = "q",
    answer: str = "a",
    hits: list[Hit] | None = None,
    ground_truth: GroundTruth | None = None,
) -> RAGEvent:
    return RAGEvent(
        trace_id="t1",
        timestamp="2026-07-22T10:00:00+00:00",
        collection="c",
        question=question,
        hits=hits or [],
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
    event = _make_event(question="what is RAG?", hits=[_hit("a.md")])

    def judge(system: str, user: str) -> str:
        return "Score: 4"

    assert context_relevance(event, judge) == 0.8


def test_faithfulness_extracts_score() -> None:
    event = _make_event(question="q", answer="a", hits=[_hit("a.md")])

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_metrics.py -v`

Expected: FAIL on new tests

- [ ] **Step 3: Write minimal implementation**

追加到 `src/rag_learn/eval/metrics.py`：

```python
import re


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
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def _extract_score(text: str, scale: tuple[int, int] = (1, 5)) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    raw = float(match.group(1))
    lo, hi = scale
    clamped = max(lo, min(hi, raw))
    return clamped / hi


def _context_text(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))


def context_relevance(event: RAGEvent, judge_fn: Callable[[str, str], str]) -> float | None:
    system = (
        "You are an evaluator. Rate how relevant the retrieved context is to the question. "
        "Output only a number from 1 to 5, where 1 = completely irrelevant, 5 = highly relevant."
    )
    user = f"Question: {event.question}\n\nContext:\n{_context_text(event.hits)}\n\nRelevance score (1-5):"
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
    user = f"Question: {event.question}\n\nAnswer: {event.answer}\n\nOverall usefulness score (1-5):"
    return _extract_score(judge_fn(system, user))


def answer_llm_correctness(
    event: RAGEvent, judge_fn: Callable[[str, str], str]
) -> float | None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_metrics.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): add answer F1 and LLM-as-judge metrics"
```

---

### Task 5: 修改 `pipeline.py` 接入 Emitter

**Files:**
- Modify: `src/rag_learn/pipeline.py`
- Modify: `tests/test_pipeline_parallel.py`
- Modify: `tests/test_e2e.py`
- Test: `tests/test_pipeline.py`（新增 emitter 用例）

**Interfaces:**
- Consumes: `MetricsEmitter`, `RAGEvent` from `rag_learn.eval`。
- Produces: `answer_stream` 返回 `Callable[[str], StreamPerf]`；无 emitter 时行为不变。

- [ ] **Step 1: Write the failing tests**

追加到 `tests/test_pipeline.py`：

```python
from collections.abc import Iterator

from rag_learn.eval.tracing import ListEmitter, RAGEvent
from rag_learn.pipeline import StreamPerf, answer_stream
from rag_learn.retriever.base import Hit


class _FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self._hits


class _FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield from self._tokens


def test_answer_stream_emits_event_when_perf_fn_called_with_answer() -> None:
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits)}
    llm = _FakeLLM(["hello", "world"])
    emitter = ListEmitter()

    out = answer_stream(
        retrievers,
        llm,
        "Q?",
        k=5,
        emitter=emitter,
        metadata={"llm_model": "dummy"},
    )

    stream, retrieved_hits, perf_fn = out["chroma"]
    answer = "".join(stream)
    perf = perf_fn(answer)

    assert isinstance(perf, StreamPerf)
    assert len(emitter.events) == 1
    event = emitter.events[0]
    assert isinstance(event, RAGEvent)
    assert event.collection == "chroma"
    assert event.question == "Q?"
    assert event.answer == answer
    assert event.hits == retrieved_hits
    assert event.metadata == {"llm_model": "dummy"}


def test_answer_stream_no_emitter_does_not_record() -> None:
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits)}
    llm = _FakeLLM(["ok"])

    out = answer_stream(retrievers, llm, "Q?")
    stream, _, perf_fn = out["chroma"]
    answer = "".join(stream)
    perf = perf_fn(answer)

    assert isinstance(perf, StreamPerf)
```

修改 `tests/test_pipeline_parallel.py` 中所有 `perf_fn()` 为 `perf_fn("")`：

```python
# 在 test_answer_stream_returns_both_sides 中：
for stream, _, perf_fn in out.values():
    list(stream)
    assert isinstance(perf_fn(""), StreamPerf)
```

其它三处同理。

修改 `tests/test_e2e.py` 中 `perf = perf_fn()` 为 `perf = perf_fn("TEST ANSWER")`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_answer_stream_emits_event_when_perf_fn_called_with_answer -v`

Expected: FAIL because `answer_stream` has no `emitter` parameter

- [ ] **Step 3: Write minimal implementation**

修改 `src/rag_learn/pipeline.py`：

1. 新增 import：

```python
import uuid
from datetime import datetime, timezone
from typing import Any

from rag_learn.eval.tracing import MetricsEmitter, RAGEvent
```

2. 修改 `answer_stream` 签名与实现：

```python
def answer_stream(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
    emitter: MetricsEmitter | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]]:
    """Parallel retrieve → build prompt per side → stream tokens per side.

    Returns ``{name: (token_iterator, hits, perf_fn)}``. The ``perf_fn``
    callable accepts the fully drained answer text, optionally emits a
    ``RAGEvent`` if an emitter was provided, and returns the populated
    :class:`StreamPerf`. It MUST be invoked AFTER the token iterator is
    fully drained by the caller.
    """
    event_metadata = metadata or {}
    retrieve_started = time.perf_counter()
    hits_by_side = _retrieve(retrievers, question, k)
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0

    def _side(
        name: str,
        hits: list[Hit],
    ) -> tuple[Iterator[str], list[Hit], Callable[[str], StreamPerf]]:
        sys_msg, user_msg = build_prompt(hits, question)
        prompt_text = f"{sys_msg}\n\n{user_msg}"
        started = time.perf_counter()
        out_perf_holder: list[StreamPerf] = []

        class _TimedIter:
            def __init__(self) -> None:
                self._gen: Iterator[str] = iter(llm.stream(sys_msg, user_msg))
                self.first_token_at: float | None = None
                self.end_at: float | None = None
                self._done = False

            def __iter__(self) -> _TimedIter:
                return self

            def __next__(self) -> str:
                if self._done:
                    raise StopIteration
                try:
                    tok = next(self._gen)
                except StopIteration:
                    self._done = True
                    self.end_at = time.perf_counter()
                    first = self.first_token_at if self.first_token_at is not None else self.end_at
                    out_perf_holder.append(_make_perf(retrieve_ms, started, first, self.end_at))
                    raise
                if self.first_token_at is None:
                    self.first_token_at = time.perf_counter()
                return tok

        it = _TimedIter()

        def get_perf(answer: str) -> StreamPerf:
            perf = out_perf_holder[0]
            if emitter is not None:
                event = RAGEvent(
                    trace_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    collection=name,
                    question=question,
                    hits=hits,
                    prompt=prompt_text,
                    answer=answer,
                    perf=perf,
                    ground_truth=None,
                    metadata={"k": k, **event_metadata},
                )
                emitter.emit(event)
            return perf

        return it, hits, get_perf

    return {name: _side(name, hits_by_side[name]) for name in hits_by_side}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/test_pipeline_parallel.py tests/test_e2e.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/pipeline.py tests/test_pipeline.py tests/test_pipeline_parallel.py tests/test_e2e.py
git commit -m "feat(pipeline): wire MetricsEmitter into answer_stream"
```

---

### Task 6: 修改 `app.py` 接入 `JSONLEmitter`

**Files:**
- Modify: `src/rag_learn/app.py`
- Test: `tests/test_app_launch.py`

**Interfaces:**
- Consumes: `JSONLEmitter` from `rag_learn.eval`。
- Produces: `build_app` 实例化 emitter 并传给 `answer_stream`；每次用户提交成功后 `data/rag_events_YYYY-MM-DD.jsonl` 增加一行。

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_app_launch.py`：

```python
import json

from rag_learn.eval.tracing import event_from_dict


def test_build_app_logs_event_to_jsonl(stub_catalog: Catalog, tmp_path: Path):
    config = _make_config(tmp_path)
    app = build_app(catalog=stub_catalog, llm=_stub_llm(), config=config)

    submit_fn = app.fns[0].fn
    outputs = submit_fn("aaa", "hello")
    assert isinstance(outputs, list) and len(outputs) == 5

    files = list(tmp_path.glob("data/rag_events_*.jsonl"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        data = json.loads(f.readline())
    event = event_from_dict(data)
    assert event.collection == "aaa"
    assert event.question == "hello"
    assert event.answer == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_launch.py::test_build_app_logs_event_to_jsonl -v`

Expected: FAIL because `build_app` does not yet create JSONLEmitter

- [ ] **Step 3: Write minimal implementation**

修改 `src/rag_learn/app.py`：

1. 新增 import：

```python
from rag_learn.eval import JSONLEmitter
```

2. 在 `build_app` 中实例化 emitter：

```python
def build_app(
    catalog: Catalog,
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks:
    emitter = JSONLEmitter(config.data_dir)
    ...
```

3. 在 `on_submit` 的 `answer_stream` 调用中传入 emitter 和 metadata：

```python
outputs = answer_stream(
    {collection_slug: retriever},
    llm,
    q,
    k=config.retrieve_k,
    emitter=emitter,
    metadata={"llm_model": config.llm_model},
)
```

4. 在 drain 完后调用 `perf_fn(answer_text)`：

```python
answer_text = _drain_to_chatbot(stream_iter)
...
perf = perf_fn(answer_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_launch.py::test_build_app_logs_event_to_jsonl -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/app.py tests/test_app_launch.py
git commit -m "feat(app): wire JSONLEmitter into build_app"
```

---

### Task 7: 实现批量评估 CLI `rag_learn/eval/batch.py`

**Files:**
- Create: `src/rag_learn/eval/batch.py`
- Test: `tests/eval/test_batch.py`

**Interfaces:**
- Consumes: `RAGEvent`, `event_from_dict` from `tracing.py`；`retrieval_*`、`answer_f1`、judge 指标 from `metrics.py`。
- Produces: `main(argv=None) -> int` CLI；输出 JSON 报告文件。

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_batch.py
import json
from pathlib import Path

import pytest

from rag_learn.eval.batch import main
from rag_learn.eval.tracing import GroundTruth, JSONLEmitter, RAGEvent
from rag_learn.pipeline import StreamPerf
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
        hits=[Hit(text="chunk", source_file="a.md", chunk_index=0, score=0.1)],
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
    assert report["with_ground_truth"] == len(report["details"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_batch.py::test_batch_dry_run_reports_counts -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag_learn/eval/batch.py
"""Batch evaluation CLI for RAG events stored in JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from rag_learn.config import load_config
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
from rag_learn.eval.tracing import RAGEvent, event_from_dict
from rag_learn.llm import DeepSeekLLM

logger = logging.getLogger(__name__)


def _load_events(events_dir: Path) -> tuple[list[RAGEvent], int]:
    events: list[RAGEvent] = []
    skipped = 0
    for path in sorted(events_dir.glob("rag_events_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(event_from_dict(data))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping corrupted line %s:%d: %s", path, line_number, exc)
                    skipped += 1
    return events, skipped


def _dedupe(events: list[RAGEvent]) -> list[RAGEvent]:
    seen: set[str] = set()
    result: list[RAGEvent] = []
    for event in events:
        if event.trace_id in seen:
            continue
        seen.add(event.trace_id)
        result.append(event)
    return result


def _compute_supervised(event: RAGEvent, k: int) -> dict[str, Any]:
    assert event.ground_truth is not None
    source_files = event.ground_truth.source_files
    metrics: dict[str, Any] = {
        f"retrieval_recall@{k}": retrieval_recall_at_k(event.hits, source_files, k),
        f"retrieval_precision@{k}": retrieval_precision_at_k(event.hits, source_files, k),
        "retrieval_mrr": retrieval_mrr(event.hits, source_files),
        f"retrieval_ndcg@{k}": retrieval_ndcg_at_k(event.hits, source_files, k),
    }
    if event.ground_truth.answer:
        metrics["answer_f1"] = answer_f1(event.answer, event.ground_truth.answer)
    return metrics


def _compute_unsupervised(
    event: RAGEvent, judge_fn: Callable[[str, str], str]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "context_relevance": context_relevance(event, judge_fn),
        "faithfulness": faithfulness(event, judge_fn),
        "answer_relevance": answer_relevance(event, judge_fn),
        "overall_usefulness": overall_usefulness(event, judge_fn),
    }
    if event.ground_truth is not None and event.ground_truth.answer:
        metrics["answer_llm_correctness"] = answer_llm_correctness(event, judge_fn)
    return metrics


def _aggregate(values: list[float | None]) -> dict[str, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    clean.sort()
    n = len(clean)
    mean = sum(clean) / n
    median = clean[n // 2] if n % 2 else (clean[n // 2 - 1] + clean[n // 2]) / 2
    p95_idx = int(n * 0.95)
    p95 = clean[min(p95_idx, n - 1)]
    return {"mean": mean, "median": median, "p95": p95}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate RAG events")
    parser.add_argument("events_dir", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--judge-model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    k = config.retrieve_k

    raw_events, skipped = _load_events(args.events_dir)
    events = _dedupe(raw_events)
    with_gt = [e for e in events if e.ground_truth is not None]
    without_gt = [e for e in events if e.ground_truth is None]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": len(events),
        "with_ground_truth": len(with_gt),
        "without_ground_truth": len(without_gt),
        "skipped_corrupted_lines": skipped,
    }

    judge_fn: Callable[[str, str], str] | None = None
    if not args.dry_run and (without_gt or with_gt):
        llm = DeepSeekLLM(
            api_key=config.deepseek_api_key,
            model=args.judge_model,
            base_url=config.deepseek_base_url,
        )

        def _judge(system: str, user: str) -> str:
            return "".join(llm.stream(system, user))

        judge_fn = _judge

    details: list[dict[str, Any]] = []
    supervised_metrics: dict[str, list[float | None]] = defaultdict(list)
    unsupervised_metrics: dict[str, list[float | None]] = defaultdict(list)
    by_collection: dict[str, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for event in events:
        event_metrics: dict[str, Any] = {}
        if event.ground_truth is not None:
            event_metrics.update(_compute_supervised(event, k))
        if judge_fn is not None:
            event_metrics.update(_compute_unsupervised(event, judge_fn))

        for key, value in event_metrics.items():
            if isinstance(value, (int, float)):
                supervised_metrics.setdefault(key, []).append(value)
                by_collection[event.collection].setdefault(key, []).append(value)
            elif value is not None:
                unsupervised_metrics.setdefault(key, []).append(value)
                by_collection[event.collection].setdefault(key, []).append(value)

        details.append({"trace_id": event.trace_id, "collection": event.collection, "metrics": event_metrics})

    report["aggregates"] = {key: _aggregate(vals) for key, vals in supervised_metrics.items()}
    report["aggregates"].update({key: _aggregate(vals) for key, vals in unsupervised_metrics.items()})
    report["by_collection"] = {
        coll: {key: _aggregate(vals) for key, vals in metrics.items()}
        for coll, metrics in by_collection.items()
    }
    report["details"] = details

    output_path = args.output or (Path("data") / f"eval_report_{date.today().isoformat()}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Wrote evaluation report to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

注意：`Callable` 需要在文件顶部 `from collections.abc import Callable`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_batch.py::test_batch_dry_run_reports_counts -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/eval/batch.py tests/eval/test_batch.py
git commit -m "feat(eval): add batch evaluation CLI"
```

---

### Task 8: 批量评估损坏行去重与 judge 集成测试

**Files:**
- Modify: `tests/eval/test_batch.py`

**Interfaces:**
- Consumes: `main` from `batch.py`。

- [ ] **Step 1: Write the failing tests**

追加到 `tests/eval/test_batch.py`：

```python
def test_batch_skips_corrupted_lines_and_counts_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_batch.py -v`

Expected: PASS（如果 Task 7 实现已包含去重和损坏行计数）；否则 FAIL

- [ ] **Step 3: Fix implementation if needed**

确保 `batch.py` 的 `_load_events` 返回 `skipped` 计数，且 `_dedupe` 按 `trace_id` 去重。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_batch.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_batch.py
git commit -m "test(eval): cover batch dedupe and corrupted line handling"
```

---

### Task 9: 全量验证与收尾

**Files:**
- All modified files

- [ ] **Step 1: Run lint**

Run: `make lint`

Expected: 无 error

- [ ] **Step 2: Run typecheck**

Run: `make typecheck`

Expected: 无 error

- [ ] **Step 3: Run tests with coverage**

Run: `make test`

Expected: 全部通过，覆盖率 ≥80%

- [ ] **Step 4: Fix any failures**

常见需要修复的点：

- ruff 行长度超过 100。
- `ty` 对 `Callable` 导入位置、泛型返回类型的报错。
- 测试覆盖率不足时补充单元测试。
- `app.py` 中 `perf_fn(answer_text)` 调用处已存在 `perf_fn()`，需同步更新。

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(eval): complete RAG tracing, metrics, and batch evaluation"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: 每个 spec 章节都有对应任务（事件模型 Task 1、Emitter Task 2、检索指标 Task 3、LLM-as-judge Task 4、pipeline 集成 Task 5、app 集成 Task 6、batch CLI Task 7-8、验证 Task 9）。
- [x] **Placeholder scan**: 无 TBD/TODO/"implement later"；所有步骤包含代码与命令。
- [x] **Type consistency**: `perf_fn` 签名统一为 `Callable[[str], StreamPerf]`；`JSONLEmitter` 接收 `dir_path: Path`；`RAGEvent.metadata` 为 `dict[str, Any]`。
- [x] **DRY**: 测试辅助函数 `_make_event`、`_hit` 在每个测试文件中局部定义，避免跨文件耦合。
- [x] **YAGNI**: 不实现 Prometheus、SQLite、UI 反馈按钮等扩展，仅保留在 spec 未来扩展章节。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-22-rag-metrics.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
