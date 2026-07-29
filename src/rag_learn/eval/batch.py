"""Batch evaluation CLI for RAG events stored in JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from rag_learn.config import Config, load_config
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
from rag_learn.eval.tracing import GroundTruth, RAGEvent, event_from_dict
from rag_learn.llm import DeepSeekLLM
from rag_learn.rate_limit import RateLimiter, is_rate_limit_error

logger = logging.getLogger(__name__)


def _load_events(events_path: Path) -> tuple[list[RAGEvent], int]:
    """Load events from either a single .jsonl file or a directory of them.

    When ``events_path`` is a file, read it directly. When it's a directory,
    glob ``rag_events_*.jsonl`` (the daily-rotation scheme used by the
    Gradio app and older runs).
    """
    if events_path.is_file():
        files = [events_path]
    else:
        files = sorted(events_path.glob("rag_events_*.jsonl"))

    events: list[RAGEvent] = []
    skipped = 0
    for path in files:
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
    metrics: dict[str, Any] = {}
    if source_files and all(isinstance(f, str) and f for f in source_files):
        metrics[f"retrieval_recall@{k}"] = retrieval_recall_at_k(event.hits, source_files, k)
        metrics[f"retrieval_precision@{k}"] = retrieval_precision_at_k(event.hits, source_files, k)
        metrics["retrieval_mrr"] = retrieval_mrr(event.hits, source_files)
        metrics[f"retrieval_ndcg@{k}"] = retrieval_ndcg_at_k(event.hits, source_files, k)
    else:
        logger.warning(
            "Skipping retrieval metrics for trace %s: no source_files "
            "(question relies on model knowledge)",
            event.trace_id,
        )
    if event.ground_truth.answer:
        metrics["answer_f1"] = answer_f1(event.answer, event.ground_truth.answer)
    return metrics


def _make_judge_fn(config: Config, model: str) -> Callable[[str, str], str]:
    """Build the default LLM-based judge. Tests monkeypatch this to inject failures."""
    llm = DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=model,
        base_url=config.deepseek_base_url,
    )

    def _judge(system: str, user: str) -> str:
        return "".join(llm.stream(system, user))

    return _judge


def _safe_judge(
    metric_fn: Callable[[RAGEvent, Callable[[str, str], str]], float | None],
    event: RAGEvent,
    judge_fn: Callable[[str, str], str],
    metric_name: str,
) -> float | None:
    try:
        return metric_fn(event, judge_fn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge failed for %s (trace %s): %s", metric_name, event.trace_id, exc)
        return None


_UNSUPERVISED_METRICS: tuple[
    tuple[str, Callable[[RAGEvent, Callable[[str, str], str]], float | None]], ...
] = (
    ("context_relevance", context_relevance),
    ("faithfulness", faithfulness),
    ("answer_relevance", answer_relevance),
    ("overall_usefulness", overall_usefulness),
)


def _compute_unsupervised(
    event: RAGEvent,
    judge_fn: Callable[[str, str], str],
    limiter: RateLimiter,
) -> dict[str, Any]:
    """Run unsupervised judge metrics under a shared ``RateLimiter``.

    All metrics are dispatched in parallel via a ThreadPoolExecutor sized
    by the limiter's concurrency cap. 429 retries and rate pacing are
    handled inside ``limiter.call``; if retries are exhausted we record
    ``None`` for that metric instead of propagating.
    """
    metrics_to_run = list(_UNSUPERVISED_METRICS)
    if event.ground_truth is not None and event.ground_truth.answer:
        metrics_to_run = [*metrics_to_run, ("answer_llm_correctness", answer_llm_correctness)]

    results: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=limiter.max_concurrency) as ex:
        futures = {
            name: ex.submit(limiter.call, fn, event, judge_fn) for name, fn in metrics_to_run
        }
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                if is_rate_limit_error(exc):
                    logger.warning(
                        "Judge %s gave up after retries for trace %s", name, event.trace_id
                    )
                else:
                    logger.warning("Judge %s failed for trace %s: %s", name, event.trace_id, exc)
                results[name] = None
    return dict(results)


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


def _ground_truth_to_dict(gt: GroundTruth | None) -> dict[str, Any] | None:
    if gt is None:
        return None
    return {
        "answer": gt.answer,
        "source_files": list(gt.source_files),
        "chunk_ids": list(gt.chunk_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate RAG events")
    parser.add_argument("events_dir", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--judge-model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max in-flight judge calls")
    parser.add_argument(
        "--rate", type=float, default=20.0, help="Judge requests per minute (per limiter)"
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per judge call on 429")
    args = parser.parse_args(argv)

    if args.dry_run:
        k = int(os.environ.get("RETRIEVE_K", "5"))
        config: Config | None = None
    else:
        config = load_config()
        k = config.retrieve_k

    raw_events, skipped = _load_events(args.events_dir)
    events = _dedupe(raw_events)
    with_gt = [e for e in events if e.ground_truth is not None]
    without_gt = [e for e in events if e.ground_truth is None]

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_events": len(events),
        "with_ground_truth": len(with_gt),
        "without_ground_truth": len(without_gt),
        "skipped_corrupted_lines": skipped,
    }

    judge_fn: Callable[[str, str], str] | None = None
    limiter: RateLimiter | None = None
    if not args.dry_run and config is not None and (without_gt or with_gt):
        judge_fn = _make_judge_fn(config, args.judge_model)
        limiter = RateLimiter(
            max_concurrency=args.max_concurrency,
            rate_per_minute=args.rate,
            max_retries=args.max_retries,
        )

    details: list[dict[str, Any]] = []
    metrics_by_key: dict[str, list[float | None]] = defaultdict(list)
    by_collection: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))

    for event in events:
        event_metrics: dict[str, Any] = {}
        if event.ground_truth is not None:
            event_metrics.update(_compute_supervised(event, k))
        if judge_fn is not None and limiter is not None:
            event_metrics.update(_compute_unsupervised(event, judge_fn, limiter))

        for key, value in event_metrics.items():
            if isinstance(value, (int, float)):
                metrics_by_key.setdefault(key, []).append(float(value))
                by_collection[event.collection].setdefault(key, []).append(float(value))

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

    report["aggregates"] = {key: _aggregate(vals) for key, vals in metrics_by_key.items()}
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
