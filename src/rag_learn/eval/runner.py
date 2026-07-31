"""Run a prepared Q&A CSV through the RAG pipeline and evaluate it.

Pacing is delegated to :class:`rag_learn.rate_limit.RateLimiter` so we
don't burst DeepSeek's free-tier RPM. By default a re-run with the same
``--output-events`` directory skips (collection, question) pairs that
already have an emitted event; pass ``resume=False`` to force a full
re-process.
"""

from __future__ import annotations

import csv
import json
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
from rag_learn.rate_limit import RateLimiter
from rag_learn.reranker import build_reranker

logger = logging.getLogger(__name__)


def _load_catalog(config: Any) -> Catalog:
    return build_catalog(
        hybrid_enabled=config.hybrid_enabled,
        hybrid_rrf_k=config.hybrid_rrf_k,
    )


def _make_llm(config: Any) -> DeepSeekLLM:
    return DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )


def _load_existing_keys(events_file: Path) -> set[tuple[str, str]]:
    """Return (collection, question) pairs already emitted to ``events_file``.

    Reads a single JSONL file (the one we'll also write to). If the file
    doesn't exist yet, returns an empty set.
    """
    keys: set[tuple[str, str]] = set()
    if not events_file.is_file():
        return keys
    with open(events_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping corrupted event line %s:%d: %s", events_file, line_number, exc
                )
                continue
            collection = record.get("collection", "")
            question = record.get("question", "")
            if collection and question:
                keys.add((collection, question))
    return keys


def _process_row(
    row: dict[str, str],
    default_collection: str | None,
    catalog: Catalog,
    llm: DeepSeekLLM,
    emitter: JSONLEmitter,
    metadata: dict[str, Any],
    k: int,
    reranker: Any,
    config: Any,
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
            reranker=reranker,
            config=config,
            catalog=catalog,
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
    *,
    max_concurrency: int = 3,
    rate_per_minute: float = 20.0,
    max_retries: int = 3,
    resume: bool = True,
) -> int:
    """Read a Q&A CSV, run each question through RAG, emit events, and evaluate."""
    config = load_config()
    catalog = _load_catalog(config)
    llm = _make_llm(config)
    reranker = build_reranker(config)
    emitter = JSONLEmitter(output_events.parent, file_name=output_events.name)
    metadata = {
        "llm_model": config.llm_model,
        "rerank_enabled": config.rerank_enabled,
        "rerank_model": config.rerank_model if config.rerank_enabled else None,
        "hybrid_enabled": config.hybrid_enabled,
        "hybrid_rrf_k": config.hybrid_rrf_k if config.hybrid_enabled else None,
    }

    limiter = RateLimiter(
        max_concurrency=max_concurrency,
        rate_per_minute=rate_per_minute,
        max_retries=max_retries,
    )

    existing: set[tuple[str, str]] = _load_existing_keys(output_events) if resume else set()

    processed_count = 0
    skipped_count = 0
    with open(qa_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question, collection_slug, _gt = parse_csv_row(row, default_collection)
            if question is None or collection_slug is None:
                continue
            if (collection_slug, question) in existing:
                logger.info("Skipping (already emitted): %s", question)
                skipped_count += 1
                continue
            try:
                limiter.call(
                    _process_row,
                    row,
                    default_collection,
                    catalog,
                    llm,
                    emitter,
                    metadata,
                    config.retrieve_k,
                    reranker,
                    config,
                )
                processed_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("Giving up on row after retries: %r (%s)", row, exc)

    logger.info("Processed %d new rows, skipped %d already-emitted", processed_count, skipped_count)

    return batch_main(
        [
            str(output_events),
            "--output",
            str(output_report),
            "--judge-model",
            judge_model or config.llm_model,
            "--max-concurrency",
            str(max_concurrency),
            "--rate",
            str(rate_per_minute),
            "--max-retries",
            str(max_retries),
        ]
    )
