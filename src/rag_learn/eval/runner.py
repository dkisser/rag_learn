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
    return build_catalog()


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
