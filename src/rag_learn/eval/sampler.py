"""Sample online RAG events into a CSV for manual labeling."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from rag_learn.eval._csv import CSV_COLUMNS, format_csv_row

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
