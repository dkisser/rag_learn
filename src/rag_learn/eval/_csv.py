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
