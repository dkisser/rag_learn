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
    question, collection, gt = parse_csv_row({"question": "q"}, default_collection="rag_doc")
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
    question, collection, gt = parse_csv_row(
        {
            "question": "q",
            "answer": "a",
            "source_files": "a.md; b.md ",
            "chunk_ids": "a.md#0; b.md#1 ",
            "collection": "rag_doc",
        }
    )
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
