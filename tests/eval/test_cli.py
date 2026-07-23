"""Tests for rag_learn.eval.cli."""

from __future__ import annotations

from rag_learn.eval import cli


def test_cli_sample_dispatched(monkeypatch: object) -> None:
    calls: list[tuple[str, object]] = []

    def fake_sample(_dir: object, n: int) -> list[object]:
        calls.append(("sample", n))
        return []

    def fake_write(rows: list[object], path: object) -> None:
        calls.append(("write", path))

    monkeypatch.setattr(cli.sampler, "sample_events", fake_sample)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli.sampler, "write_samples", fake_write)  # type: ignore[attr-defined]

    rc = cli.main(["sample", "data", "--samples-per-collection", "3", "--output", "out.csv"])
    assert rc == 0
    assert ("sample", 3) in calls
    assert any(c[0] == "write" and str(c[1]) == "out.csv" for c in calls)


def test_cli_run_dispatched(monkeypatch: object) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_run(*args: object, **kwargs: object) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli.runner, "run_qa_csv", fake_run)  # type: ignore[attr-defined]

    rc = cli.main(
        [
            "run",
            "qa.csv",
            "--collection",
            "rag_doc",
            "--output-events",
            "data/events.jsonl",
            "--output-report",
            "data/report.json",
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    args = calls[0]
    assert str(args[0]) == "qa.csv"
    assert args[1] == "rag_doc"


def test_cli_evaluate_dispatched(monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_batch(argv: list[str] | None) -> int:
        if argv is None:
            argv_list: list[str] = []
        else:
            argv_list = argv
        calls.append(argv_list)
        return 0

    monkeypatch.setattr(cli.batch, "main", fake_batch)  # type: ignore[attr-defined]

    rc = cli.main(["evaluate", "data", "--output", "report.json", "--dry-run"])
    assert rc == 0
    assert len(calls) == 1
    assert "data" in calls[0]
    assert "--dry-run" in calls[0]
