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
    run_parser.add_argument("--max-concurrency", type=int, default=3)
    run_parser.add_argument("--rate", type=float, default=20.0)
    run_parser.add_argument("--max-retries", type=int, default=3)
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all rows even if events for the same question exist on disk",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing events")
    eval_parser.add_argument("events_dir", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--judge-model", default=None)
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.add_argument("--max-concurrency", type=int, default=3)
    eval_parser.add_argument("--rate", type=float, default=20.0)
    eval_parser.add_argument("--max-retries", type=int, default=3)

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
            max_concurrency=args.max_concurrency,
            rate_per_minute=args.rate,
            max_retries=args.max_retries,
            resume=not args.no_resume,
        )

    if args.command == "evaluate":
        batch_argv = [
            str(args.events_dir),
            "--output",
            str(args.output),
            "--max-concurrency",
            str(args.max_concurrency),
            "--rate",
            str(args.rate),
            "--max-retries",
            str(args.max_retries),
        ]
        if args.judge_model:
            batch_argv.extend(["--judge-model", args.judge_model])
        if args.dry_run:
            batch_argv.append("--dry-run")
        return batch.main(batch_argv)

    return 1


if __name__ == "__main__":
    sys.exit(main())
