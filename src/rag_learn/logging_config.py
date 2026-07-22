"""Process-wide logging configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

DEFAULT_LOG_LEVEL_NAME: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR_NAME: str = "logs"
LOG_FILE_NAME: str = "app.log"


def _repo_root(start_path: Path | None = None) -> Path:
    """Walk up from ``start_path`` until we find pyproject.toml.

    Robust to worktree directories whose names differ from the main
    checkout's directory name.
    """
    p = start_path or Path(__file__).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"could not locate repo root from {p} (no pyproject.toml found in any parent)"
    )


def get_log_level(level_name: str | None = None) -> int:
    """Resolve a log-level name to a ``logging`` level integer.

    Reads from the ``LOG_LEVEL`` environment variable when no explicit name is
    supplied, and falls back to ``INFO`` when the configured name is unknown.
    """
    name = (level_name or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL_NAME)).upper()
    level = getattr(logging, name, None)
    if not isinstance(level, int):
        return logging.INFO
    return level


def create_handlers(log_dir: Path) -> list[logging.Handler]:
    """Create console and file handlers, ensuring the log directory exists."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    return [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]


def setup_logging() -> None:
    """Configure the root logger with console and file handlers."""
    repo_root = _repo_root()
    logging.basicConfig(
        level=get_log_level(),
        format=LOG_FORMAT,
        handlers=create_handlers(repo_root / LOG_DIR_NAME),
    )
