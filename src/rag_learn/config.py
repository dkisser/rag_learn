"""Process-wide configuration loaded once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env at import time so subprocesses (pytest, gr.launch) see the same values.
load_dotenv()

# Search constants — exposed at module level because they are also used by the
# prompt builder (see pipeline.build_prompt) for content truncation.
CHUNK_DISPLAY_CHARS: int = 600


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    deepseek_api_key: str
    llm_model: str
    deepseek_base_url: str
    retrieve_k: int
    chunk_size: int
    chunk_overlap: int
    repo_root: Path
    docs_dir: Path
    data_dir: Path
    chroma_dir: Path
    milvus_path: Path


def _repo_root() -> Path:
    """Walk up from this file until we find pyproject.toml.

    Robust to worktree directories whose names differ from the main
    checkout's directory name.
    """
    p = Path(__file__).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"could not locate repo root from {p} (no pyproject.toml found in any parent)"
    )


def load_config() -> Config:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigError("DEEPSEEK_API_KEY is required. Copy .env.example to .env and set it.")

    repo_root = _repo_root()
    return Config(
        deepseek_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        retrieve_k=int(os.environ.get("RETRIEVE_K", "5")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
        repo_root=repo_root,
        docs_dir=repo_root / "docs" / "rag_doc",
        data_dir=repo_root / "data",
        chroma_dir=repo_root / "data" / "chroma",
        milvus_path=repo_root / "data" / "milvus.db",
    )
