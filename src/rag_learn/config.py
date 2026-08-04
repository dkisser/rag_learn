"""Process-wide configuration loaded once from environment variables."""

from __future__ import annotations

import math
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
    rerank_enabled: bool
    rerank_model: str
    rerank_factor: int
    rerank_k: int | None
    rerank_batch_size: int
    rerank_device: str | None
    hybrid_enabled: bool
    hybrid_rrf_k: int
    intent_enabled: bool
    intent_timeout_s: float
    decompose_enabled: bool
    decompose_timeout_s: float
    decompose_max: int
    catalog_sub_k: int
    catalog_recall_k: int
    chroma_max_distance: float | None = None
    rerank_min_score: float | None = None


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


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_optional_float(value: str | None, *, name: str, default: float | None) -> float | None:
    """解析可选的有限浮点数，留空表示禁用该配置。"""
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{name} must be a finite number")
    return parsed


def load_config() -> Config:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigError("DEEPSEEK_API_KEY is required. Copy .env.example to .env and set it.")

    repo_root = _repo_root()
    rerank_k_raw = os.environ.get("RERANK_K")
    rerank_device = os.environ.get("RERANK_DEVICE", "auto")
    chroma_max_distance = _parse_optional_float(
        os.environ.get("CHROMA_MAX_DISTANCE"),
        name="CHROMA_MAX_DISTANCE",
        default=1.0,
    )
    if chroma_max_distance is not None and chroma_max_distance < 0:
        raise ConfigError("CHROMA_MAX_DISTANCE must be non-negative")
    rerank_min_score = _parse_optional_float(
        os.environ.get("RERANK_MIN_SCORE"),
        name="RERANK_MIN_SCORE",
        default=0.001,
    )
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
        rerank_enabled=_parse_bool(os.environ.get("RERANK_ENABLED"), False),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base"),
        rerank_factor=int(os.environ.get("RERANK_FACTOR", "4")),
        rerank_k=int(rerank_k_raw) if rerank_k_raw is not None else None,
        rerank_batch_size=int(os.environ.get("RERANK_BATCH_SIZE", "8")),
        rerank_device=rerank_device if rerank_device.strip() else None,
        hybrid_enabled=_parse_bool(os.environ.get("HYBRID_ENABLED"), False),
        hybrid_rrf_k=int(os.environ.get("HYBRID_RRF_K", "60")),
        intent_enabled=_parse_bool(os.environ.get("INTENT_ENABLED"), False),
        intent_timeout_s=float(os.environ.get("INTENT_TIMEOUT_S", "8.0")),
        decompose_enabled=_parse_bool(os.environ.get("DECOMPOSE_ENABLED"), False),
        decompose_timeout_s=float(os.environ.get("DECOMPOSE_TIMEOUT_S", "15.0")),
        decompose_max=int(os.environ.get("DECOMPOSE_MAX", "8")),
        catalog_sub_k=int(os.environ.get("CATALOG_SUB_K", "8")),
        catalog_recall_k=int(os.environ.get("CATALOG_RECALL_K", "20")),
        chroma_max_distance=chroma_max_distance,
        rerank_min_score=rerank_min_score,
    )
