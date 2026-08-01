from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag_learn.config import Config
from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_docs"


@pytest.fixture
def make_routing_config() -> Callable[..., Config]:
    """Factory for a Config with the intent/decompose routing knobs set.

    Keyword overrides are applied last, so a test can tweak a single field
    (e.g. ``catalog_recall_k=3``) without restating the whole dataclass.
    """

    def _make(**overrides: Any) -> Config:
        base = Path(__file__).parent.parent / "src"
        defaults: dict[str, Any] = {
            "deepseek_api_key": "k",
            "llm_model": "m",
            "deepseek_base_url": "u",
            "retrieve_k": 2,
            "chunk_size": 800,
            "chunk_overlap": 50,
            "repo_root": base,
            "docs_dir": base / "docs" / "rag_doc",
            "data_dir": base / "data",
            "chroma_dir": base / "data" / "chroma",
            "milvus_path": base / "data" / "milvus.db",
            "rerank_enabled": False,
            "rerank_model": "BAAI/bge-reranker-base",
            "rerank_factor": 4,
            "rerank_k": None,
            "rerank_batch_size": 8,
            "rerank_device": None,
            "hybrid_enabled": False,
            "hybrid_rrf_k": 60,
            "intent_enabled": True,
            "intent_timeout_s": 2.0,
            "decompose_enabled": True,
            "decompose_timeout_s": 2.0,
            "decompose_max": 8,
            "catalog_sub_k": 10,
            "catalog_recall_k": 10,
        }
        return Config(**{**defaults, **overrides})

    return _make


@pytest.fixture
def sample_hits(fixtures_dir: Path) -> list[Hit]:
    chunks = load_documents(fixtures_dir)
    return [
        Hit(
            text=c.text,
            source_file=c.source_file,
            chunk_index=c.chunk_index,
            score=0.0,  # not used when we feed manually
        )
        for c in chunks
    ]
