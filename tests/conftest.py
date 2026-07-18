from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_docs"


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

# milvus-lite 2.6+ on macOS ARM SIGSEGVs inside load_collection
# (see CLAUDE.md “milvus-lite deadlocks on the full 25-doc corpus”).
darwin_milvus_reload_skip = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="milvus-lite load_collection SIGSEGVs on macOS ARM (CLAUDE.md)",
)
