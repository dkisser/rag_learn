"""Factory that picks the right retriever implementation based on config."""

from __future__ import annotations

from pathlib import Path

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.chroma_impl import ChromaRetriever
from rag_learn.retriever.hybrid_impl import HybridRetriever


def build_retriever(
    persist_dir: Path,
    collection_name: str,
    *,
    hybrid_enabled: bool = False,
    hybrid_rrf_k: int = 60,
) -> BaseRetriever:
    """Build a retriever instance.

    When ``hybrid_enabled`` is True, wraps ``ChromaRetriever`` together with a
    BM25 keyword index and fuses the two with Reciprocal Rank Fusion.
    Otherwise falls back to the plain ``ChromaRetriever`` so existing
    behavior is preserved exactly.
    """
    if hybrid_enabled:
        return HybridRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            rrf_k=hybrid_rrf_k,
        )
    return ChromaRetriever(persist_dir=persist_dir, collection_name=collection_name)
