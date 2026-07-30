"""Factory for building a reranker from configuration."""

from __future__ import annotations

import logging

from rag_learn.config import Config
from rag_learn.reranker.base import Reranker
from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

logger = logging.getLogger(__name__)


def build_reranker(config: Config) -> Reranker | None:
    """Build a reranker from config, or return None if disabled or unavailable."""
    if not config.rerank_enabled:
        return None
    try:
        return CrossEncoderReranker(
            model_name=config.rerank_model,
            device=config.rerank_device,
            batch_size=config.rerank_batch_size,
        )
    except Exception as exc:  # noqa: BLE001 — load failure is fail-open
        logger.warning(
            "Reranker model %r failed to load (%s); falling back to vector ranking.",
            config.rerank_model,
            exc,
        )
        return None
