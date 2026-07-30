"""Cross-encoder reranker implementation backed by sentence-transformers."""

from __future__ import annotations

import logging

from rag_learn.retriever.base import Hit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_seq_length: int | None = None,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device, max_length=max_seq_length)

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Score each (query, hit) pair and return hits sorted by descending score."""
        if not hits:
            return []

        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored = sorted(
            zip(hits, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            Hit(
                text=h.text,
                source_file=h.source_file,
                chunk_index=h.chunk_index,
                score=float(score),
            )
            for h, score in scored
        ]
