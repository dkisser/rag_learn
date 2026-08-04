"""Cross-encoder reranker implementation backed by sentence-transformers."""

from __future__ import annotations

import logging
import math

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
        min_score: float | None = None,
    ) -> None:
        if min_score is not None and not math.isfinite(min_score):
            raise ValueError("min_score must be a finite number")
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._batch_size = batch_size
        self._min_score = min_score
        self._model = CrossEncoder(model_name, device=device, max_length=max_seq_length)

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """为每个问题-命中对打分，按降序返回过滤后的命中。"""
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
        ranked = [
            Hit(
                text=h.text,
                source_file=h.source_file,
                chunk_index=h.chunk_index,
                score=float(score),
            )
            for h, score in scored
        ]
        if self._min_score is None:
            return ranked
        return [hit for hit in ranked if hit.score >= self._min_score]
