"""Performance timing data class shared across the package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamPerf:
    retrieve_ms: float
    first_token_ms: float
    total_ms: float
    finished_at: str  # HH:MM:SS.mmm
