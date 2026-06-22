"""Configuration objects for the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Deterministic runtime configuration for ranking."""

    top_k: int = 100
    random_seed: int = 20260622
    min_score: float = 0.0
    max_score: float = 100.0
    csv_columns: tuple[str, str, str, str] = (
        "candidate_id",
        "rank",
        "score",
        "reasoning",
    )

    def __post_init__(self) -> None:
        if self.top_k != 100:
            raise ValueError("Challenge submission must contain exactly 100 rows.")
        if self.min_score >= self.max_score:
            raise ValueError("min_score must be lower than max_score.")
