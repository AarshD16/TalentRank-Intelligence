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
    scoring_weights: dict[str, float] | None = None
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
        weights = self.resolved_scoring_weights()
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight:.4f}.")

    def resolved_scoring_weights(self) -> dict[str, float]:
        """Return component weights for final 0-1 scoring."""
        if self.scoring_weights is not None:
            return dict(self.scoring_weights)
        return {
            "role_fit_score": 0.16,
            "production_ml_score": 0.17,
            "retrieval_ranking_score": 0.18,
            "evaluation_score": 0.10,
            "experience_score": 0.10,
            "company_context_score": 0.08,
            "skill_quality_score": 0.08,
            "redrob_availability_score": 0.06,
            "trust_consistency_score": 0.04,
            "logistics_score": 0.03,
        }
