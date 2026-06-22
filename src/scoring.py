"""Scoring and ranking interfaces."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import RankingConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Final row written to submission.csv."""

    candidate_id: str
    rank: int
    score: float
    reasoning: str


def score_candidate(profile: dict[str, Any], config: RankingConfig) -> float:
    """Return the deterministic score for one candidate."""
    # TODO: Wire normalization, evidence extraction, feature scoring, and penalties.
    LOGGER.debug("Scoring candidate with config=%s", config)
    raise NotImplementedError("Candidate scoring is not implemented yet.")


def rank_candidates(
    candidates: Sequence[dict[str, Any]],
    config: RankingConfig,
) -> list[RankedCandidate]:
    """Rank candidates and return exactly config.top_k rows."""
    # TODO: Implement stable ordering, deterministic tie-breaks, and top-k selection.
    LOGGER.info("Ranking %d candidates", len(candidates))
    raise NotImplementedError("Candidate ranking is not implemented yet.")
