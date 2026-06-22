"""Penalty calculation interfaces for ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.evidence_extractor import EvidenceBundle


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PenaltyResult:
    """Penalty details applied to a candidate score."""

    total_penalty: float
    reasons: tuple[str, ...]


def calculate_penalties(evidence: EvidenceBundle) -> PenaltyResult:
    """Calculate deterministic penalties from extracted evidence."""
    # TODO: Penalize missing core evidence, weak recency, or mismatched requirements.
    LOGGER.debug("Calculating penalties for candidate_id=%s", evidence.candidate_id)
    raise NotImplementedError("Penalty calculation is not implemented yet.")
