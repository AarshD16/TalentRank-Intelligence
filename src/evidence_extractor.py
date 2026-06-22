"""Evidence extraction interfaces for explainable ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.normalization import NormalizedCandidate


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Interpretable evidence extracted from a normalized candidate profile."""

    candidate_id: str
    positive_signals: tuple[str, ...]
    risk_signals: tuple[str, ...]
    feature_values: dict[str, float]


def extract_evidence(candidate: NormalizedCandidate) -> EvidenceBundle:
    """Extract deterministic evidence used by scoring and reasoning."""
    # TODO: Add local, deterministic parsing for skills, seniority, domains, and impact.
    LOGGER.debug("Extracting evidence for candidate_id=%s", candidate.candidate_id)
    raise NotImplementedError("Evidence extraction is not implemented yet.")
