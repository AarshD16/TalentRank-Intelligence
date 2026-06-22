"""Human-readable reasoning generation for ranked candidates."""

from __future__ import annotations

import logging

from src.evidence_extractor import EvidenceBundle
from src.penalties import PenaltyResult


LOGGER = logging.getLogger(__name__)


def build_reasoning(evidence: EvidenceBundle, penalties: PenaltyResult) -> str:
    """Build a concise deterministic explanation for a ranked candidate."""
    # TODO: Compose short evidence-backed reasoning without hosted LLM calls.
    LOGGER.debug("Building reasoning for candidate_id=%s", evidence.candidate_id)
    raise NotImplementedError("Reasoning generation is not implemented yet.")
