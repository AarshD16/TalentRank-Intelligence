"""Candidate profile normalization interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NormalizedCandidate:
    """Canonical candidate representation used by downstream ranking stages."""

    candidate_id: str
    raw_profile: dict[str, Any]
    normalized_text: str
    structured_features: dict[str, Any]


def normalize_candidate(profile: dict[str, Any]) -> NormalizedCandidate:
    """Normalize a raw candidate profile into the internal schema."""
    # TODO: Implement robust field mapping for the actual candidates.jsonl schema.
    candidate_id = str(profile.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("Candidate profile is missing candidate_id.")
    LOGGER.debug("Normalizing candidate_id=%s", candidate_id)
    raise NotImplementedError("Candidate normalization is not implemented yet.")
