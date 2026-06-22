"""Input and output helpers for candidate ranking."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from src.config import RankingConfig
from src.scoring import RankedCandidate


LOGGER = logging.getLogger(__name__)

CandidateProfile = dict[str, Any]


def iter_candidates_jsonl(path: Path) -> Iterator[CandidateProfile]:
    """Yield raw candidate profiles from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                LOGGER.debug("Skipping blank line at %s:%s", path, line_number)
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield payload


def read_candidates_jsonl(path: Path) -> list[CandidateProfile]:
    """Load candidate profiles into memory.

    TODO: Replace list materialization with streaming or chunked processing if profiling
    shows memory pressure at 100,000 rows.
    """
    candidates = list(iter_candidates_jsonl(path))
    LOGGER.info("Loaded %d candidate profiles", len(candidates))
    return candidates


def write_submission_csv(path: Path, rows: Sequence[RankedCandidate]) -> None:
    """Write ranked candidates using the challenge-required CSV schema."""
    columns = RankingConfig().csv_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": f"{row.score:.6f}",
                    "reasoning": row.reasoning,
                }
            )
