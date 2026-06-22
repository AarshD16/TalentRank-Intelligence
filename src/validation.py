"""Validation helpers for inputs and challenge submission output."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.scoring import RankedCandidate


def validate_input_path(path: Path) -> None:
    """Validate that the input file exists and is readable."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path must be a file: {path}")


def validate_submission_rows(rows: Sequence[RankedCandidate], expected_rows: int = 100) -> None:
    """Validate final rows against the challenge output contract."""
    if len(rows) != expected_rows:
        raise ValueError(f"Submission must contain exactly {expected_rows} rows.")

    seen_ids: set[str] = set()
    expected_rank = 1
    for row in rows:
        if not row.candidate_id:
            raise ValueError("candidate_id must be non-empty.")
        if row.candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id in submission: {row.candidate_id}")
        if row.rank != expected_rank:
            raise ValueError(f"Expected rank {expected_rank}, got {row.rank}.")
        if not 0.0 <= row.score <= 100.0:
            raise ValueError(f"Score out of range for {row.candidate_id}: {row.score}")
        if not row.reasoning:
            raise ValueError(f"Reasoning must be non-empty for {row.candidate_id}.")
        seen_ids.add(row.candidate_id)
        expected_rank += 1
