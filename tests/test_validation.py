from __future__ import annotations

import pytest

from src.scoring import RankedCandidate
from src.validation import validate_submission_rows


def _valid_rows(count: int = 100) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            candidate_id=f"candidate_{index:03d}",
            rank=index,
            score=100.0 - index / 1000.0,
            reasoning="Skeleton validation fixture.",
        )
        for index in range(1, count + 1)
    ]


def test_validate_submission_rows_accepts_valid_contract() -> None:
    validate_submission_rows(_valid_rows())


def test_validate_submission_rows_requires_100_rows() -> None:
    with pytest.raises(ValueError, match="exactly 100"):
        validate_submission_rows(_valid_rows(99))


def test_validate_submission_rows_rejects_duplicate_ids() -> None:
    rows = _valid_rows()
    rows[1] = RankedCandidate(
        candidate_id=rows[0].candidate_id,
        rank=2,
        score=99.0,
        reasoning="Duplicate fixture.",
    )

    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        validate_submission_rows(rows)
