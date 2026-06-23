"""Validate a Redrob submission CSV contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission.csv format.")
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional candidates.jsonl path to validate candidate_id membership.",
    )
    return parser.parse_args()


def validate_submission(path: Path, candidates_path: Path | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {path}")
    candidate_ids = _load_candidate_ids(candidates_path) if candidates_path else None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"Columns must be exactly {REQUIRED_COLUMNS}, got {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != 100:
        raise ValueError(f"Submission must contain exactly 100 rows, got {len(rows)}")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    previous_score = float("inf")
    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"].strip()
        if not candidate_id:
            raise ValueError(f"Empty candidate_id at CSV row {row_number}")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        if candidate_ids is not None and candidate_id not in candidate_ids:
            raise ValueError(f"candidate_id not present in candidates file: {candidate_id}")
        seen_ids.add(candidate_id)

        try:
            rank = int(row["rank"])
        except ValueError as exc:
            raise ValueError(f"Invalid rank at CSV row {row_number}: {row['rank']}") from exc
        if not 1 <= rank <= 100:
            raise ValueError(f"Rank out of range at CSV row {row_number}: {rank}")
        if rank in seen_ranks:
            raise ValueError(f"Duplicate rank: {rank}")
        seen_ranks.add(rank)

        try:
            score = float(row["score"])
        except ValueError as exc:
            raise ValueError(f"Invalid score at CSV row {row_number}: {row['score']}") from exc
        if score > previous_score:
            raise ValueError(f"Scores must be non-increasing; row {row_number} has {score} after {previous_score}")
        previous_score = score

        if not row["reasoning"].strip():
            raise ValueError(f"Empty reasoning at CSV row {row_number}")

    if seen_ranks != set(range(1, 101)):
        raise ValueError("Ranks must contain each integer 1 through 100 exactly once")


def _load_candidate_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"candidate_id"' not in line:
                continue
            # Avoid importing json in the hot rank path; validator can stay simple.
            import json

            payload = json.loads(line)
            ids.add(str(payload.get("candidate_id", "")).strip())
    return ids


def main() -> int:
    args = parse_args()
    validate_submission(args.submission, args.candidates)
    print(f"PASS {args.submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
