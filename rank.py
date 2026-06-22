"""CLI entrypoint for deterministic candidate ranking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import RankingConfig
from src.io import read_candidates_jsonl, write_submission_csv
from src.scoring import rank_candidates
from src.validation import validate_input_path, validate_submission_rows


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Intelligent Candidate Discovery challenge."
    )
    parser.add_argument(
        "--input",
        default="candidates.jsonl",
        type=Path,
        help="Path to the candidates JSONL input file.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        type=Path,
        help="Path where submission CSV will be written.",
    )
    parser.add_argument(
        "--top-k",
        default=100,
        type=int,
        help="Number of ranked candidates to output. Challenge requires 100.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> int:
    """Run the ranking pipeline."""
    args = parse_args()
    configure_logging(args.log_level)

    config = RankingConfig(top_k=args.top_k)
    validate_input_path(args.input)

    LOGGER.info("Loading candidates from %s", args.input)
    candidates = read_candidates_jsonl(args.input)

    LOGGER.info("Ranking candidates with deterministic config: %s", config)
    ranked_rows = rank_candidates(candidates=candidates, config=config)

    validate_submission_rows(ranked_rows, expected_rows=config.top_k)
    write_submission_csv(args.output, ranked_rows)

    LOGGER.info("Wrote submission to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
