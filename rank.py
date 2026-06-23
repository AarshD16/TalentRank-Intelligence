"""CLI entrypoint for deterministic candidate ranking."""

from __future__ import annotations

import argparse
import csv
import heapq
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import RankingConfig
from src.io import iter_candidates_jsonl, write_submission_csv
from src.reasoning import build_reasoning
from src.scoring import RankedCandidate, ScoreBreakdown, score_candidate_with_components
from src.validation import validate_input_path, validate_submission_rows


LOGGER = logging.getLogger(__name__)
DEBUG_TOP_K = 200


class ReverseString:
    """Reverse lexical ordering for heap tie handling."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: "ReverseString") -> bool:
        return self.value > other.value


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Candidate data retained only for current top-k heap entries."""

    candidate: dict[str, Any]
    breakdown: ScoreBreakdown


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Intelligent Candidate Discovery challenge."
    )
    parser.add_argument(
        "--candidates",
        "--input",
        dest="candidates",
        default=Path("candidates.jsonl"),
        type=Path,
        help="Path to the candidates JSONL input file.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="out",
        default=Path("submission.csv"),
        type=Path,
        help="Path where submission CSV will be written.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write optional debug artifacts under outputs/.",
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
    """Run the streaming ranking pipeline."""
    args = parse_args()
    configure_logging(args.log_level)

    config = RankingConfig(top_k=args.top_k)
    validate_input_path(args.candidates)

    keep_k = DEBUG_TOP_K if args.debug else config.top_k
    LOGGER.info("Streaming candidates from %s", args.candidates)
    top_candidates = _score_streaming(args.candidates, config=config, keep_k=keep_k)
    if len(top_candidates) < config.top_k:
        raise ValueError(f"Need at least {config.top_k} candidates to produce a valid submission.")

    sorted_candidates = _sort_best_first(top_candidates)
    ranked_rows = _build_ranked_rows(sorted_candidates[: config.top_k])

    validate_submission_rows(ranked_rows, expected_rows=config.top_k)
    _validate_non_increasing_scores(ranked_rows)
    write_submission_csv(args.out, ranked_rows)
    LOGGER.info("Wrote %d ranked rows to %s", len(ranked_rows), args.out)

    if args.debug:
        _write_debug_artifacts(sorted_candidates[:DEBUG_TOP_K], output_dir=Path("outputs"))

    _run_external_validator_if_present(args.out)
    return 0


def _score_streaming(path: Path, config: RankingConfig, keep_k: int) -> list[CandidateScore]:
    heap: list[tuple[float, ReverseString, CandidateScore]] = []
    total = 0
    for candidate in iter_candidates_jsonl(path):
        total += 1
        breakdown = score_candidate_with_components(candidate, config)
        item = CandidateScore(candidate=candidate, breakdown=breakdown)
        heap_item = (breakdown.final_score, ReverseString(breakdown.candidate_id), item)
        if len(heap) < keep_k:
            heapq.heappush(heap, heap_item)
        else:
            heapq.heappushpop(heap, heap_item)
        if total % 10000 == 0:
            LOGGER.info("Scored %d candidates", total)
    LOGGER.info("Finished scoring %d candidates", total)
    return [item for _score, _reverse_id, item in heap]


def _sort_best_first(items: list[CandidateScore]) -> list[CandidateScore]:
    return sorted(
        items,
        key=lambda item: (-item.breakdown.final_score, item.breakdown.candidate_id),
    )


def _build_ranked_rows(items: list[CandidateScore]) -> list[RankedCandidate]:
    rows: list[RankedCandidate] = []
    for rank, item in enumerate(items, start=1):
        breakdown = item.breakdown
        reasoning = build_reasoning(
            evidence=breakdown.evidence,
            penalties=breakdown.penalties,
            rank=rank,
            component_scores=breakdown.component_scores,
            candidate=item.candidate,
        )
        rows.append(
            RankedCandidate(
                candidate_id=breakdown.candidate_id,
                rank=rank,
                score=breakdown.final_score,
                reasoning=reasoning,
                component_scores=breakdown.component_scores,
            )
        )
    return rows


def _validate_non_increasing_scores(rows: list[RankedCandidate]) -> None:
    previous = float("inf")
    for row in rows:
        if row.score > previous:
            raise ValueError(f"Score increases at rank {row.rank}: {row.score} > {previous}")
        previous = row.score


def _write_debug_artifacts(items: list[CandidateScore], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_rows = _build_ranked_rows(items)
    _write_top_debug(output_dir / "top_200_debug.csv", ranked_rows, items)
    _write_component_scores(output_dir / "component_scores.csv", ranked_rows)
    _write_penalty_audit(output_dir / "penalty_audit.csv", ranked_rows, items)
    LOGGER.info("Wrote debug artifacts to %s", output_dir)


def _write_top_debug(path: Path, rows: list[RankedCandidate], items: list[CandidateScore]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "debug_rank",
                "score",
                "current_title",
                "years_of_experience",
                "reasoning",
            ],
        )
        writer.writeheader()
        for row, item in zip(rows, items, strict=True):
            profile = _profile(item.candidate)
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "debug_rank": row.rank,
                    "score": f"{row.score:.6f}",
                    "current_title": profile.get("current_title", ""),
                    "years_of_experience": profile.get("years_of_experience", ""),
                    "reasoning": row.reasoning,
                }
            )


def _write_component_scores(path: Path, rows: list[RankedCandidate]) -> None:
    component_names = sorted({name for row in rows for name in row.component_scores})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "debug_rank", "score", *component_names],
        )
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "candidate_id": row.candidate_id,
                "debug_rank": row.rank,
                "score": f"{row.score:.6f}",
            }
            payload.update({name: f"{row.component_scores.get(name, 0.0):.6f}" for name in component_names})
            writer.writerow(payload)


def _write_penalty_audit(path: Path, rows: list[RankedCandidate], items: list[CandidateScore]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "debug_rank",
                "penalty_score",
                "hard_flags",
                "soft_flags",
                "penalty_reasons",
            ],
        )
        writer.writeheader()
        for row, item in zip(rows, items, strict=True):
            penalties = item.breakdown.penalties
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "debug_rank": row.rank,
                    "penalty_score": f"{penalties.penalty_score:.4f}",
                    "hard_flags": " | ".join(penalties.hard_flags),
                    "soft_flags": " | ".join(penalties.soft_flags),
                    "penalty_reasons": " | ".join(penalties.penalty_reasons),
                }
            )


def _run_external_validator_if_present(output_path: Path) -> None:
    validator = Path("validate_submission.py")
    if not validator.exists():
        LOGGER.info("No validate_submission.py found; skipped external validation")
        return
    LOGGER.info("Running external validator: %s", validator)
    subprocess.run([sys.executable, str(validator), str(output_path)], check=True)


def _profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    return profile if isinstance(profile, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
