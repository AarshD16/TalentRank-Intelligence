"""CLI entrypoint for deterministic candidate ranking."""

from __future__ import annotations

import argparse
import csv
import heapq
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.config import RankingConfig
from src.io import CandidateProfile, iter_candidates_jsonl, write_submission_csv
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


@dataclass(slots=True)
class PipelineTimings:
    """Simple perf-counter timings for the ranking hot path."""

    read_seconds: float = 0.0
    score_seconds: float = 0.0
    heap_seconds: float = 0.0
    sort_seconds: float = 0.0
    reasoning_seconds: float = 0.0
    validation_seconds: float = 0.0
    write_seconds: float = 0.0
    debug_seconds: float = 0.0
    external_validation_seconds: float = 0.0

    def total(self) -> float:
        return (
            self.read_seconds
            + self.score_seconds
            + self.heap_seconds
            + self.sort_seconds
            + self.reasoning_seconds
            + self.validation_seconds
            + self.write_seconds
            + self.debug_seconds
            + self.external_validation_seconds
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "read": self.read_seconds,
            "score": self.score_seconds,
            "heap": self.heap_seconds,
            "sort": self.sort_seconds,
            "reasoning": self.reasoning_seconds,
            "validation": self.validation_seconds,
            "write": self.write_seconds,
            "debug": self.debug_seconds,
            "external_validation": self.external_validation_seconds,
        }


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
        "--limit",
        default=None,
        type=int,
        help="Process only the first N candidates for quick testing.",
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
    timings = PipelineTimings()
    total_start = time.perf_counter()

    keep_k = DEBUG_TOP_K if args.debug else config.top_k
    LOGGER.info("Streaming candidates from %s", args.candidates)
    top_candidates, processed = _score_streaming(
        args.candidates,
        config=config,
        keep_k=keep_k,
        timings=timings,
        limit=args.limit,
    )
    if len(top_candidates) < config.top_k:
        raise ValueError(f"Need at least {config.top_k} candidates to produce a valid submission.")

    sort_start = time.perf_counter()
    sorted_candidates = _sort_best_first(top_candidates)
    timings.sort_seconds += time.perf_counter() - sort_start

    reasoning_start = time.perf_counter()
    ranked_rows = _build_ranked_rows(sorted_candidates[: config.top_k])
    timings.reasoning_seconds += time.perf_counter() - reasoning_start

    validation_start = time.perf_counter()
    validate_submission_rows(ranked_rows, expected_rows=config.top_k)
    _validate_non_increasing_scores(ranked_rows)
    timings.validation_seconds += time.perf_counter() - validation_start

    write_start = time.perf_counter()
    write_submission_csv(args.out, ranked_rows)
    timings.write_seconds += time.perf_counter() - write_start
    LOGGER.info("Wrote %d ranked rows to %s", len(ranked_rows), args.out)

    if args.debug:
        debug_start = time.perf_counter()
        _write_debug_artifacts(sorted_candidates[:DEBUG_TOP_K], output_dir=Path("outputs"))
        timings.debug_seconds += time.perf_counter() - debug_start

    external_start = time.perf_counter()
    _run_external_validator_if_present(args.out)
    timings.external_validation_seconds += time.perf_counter() - external_start
    _log_timing_summary(timings, processed=processed, total_elapsed=time.perf_counter() - total_start)
    return 0


def _timed_candidates(path: Path, timings: PipelineTimings, limit: int | None = None) -> Iterator[CandidateProfile]:
    for index, candidate in enumerate(iter_candidates_jsonl(path), start=1):
        yield candidate
        if limit is not None and index >= limit:
            break


def _score_streaming(
    path: Path,
    config: RankingConfig,
    keep_k: int,
    timings: PipelineTimings | None = None,
    limit: int | None = None,
) -> tuple[list[CandidateScore], int]:
    heap: list[tuple[float, ReverseString, CandidateScore]] = []
    total = 0
    candidate_iter = _timed_candidates(path, timings or PipelineTimings(), limit=limit)
    while True:
        read_start = time.perf_counter()
        try:
            candidate = next(candidate_iter)
        except StopIteration:
            break
        if timings is not None:
            timings.read_seconds += time.perf_counter() - read_start
        total += 1

        score_start = time.perf_counter()
        breakdown = score_candidate_with_components(candidate, config)
        if timings is not None:
            timings.score_seconds += time.perf_counter() - score_start

        heap_start = time.perf_counter()
        item = CandidateScore(candidate=candidate, breakdown=breakdown)
        heap_item = (breakdown.final_score, ReverseString(breakdown.candidate_id), item)
        if len(heap) < keep_k:
            heapq.heappush(heap, heap_item)
        else:
            heapq.heappushpop(heap, heap_item)
        if timings is not None:
            timings.heap_seconds += time.perf_counter() - heap_start
        if total % 10000 == 0:
            LOGGER.info("Scored %d candidates", total)
    LOGGER.info("Finished scoring %d candidates", total)
    return [item for _score, _reverse_id, item in heap], total


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


def _log_timing_summary(timings: PipelineTimings, processed: int, total_elapsed: float) -> None:
    rate = processed / total_elapsed if total_elapsed > 0 else 0.0
    slowest_stage, slowest_seconds = max(timings.as_dict().items(), key=lambda item: item[1])
    LOGGER.info("Processed %d candidates in %.3fs (%.1f candidates/sec)", processed, total_elapsed, rate)
    LOGGER.info("Slowest stage: %s %.3fs", slowest_stage, slowest_seconds)
    LOGGER.debug("Timing breakdown: %s", timings.as_dict())


def _profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    return profile if isinstance(profile, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
