"""CLI entrypoint for deterministic candidate ranking."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.config import RankingConfig
from src.io import CandidateProfile, iter_candidates_jsonl, write_submission_csv
from src.reasoning import build_reasoning, build_rolespec_reasoning
from src.rolespec import RoleSpec, load_rolespec, rolespec_from_mapping, save_rolespec
from src.scoring import RankedCandidate, ScoreBreakdown, score_candidate_with_components
from src.validation import validate_input_path, validate_submission_rows
from scripts.generate_rolespec import _emit_rolespec_diagnostics, _generate_with_ollama, _harden_rolespec, _load_dotenv, _read_jd


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
    _load_dotenv(Path(".env"))
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
        "--role",
        default=Path("configs/rolespec_redrob_ai_engineer.yaml"),
        type=Path,
        help="Path to RoleSpec YAML generated from the job description.",
    )
    parser.add_argument(
        "--jd",
        default=None,
        type=Path,
        help="Optional job description path. When provided, rank.py first calls the LLM to compile this JD into --role.",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "draft-json"),
        default="ollama",
        help="RoleSpec compiler provider used only when --jd is provided.",
    )
    parser.add_argument(
        "--draft-json",
        type=Path,
        help="Use a pre-generated RoleSpec JSON draft when --jd is provided with --provider draft-json.",
    )
    parser.add_argument("--ollama-base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    parser.add_argument(
        "--ollama-api-key",
        default=os.environ.get("OLLAMA_API_KEY", ""),
        help="Bearer token for hosted/Ollama-compatible cloud endpoints.",
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

    if args.jd is not None:
        _compile_rolespec_from_jd(args)
    else:
        LOGGER.info(
            "No --jd provided; using existing saved RoleSpec from %s. "
            "Pass --jd to call the LLM and compile a fresh scoring policy first.",
            args.role,
        )
    run_ranking(
        candidates_path=args.candidates,
        role_path=args.role,
        out_path=args.out,
        debug=args.debug,
        shortlist_size=args.top_k,
        debug_dir=Path("outputs"),
        limit=args.limit,
        external_validate=True,
    )
    return 0


def run_ranking(
    candidates_path: Path,
    role_path: Path,
    out_path: Path,
    debug: bool = False,
    shortlist_size: int = 100,
    debug_dir: Path | None = None,
    limit: int | None = None,
    external_validate: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run deterministic ranking and write CSV outputs.

    This is the callable integration point used by the FastAPI backend. It
    reuses the same scoring, sorting, reasoning, validation, and CSV writers as
    the CLI entrypoint.
    """
    if shortlist_size <= 0 or shortlist_size > 100:
        raise ValueError("shortlist_size must be between 1 and 100")

    config = RankingConfig()
    validate_input_path(candidates_path)
    LOGGER.info("Loading structured scoring policy from %s", role_path)
    rolespec = load_rolespec(role_path)
    timings = PipelineTimings()
    total_start = time.perf_counter()

    if progress_callback:
        progress_callback(30, "Candidate stream started", "Streaming candidates from uploaded file")

    keep_k = max(shortlist_size, DEBUG_TOP_K if debug else shortlist_size)
    LOGGER.info("Streaming candidates from %s", candidates_path)
    top_candidates, processed = _score_streaming(
        candidates_path,
        config=config,
        rolespec=rolespec,
        keep_k=keep_k,
        timings=timings,
        limit=limit,
    )
    if len(top_candidates) < shortlist_size:
        raise ValueError(f"Need at least {shortlist_size} candidates to produce a valid submission.")

    if progress_callback:
        progress_callback(75, "Generating explanations", "Sorting candidates and building reasoning")

    sort_start = time.perf_counter()
    sorted_candidates = _sort_best_first(top_candidates)
    timings.sort_seconds += time.perf_counter() - sort_start

    reasoning_start = time.perf_counter()
    ranked_rows = _build_ranked_rows(sorted_candidates[:shortlist_size], rolespec=rolespec)
    timings.reasoning_seconds += time.perf_counter() - reasoning_start

    validation_start = time.perf_counter()
    validate_submission_rows(ranked_rows, expected_rows=shortlist_size)
    _validate_non_increasing_scores(ranked_rows)
    timings.validation_seconds += time.perf_counter() - validation_start

    if progress_callback:
        progress_callback(90, "CSV/export built", "Writing submission and optional debug artifacts")

    write_start = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(out_path, ranked_rows)
    timings.write_seconds += time.perf_counter() - write_start
    LOGGER.info("Wrote %d ranked rows to %s", len(ranked_rows), out_path)

    if debug:
        debug_start = time.perf_counter()
        _write_debug_artifacts(sorted_candidates[:DEBUG_TOP_K], output_dir=debug_dir or Path("outputs"))
        timings.debug_seconds += time.perf_counter() - debug_start

    if external_validate:
        external_start = time.perf_counter()
        _run_external_validator_if_present(out_path)
        timings.external_validation_seconds += time.perf_counter() - external_start

    elapsed = time.perf_counter() - total_start
    _log_timing_summary(timings, processed=processed, total_elapsed=elapsed)
    return {
        "processed": processed,
        "rows": len(ranked_rows),
        "elapsed_seconds": elapsed,
        "timings": timings.as_dict(),
        "submission_path": str(out_path),
        "debug_dir": str(debug_dir or Path("outputs")) if debug else "",
    }


def _compile_rolespec_from_jd(args: argparse.Namespace) -> None:
    LOGGER.info("Compiling job description into structured RoleSpec policy")
    LOGGER.info("Reading job description from %s", args.jd)
    jd_text = _read_jd(args.jd)
    LOGGER.info("Loaded job description (%d characters)", len(jd_text))
    if args.provider == "draft-json":
        if args.draft_json is None:
            raise ValueError("--draft-json is required when --provider draft-json is used")
        LOGGER.info("Using draft JSON provider; no LLM call will be made")
        payload = json.loads(args.draft_json.read_text(encoding="utf-8"))
    else:
        LOGGER.info(
            "Calling LLM provider=ollama endpoint=%s model=%s auth=%s",
            args.ollama_base_url.rstrip("/"),
            args.model,
            "enabled" if args.ollama_api_key else "not set",
        )
        payload = _generate_with_ollama(
            jd_text=jd_text,
            base_url=args.ollama_base_url,
            model=args.model,
            api_key=args.ollama_api_key,
        )
    LOGGER.info("Validating LLM-generated RoleSpec before candidate scoring")
    rolespec = rolespec_from_mapping(payload)
    rolespec = _harden_rolespec(rolespec)
    save_rolespec(args.role, rolespec)
    _emit_rolespec_diagnostics(rolespec)
    LOGGER.info(
        "Saved structured scoring policy to %s; candidate ranking will now run offline against this policy",
        args.role,
    )


def _timed_candidates(path: Path, timings: PipelineTimings, limit: int | None = None) -> Iterator[CandidateProfile]:
    for index, candidate in enumerate(iter_candidates_jsonl(path), start=1):
        yield candidate
        if limit is not None and index >= limit:
            break


def _score_streaming(
    path: Path,
    config: RankingConfig,
    keep_k: int,
    rolespec: RoleSpec | None = None,
    timings: PipelineTimings | None = None,
    limit: int | None = None,
) -> tuple[list[CandidateScore], int]:
    heap: list[tuple[float, ReverseString, CandidateScore]] = []
    forced: dict[str, CandidateScore] = {}
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
        breakdown = score_candidate_with_components(candidate, config, rolespec=rolespec)
        if timings is not None:
            timings.score_seconds += time.perf_counter() - score_start

        heap_start = time.perf_counter()
        item = CandidateScore(candidate=candidate, breakdown=breakdown)
        if "high_recall_safety_gate" in breakdown.score_flags or "complete_match_bundle" in breakdown.score_flags:
            forced[breakdown.candidate_id] = item
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
    if forced:
        LOGGER.info("Forced %d high-recall/complete-match candidates into reranking pool", len(forced))
    retained = {item.breakdown.candidate_id: item for _score, _reverse_id, item in heap}
    retained.update(forced)
    return list(retained.values()), total


def _sort_best_first(items: list[CandidateScore]) -> list[CandidateScore]:
    return sorted(
        items,
        key=lambda item: (-item.breakdown.final_score, item.breakdown.candidate_id),
    )


def _build_ranked_rows(items: list[CandidateScore], rolespec: RoleSpec | None = None) -> list[RankedCandidate]:
    rows: list[RankedCandidate] = []
    for rank, item in enumerate(items, start=1):
        breakdown = item.breakdown
        if rolespec is not None and breakdown.candidate_evidence is not None:
            reasoning = build_rolespec_reasoning(
                candidate_evidence=breakdown.candidate_evidence,
                rolespec=rolespec,
                penalties=breakdown.penalties,
                rank=rank,
                candidate=item.candidate,
            )
        else:
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
    _write_component_scores(output_dir / "component_scores.csv", ranked_rows, items)
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
                "recruiter_response_rate",
                "notice_period_days",
                "reasoning",
                "top_competencies",
                "missing_must_have",
                "score_flags",
            ],
        )
        writer.writeheader()
        for row, item in zip(rows, items, strict=True):
            profile = _profile(item.candidate)
            redrob_signals = item.candidate.get("redrob_signals", {})
            if not isinstance(redrob_signals, dict):
                redrob_signals = {}
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "debug_rank": row.rank,
                    "score": f"{row.score:.6f}",
                    "current_title": profile.get("current_title", ""),
                    "years_of_experience": profile.get("years_of_experience", ""),
                    "recruiter_response_rate": redrob_signals.get("recruiter_response_rate", ""),
                    "notice_period_days": redrob_signals.get("notice_period_days", ""),
                    "reasoning": row.reasoning,
                    "top_competencies": _top_competencies(item.breakdown),
                    "missing_must_have": " | ".join(item.breakdown.missing_must_have),
                    "score_flags": " | ".join(item.breakdown.score_flags),
                }
            )


def _write_component_scores(path: Path, rows: list[RankedCandidate], items: list[CandidateScore] | None = None) -> None:
    component_names = sorted({name for row in rows for name in row.component_scores})
    competency_names = sorted(
        {name for item in items or [] for name in item.breakdown.competency_scores}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "debug_rank",
                "score",
                *component_names,
                *[f"competency_{name}" for name in competency_names],
            ],
        )
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "candidate_id": row.candidate_id,
                "debug_rank": row.rank,
                "score": f"{row.score:.6f}",
            }
            payload.update({name: f"{row.component_scores.get(name, 0.0):.6f}" for name in component_names})
            if items is not None:
                item = items[row.rank - 1]
                payload.update(
                    {
                        f"competency_{name}": f"{item.breakdown.competency_scores.get(name, 0.0):.6f}"
                        for name in competency_names
                    }
                )
            writer.writerow(payload)


def _top_competencies(breakdown: ScoreBreakdown, limit: int = 5) -> str:
    return " | ".join(
        f"{name}:{score:.3f}"
        for name, score in sorted(
            breakdown.competency_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    )


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
