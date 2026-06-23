"""Proxy quality evaluation for the deterministic ranker.

Hidden relevance labels are unavailable, so this script inspects whether the
top-ranked candidates match the JD's observable expectations.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RankingConfig
from src.io import iter_candidates_jsonl
from src.reasoning import build_reasoning
from src.scoring import COMPONENT_NAMES, ScoreBreakdown, score_candidate_with_components


LOGGER = logging.getLogger(__name__)
REFERENCE_DATE = date(2026, 6, 23)
AI_TITLE_TERMS = {
    "ai",
    "machine learning",
    "ml",
    "data scientist",
    "applied scientist",
    "recommendation",
    "search",
    "ranking",
    "retrieval",
    "nlp",
    "mlops",
}


class ReverseString:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: "ReverseString") -> bool:
        return self.value > other.value


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    candidate: dict[str, Any]
    breakdown: ScoreBreakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy-evaluate ranker output without hidden labels.")
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--top-k", default=100, type=int)
    parser.add_argument("--print-top", default=25, type=int)
    parser.add_argument("--high-penalty", default=15, type=int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = RankingConfig()

    top_candidates, high_penalty, total = _scan_candidates(
        args.candidates,
        config=config,
        top_k=args.top_k,
        high_penalty_k=args.high_penalty,
    )
    ranked = sorted(top_candidates, key=lambda item: (-item.breakdown.final_score, item.breakdown.candidate_id))
    high_penalty_sorted = sorted(
        high_penalty,
        key=lambda item: (-item.breakdown.penalties.penalty_score, -item.breakdown.final_score, item.breakdown.candidate_id),
    )

    print(f"# Proxy Ranker Quality Evaluation")
    print(f"Candidates scanned: {total}")
    print(f"Top candidates retained: {len(ranked)}")
    _print_top_candidates(ranked[: args.print_top], printed_count=args.print_top)
    _print_high_penalty_examples(high_penalty_sorted)
    _print_top_100_metrics(ranked[:100])
    _print_skill_high_career_low_flags(ranked[:100])
    return 0


def _scan_candidates(
    path: Path,
    config: RankingConfig,
    top_k: int,
    high_penalty_k: int,
) -> tuple[list[EvaluatedCandidate], list[EvaluatedCandidate], int]:
    top_heap: list[tuple[float, ReverseString, EvaluatedCandidate]] = []
    penalty_heap: list[tuple[float, str, EvaluatedCandidate]] = []
    total = 0
    for candidate in iter_candidates_jsonl(path):
        total += 1
        breakdown = score_candidate_with_components(candidate, config)
        item = EvaluatedCandidate(candidate=candidate, breakdown=breakdown)
        top_entry = (breakdown.final_score, ReverseString(breakdown.candidate_id), item)
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, top_entry)
        else:
            heapq.heappushpop(top_heap, top_entry)

        penalty_entry = (breakdown.penalties.penalty_score, breakdown.candidate_id, item)
        if len(penalty_heap) < high_penalty_k:
            heapq.heappush(penalty_heap, penalty_entry)
        else:
            heapq.heappushpop(penalty_heap, penalty_entry)

        if total % 10000 == 0:
            LOGGER.info("Evaluated %d candidates", total)
    return (
        [item for _score, _candidate_id, item in top_heap],
        [item for _penalty, _candidate_id, item in penalty_heap],
        total,
    )


def _print_top_candidates(items: list[EvaluatedCandidate], printed_count: int) -> None:
    print(f"\n## Top {printed_count} Candidates")
    for rank, item in enumerate(items, start=1):
        profile = _profile(item.candidate)
        reasoning = build_reasoning(
            evidence=item.breakdown.evidence,
            penalties=item.breakdown.penalties,
            rank=rank,
            component_scores=item.breakdown.component_scores,
            candidate=item.candidate,
        )
        components = " ".join(
            f"{name.replace('_score', '')}={item.breakdown.component_scores[name]:.2f}"
            for name in COMPONENT_NAMES
        )
        print(
            f"{rank:>2}. {item.breakdown.candidate_id} score={item.breakdown.final_score:.4f} "
            f"title={profile.get('current_title', '')!r} years={profile.get('years_of_experience', '')}"
        )
        print(f"    {components}")
        print(f"    reasoning: {reasoning}")


def _print_high_penalty_examples(items: list[EvaluatedCandidate]) -> None:
    print("\n## Bottom / High-Penalty Examples")
    for item in items:
        profile = _profile(item.candidate)
        penalties = item.breakdown.penalties
        print(
            f"- {item.breakdown.candidate_id} score={item.breakdown.final_score:.4f} "
            f"penalty={penalties.penalty_score:.4f} title={profile.get('current_title', '')!r}"
        )
        if penalties.penalty_reasons:
            print(f"  reasons: {' | '.join(penalties.penalty_reasons[:4])}")


def _print_top_100_metrics(items: list[EvaluatedCandidate]) -> None:
    print("\n## Top 100 Proxy Metrics")
    if not items:
        print("No top candidates available.")
        return

    suspicious = sum(1 for item in items if item.breakdown.penalties.hard_flags)
    retrieval_count = sum(
        1 for item in items if item.breakdown.evidence.value("retrieval_search_ranking_experience") >= 0.45
    )
    non_ai_titles = sum(1 for item in items if not _is_ai_title(_profile(item.candidate).get("current_title", "")))
    title_categories = Counter(_title_category(_profile(item.candidate).get("current_title", "")) for item in items)
    response_rates = [_safe_float(_signals(item.candidate).get("recruiter_response_rate")) for item in items]
    notice_periods = [_safe_float(_signals(item.candidate).get("notice_period_days")) for item in items]
    recencies = [_last_active_days(_signals(item.candidate).get("last_active_date")) for item in items]
    recencies = [value for value in recencies if value is not None]

    print(f"Suspicious profiles with hard flags: {suspicious}/{len(items)}")
    print(f"Retrieval/search/ranking evidence >= 0.45: {retrieval_count}/{len(items)}")
    print(f"Non-AI current titles: {non_ai_titles}/{len(items)}")
    print("Title categories:")
    for category, count in title_categories.most_common():
        print(f"  {category}: {count}")
    print(f"Average recruiter response rate: {_mean(response_rates):.3f}")
    print(f"Average days since last active: {_mean(recencies):.1f}" if recencies else "Average days since last active: n/a")
    print(f"Average notice period: {_mean(notice_periods):.1f} days")


def _print_skill_high_career_low_flags(items: list[EvaluatedCandidate]) -> None:
    print("\n## High Skill Score / Low Career Evidence Flags")
    flagged = []
    for rank, item in enumerate(items, start=1):
        components = item.breakdown.component_scores
        career_evidence = max(
            components.get("production_ml_score", 0.0),
            components.get("retrieval_ranking_score", 0.0),
            components.get("role_fit_score", 0.0),
        )
        if components.get("skill_quality_score", 0.0) >= 0.45 and career_evidence < 0.35:
            flagged.append((rank, item))
    if not flagged:
        print("No top-100 candidates with high skill score but low career evidence.")
        return
    for rank, item in flagged[:20]:
        profile = _profile(item.candidate)
        print(
            f"- rank={rank} {item.breakdown.candidate_id} title={profile.get('current_title', '')!r} "
            f"skill={item.breakdown.component_scores['skill_quality_score']:.2f} "
            f"production={item.breakdown.component_scores['production_ml_score']:.2f} "
            f"retrieval={item.breakdown.component_scores['retrieval_ranking_score']:.2f}"
        )


def _profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    return profile if isinstance(profile, dict) else {}


def _signals(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _is_ai_title(title: Any) -> bool:
    lowered = str(title).lower()
    return any(term in lowered for term in AI_TITLE_TERMS)


def _title_category(title: Any) -> str:
    lowered = str(title).lower()
    if _is_ai_title(lowered):
        return "ai_ml_data"
    if "engineer" in lowered or "developer" in lowered or "architect" in lowered:
        return "software_engineering"
    if "manager" in lowered or "lead" in lowered:
        return "management_business"
    if "analyst" in lowered:
        return "analyst"
    return "other"


def _last_active_days(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None
    return float((REFERENCE_DATE - parsed).days)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
