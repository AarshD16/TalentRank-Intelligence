"""Diagnostics for capability-vs-hireability ranking behavior."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RankingConfig
from src.reasoning import build_reasoning
from src.rolespec import load_rolespec
from src.scoring import score_candidate_with_components


PHRASES = (
    "BM25 + dense retrieval",
    "BM25/Elasticsearch retrieval",
    "NDCG/MRR evaluation",
    "MRR/MAP evaluation",
    "A/B testing",
    "offline-online A/B correlation",
    "production recommendation system",
    "semantic search over 500K documents",
    "semantic search",
    "discovery feed ranking",
    "learning-to-rank",
    "learning to rank",
    "recruiter response rate",
    "notice period",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate capability/hireability ranking diagnostics.")
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--submission", default=Path("submission.csv"), type=Path)
    parser.add_argument("--previous", default=Path("outputs/submission_updated.csv"), type=Path)
    parser.add_argument("--teacher", default=Path("top_100_candidates.csv"), type=Path)
    parser.add_argument("--role", default=Path("configs/rolespec_redrob_ai_engineer.yaml"), type=Path)
    parser.add_argument("--out", default=Path("outputs/capability_hireability_diagnostics.md"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_rank = _read_rank_csv(args.submission)
    top_ids = set(submission_rank)
    rolespec = load_rolespec(args.role)
    config = RankingConfig()

    top_rows: list[tuple[dict[str, Any], Any]] = []
    flag_counts: Counter[str] = Counter()
    boosted: list[str] = []
    demoted: list[str] = []
    capped: Counter[str] = Counter()

    with args.candidates.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            breakdown = score_candidate_with_components(candidate, config, rolespec=rolespec)
            flag_counts.update(breakdown.score_flags)
            for flag in breakdown.score_flags:
                if flag.startswith("cap_"):
                    capped[flag] += 1
            if "hireability_boost" in breakdown.score_flags:
                boosted.append(breakdown.candidate_id)
            if "low_activity_response_demotion" in breakdown.score_flags:
                demoted.append(breakdown.candidate_id)
            if breakdown.candidate_id in top_ids:
                top_rows.append((candidate, breakdown))

    top_rows.sort(key=lambda pair: submission_rank[pair[1].candidate_id])
    title_distribution = Counter(_title_bucket(candidate) for candidate, _breakdown in top_rows)
    signal_averages = _redrob_signal_averages([candidate for candidate, _breakdown in top_rows])
    phrase_counts = _reasoning_phrase_counts(top_rows)
    overlaps = _overlaps(args.submission, args.previous, args.teacher)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _render_report(
            total_scanned=sum(flag_counts.values()) or 0,
            title_distribution=title_distribution,
            signal_averages=signal_averages,
            capped=capped,
            boosted_count=len(boosted),
            demoted_count=len(demoted),
            phrase_counts=phrase_counts,
            overlaps=overlaps,
        ),
        encoding="utf-8",
    )
    print(f"Wrote diagnostics to {args.out}")
    return 0


def _read_rank_csv(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    id_header = "candidate_id" if "candidate_id" in rows[0] else "candidate id" if "candidate id" in rows[0] else "id"
    return {row[id_header].strip(): int(row.get("rank", index + 1)) for index, row in enumerate(rows) if row.get(id_header)}


def _title_bucket(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    title = str(profile.get("current_title", "")).lower()
    if "recommendation" in title:
        return "recommendation_systems"
    if "search" in title:
        return "search"
    if "nlp" in title:
        return "nlp"
    if "senior data scientist" in title:
        return "senior_data_scientist"
    if "ai engineer" in title:
        return "ai_engineer"
    if "machine learning" in title or re.search(r"\bml engineer\b", title):
        return "ml_engineer"
    if "applied ml" in title:
        return "applied_ml"
    return "other"


def _redrob_signal_averages(candidates: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "recruiter_response_rate",
        "avg_response_time_hours",
        "notice_period_days",
        "interview_completion_rate",
        "offer_acceptance_rate",
        "applications_submitted_30d",
        "profile_views_received_30d",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
    )
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        signals = candidate.get("redrob_signals", {}) if isinstance(candidate.get("redrob_signals", {}), dict) else {}
        for name in names:
            value = signals.get(name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if name == "offer_acceptance_rate" and number < 0:
                continue
            totals[name] += number
            counts[name] += 1
    return {name: totals[name] / counts[name] for name in names if counts[name]}


def _reasoning_phrase_counts(top_rows: list[tuple[dict[str, Any], Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rank, (candidate, breakdown) in enumerate(top_rows, start=1):
        reasoning = build_reasoning(
            evidence=breakdown.evidence,
            penalties=breakdown.penalties,
            rank=rank,
            component_scores=breakdown.component_scores,
            candidate=candidate,
        )
        lowered = reasoning.lower()
        for phrase in PHRASES:
            if phrase.lower() in lowered:
                counts[phrase] += 1
    return counts


def _overlaps(submission: Path, previous: Path, teacher: Path) -> dict[str, int]:
    current = set(_read_rank_csv(submission))
    result: dict[str, int] = {}
    if previous.exists():
        result["previous_overlap_at_100"] = len(current & set(_read_rank_csv(previous)))
    if teacher.exists():
        result["teacher_overlap_at_100"] = len(current & set(_read_rank_csv(teacher)))
    return result


def _render_report(
    total_scanned: int,
    title_distribution: Counter[str],
    signal_averages: dict[str, float],
    capped: Counter[str],
    boosted_count: int,
    demoted_count: int,
    phrase_counts: Counter[str],
    overlaps: dict[str, int],
) -> str:
    lines = ["# Capability vs Hireability Diagnostics", ""]
    lines.append("## Top 100 Title Distribution")
    lines.extend(f"- {name}: {count}" for name, count in title_distribution.most_common())
    lines.append("")
    lines.append("## Top 100 Redrob Signal Averages")
    lines.extend(f"- {name}: {value:.3f}" for name, value in signal_averages.items())
    lines.append("")
    lines.append("## Candidates Capped By Rule")
    lines.extend(f"- {name}: {count}" for name, count in capped.most_common())
    lines.append("")
    lines.append("## Hireability Modifier Effects")
    lines.append(f"- candidates boosted by hireability: {boosted_count}")
    lines.append(f"- candidates demoted by low activity/response: {demoted_count}")
    lines.append("")
    lines.append("## Reasoning Phrase Frequency")
    lines.extend(f"- {phrase}: {count}" for phrase, count in phrase_counts.most_common())
    lines.append("")
    lines.append("## Overlap")
    lines.extend(f"- {name}: {count}" for name, count in overlaps.items())
    lines.append("")
    lines.append(f"Scored flag events counted across scan: {total_scanned}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
