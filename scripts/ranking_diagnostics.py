"""Diagnostics for final ranking quality and evidence composition."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_evidence import extract_candidate_evidence
from src.io import iter_candidates_jsonl
from src.rolespec import load_rolespec
from src.scoring import score_candidate_with_components
from src.config import RankingConfig


DEBUG_IDS = ("CAND_0086022", "CAND_0046525", "CAND_0036184", "CAND_0069905")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect final ranking evidence composition.")
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--submission", default=Path("submission.csv"), type=Path)
    parser.add_argument("--role", default=Path("configs/rolespec_redrob_senior_ai_engineer.yaml"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rolespec = load_rolespec(args.role)
    submission = _read_submission(args.submission)
    rank_by_id = {row["candidate_id"]: int(row["rank"]) for row in submission}
    reasoning_by_id = {row["candidate_id"]: row.get("reasoning", "") for row in submission}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    complete_ids: list[str] = []
    scored_debug: dict[str, Any] = {}

    for candidate in iter_candidates_jsonl(args.candidates):
        cid = str(candidate.get("candidate_id", ""))
        if cid in rank_by_id or cid in DEBUG_IDS:
            candidates_by_id[cid] = candidate
        evidence = extract_candidate_evidence(candidate)
        breakdown = score_candidate_with_components(candidate, RankingConfig(), rolespec)
        if "complete_match_bundle" in breakdown.score_flags:
            complete_ids.append(cid)
        if cid in DEBUG_IDS:
            scored_debug[cid] = breakdown

    print("# Ranking Diagnostics")
    print(f"Submission rows: {len(submission)}")
    print(f"Complete-match candidates in full set: {len(complete_ids)}")
    print(f"Complete-match candidates excluded from top100: {len([cid for cid in complete_ids if cid not in rank_by_id])}")

    for cutoff in (10, 25, 50, 100):
        ids = [row["candidate_id"] for row in submission if int(row["rank"]) <= cutoff]
        print(f"\n## Top {cutoff}")
        print(_cutoff_metrics(ids, candidates_by_id, rolespec))

    excluded = [cid for cid in complete_ids if cid not in rank_by_id]
    print("\n## Complete Matches Excluded From Top100")
    if not excluded:
        print("None")
    else:
        for cid in excluded[:30]:
            print(f"- {cid}")

    print("\n## Reason Phrase Frequency")
    for phrase, count in _reason_phrase_frequency(reasoning_by_id.values()).most_common(20):
        print(f"- {phrase}: {count}")

    print("\n## Debug Examples")
    for cid in DEBUG_IDS:
        breakdown = scored_debug.get(cid)
        rank = rank_by_id.get(cid)
        if breakdown is None:
            print(f"- {cid}: not found in candidates file")
            continue
        flags = ",".join(breakdown.score_flags) or "-"
        reason = reasoning_by_id.get(cid, "")
        print(
            f"- {cid}: rank={rank or 'excluded'} score={breakdown.final_score:.6f} "
            f"flags={flags} missing={','.join(breakdown.missing_must_have) or '-'}"
        )
        if reason:
            print(f"  reasoning: {reason}")
    return 0


def _read_submission(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cutoff_metrics(ids: list[str], candidates_by_id: dict[str, dict[str, Any]], rolespec) -> str:
    counts = Counter()
    response_rates: list[float] = []
    notices: list[float] = []
    for cid in ids:
        candidate = candidates_by_id.get(cid)
        if not candidate:
            continue
        breakdown = score_candidate_with_components(candidate, RankingConfig(), rolespec)
        evidence = breakdown.evidence
        flags = set(breakdown.score_flags)
        if "complete_match_bundle" in flags:
            counts["complete_match"] += 1
        if _career_signal(evidence, "evaluation_framework_experience") >= 0.35:
            counts["career_eval"] += 1
        if _career_signal(evidence, "vector_database_experience") >= 0.35:
            counts["career_vector"] += 1
        if _career_signal(evidence, "retrieval_search_ranking_experience") >= 0.35:
            counts["career_rank_or_retrieval"] += 1
        title = str(candidate.get("profile", {}).get("current_title", "")).lower()
        if "junior" in title:
            counts["junior"] += 1
        if "cap_unrelated_title" in flags or "cap_unrelated_title_and_history" in flags:
            counts["unrelated"] += 1
        if "cap_skills_only_match" in flags:
            counts["skills_only"] += 1
        signals = candidate.get("redrob_signals", {}) if isinstance(candidate.get("redrob_signals", {}), dict) else {}
        response_rates.append(_safe_float(signals.get("recruiter_response_rate")))
        notices.append(_safe_float(signals.get("notice_period_days")))
    avg_response = sum(response_rates) / len(response_rates) if response_rates else 0.0
    median_notice = statistics.median(notices) if notices else 0.0
    return (
        f"complete_match={counts['complete_match']}/{len(ids)} "
        f"career_eval={counts['career_eval']}/{len(ids)} "
        f"career_vector={counts['career_vector']}/{len(ids)} "
        f"career_rank_or_retrieval={counts['career_rank_or_retrieval']}/{len(ids)} "
        f"junior={counts['junior']} unrelated={counts['unrelated']} skills_only={counts['skills_only']} "
        f"avg_response_rate={avg_response:.3f} median_notice={median_notice:g}"
    )


def _career_signal(evidence, signal_name: str) -> float:
    signal = evidence.signals.get(signal_name)
    if signal is None:
        return 0.0
    if not any(field.startswith("career_history") for field in signal.source_fields):
        return 0.0
    return signal.value


def _reason_phrase_frequency(reasonings) -> Counter:
    counter: Counter[str] = Counter()
    patterns = (
        "BM25 + dense retrieval",
        "BM25 + FAISS retrieval",
        "FAISS HNSW retrieval",
        "NDCG/MRR/recall@K evaluation",
        "NDCG/MRR evaluation",
        "A/B testing",
        "A/B engagement metrics",
        "offline-online A/B correlation",
        "human relevance judgments",
        "semantic search over 500K documents",
        "ranking pipeline serving 50M+ queries/month",
        "30M+ candidate corpus",
        "recruiter engagement lift",
        "time-to-shortlist improvement",
    )
    for reasoning in reasonings:
        for pattern in patterns:
            if pattern.lower() in reasoning.lower():
                counter[pattern] += 1
        if not any(pattern.lower() in reasoning.lower() for pattern in patterns):
            first = re.split(r"[,.:]", reasoning)[0].strip()
            if first:
                counter[first] += 1
    return counter


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
