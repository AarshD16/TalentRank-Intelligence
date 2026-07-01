"""Compare current submission against teacher top-100 for calibration only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_extractor import extract_evidence


PATTERN_TERMS = {
    "semantic_search": ("semantic search",),
    "discovery_feed": ("discovery feed",),
    "personalization": ("personalization", "personalized"),
    "learning_to_rank": ("learning to rank", "learning-to-rank"),
    "ab_testing": ("a/b", "ab testing", "a/b testing"),
    "ranking_metrics": ("ndcg", "mrr", "map"),
    "offline_online": ("offline-online", "offline online"),
    "dense_retrieval": ("dense retrieval", "nearest-neighbor", "nearest neighbor"),
    "hybrid_search": ("hybrid search", "bm25"),
    "vector_ops": ("faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "pgvector"),
    "production_recsys": ("recommendation system", "ranking layer", "product's discovery feed"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare submission.csv with teacher CSV for calibration.")
    parser.add_argument("--submission", default=Path("submission.csv"), type=Path)
    parser.add_argument("--teacher", default=Path("top_100_candidates.csv"), type=Path)
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--inspect", default=25, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission = _read_rank_csv(args.submission, ("candidate_id",))
    teacher = _read_rank_csv(args.teacher, ("candidate_id", "candidate id", "id"))
    submission_rank = {row["candidate_id"]: int(row["rank"]) for row in submission}
    teacher_rank = {row["candidate_id"]: int(row["rank"]) for row in teacher}

    overlap = set(submission_rank) & set(teacher_rank)
    missing = sorted((cid for cid in teacher_rank if cid not in submission_rank), key=lambda cid: teacher_rank[cid])
    low = sorted(
        (cid for cid in teacher_rank if cid in submission_rank and submission_rank[cid] - teacher_rank[cid] >= 30),
        key=lambda cid: teacher_rank[cid],
    )

    print("# Teacher Calibration Comparison")
    print(f"Submission rows: {len(submission_rank)}")
    print(f"Teacher rows: {len(teacher_rank)}")
    print(f"Overlap@100: {len(overlap)}")
    print(f"Teacher high-ranked but missing: {len(missing)}")
    print(f"Teacher high-ranked but 30+ ranks lower: {len(low)}")

    inspect_ids = missing[: args.inspect]
    profiles = _load_profiles(args.candidates, set(inspect_ids))
    pattern_counts: Counter[str] = Counter()
    print("\n## Teacher Missing / Low Examples")
    for cid in inspect_ids:
        profile = profiles.get(cid)
        if not profile:
            continue
        evidence = extract_evidence(profile)
        matched = _matched_patterns(profile)
        pattern_counts.update(matched)
        p = profile.get("profile", {})
        print(
            f"- teacher_rank={teacher_rank[cid]} {cid} title={p.get('current_title')} "
            f"years={p.get('years_of_experience')} semantic={evidence.value('semantic_jd_intent_score'):.3f} "
            f"retrieval={evidence.value('semantic_production_retrieval_score'):.3f} "
            f"eval={evidence.value('semantic_evaluation_score'):.3f} patterns={','.join(matched)}"
        )

    print("\n## Missing Pattern Counts")
    for pattern, count in pattern_counts.most_common():
        print(f"{pattern}: {count}")
    print("\nNote: this script is for development calibration only. Do not call it from rank.py.")
    return 0


def _read_rank_csv(path: Path, id_headers: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        candidate_id = ""
        for header in id_headers:
            if header in row:
                candidate_id = row[header].strip()
                break
        if not candidate_id:
            continue
        normalized.append({"candidate_id": candidate_id, "rank": row.get("rank", "999999")})
    return normalized


def _load_profiles(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            candidate_id = candidate.get("candidate_id")
            if candidate_id in candidate_ids:
                profiles[candidate_id] = candidate
                if len(profiles) == len(candidate_ids):
                    break
    return profiles


def _matched_patterns(candidate: dict[str, Any]) -> list[str]:
    text = json.dumps(candidate, ensure_ascii=False).lower()
    return [name for name, terms in PATTERN_TERMS.items() if any(term in text for term in terms)]


if __name__ == "__main__":
    raise SystemExit(main())
