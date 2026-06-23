"""Audit generated submission reasoning for grounding and tone."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TECH_SKILL_TERMS = {
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "spark",
    "airflow",
    "kafka",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "langchain",
    "openai",
    "rag",
    "llm",
    "nlp",
    "mlflow",
    "kubeflow",
    "bentoml",
    "xgboost",
    "sql",
    "recommendation",
    "retrieval",
    "ranking",
    "search",
    "embeddings",
}

STRONG_TONE = {"high-confidence", "strong", "compelling", "clear fit"}
CAUTIOUS_TONE = {"cautious", "borderline", "lower-confidence", "possible fit", "concern", "watch-out", "caveat", "tradeoff", "weaker"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit submission reasoning quality.")
    parser.add_argument("--submission", default="submission.csv", type=Path)
    parser.add_argument("--candidates", default="candidates.jsonl", type=Path)
    parser.add_argument("--sample-size", default=20, type=int)
    return parser.parse_args()


def read_submission(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_candidates(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while not first:
            first = handle.read(1)
            if first == "":
                return
            first = first.strip()
        handle.seek(0)
        if first == "[":
            for item in json.load(handle):
                yield item
            return
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def load_candidate_subset(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for candidate in iter_candidates(path):
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if candidate_id in candidate_ids:
            found[candidate_id] = candidate
            if len(found) == len(candidate_ids):
                break
    return found


def sample_rows(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, str]]:
    if len(rows) <= sample_size:
        return rows
    indexes = {0, 1, 2, 9, len(rows) - 1, len(rows) - 2, len(rows) - 3}
    step = max(1, len(rows) // sample_size)
    indexes.update(range(0, len(rows), step))
    selected = [rows[index] for index in sorted(index for index in indexes if 0 <= index < len(rows))]
    return selected[:sample_size]


def candidate_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    parts = [
        str(profile.get("current_title", "")),
        str(profile.get("headline", "")),
        str(profile.get("summary", "")),
        str(profile.get("current_company", "")),
        str(profile.get("current_industry", "")),
    ]
    for skill in candidate.get("skills", []):
        if isinstance(skill, dict):
            parts.append(str(skill.get("name", "")))
    for role in candidate.get("career_history", []):
        if isinstance(role, dict):
            parts.extend([str(role.get("company", "")), str(role.get("title", "")), str(role.get("description", ""))])
    return " ".join(parts).lower()


def unsupported_skill_claims(reasoning: str, candidate: dict[str, Any]) -> list[str]:
    lowered_reasoning = reasoning.lower()
    text = candidate_text(candidate)
    unsupported: list[str] = []
    for term in sorted(TECH_SKILL_TERMS):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered_reasoning) and term not in text:
            unsupported.append(term)
    return unsupported


def tone_errors(row: dict[str, str]) -> list[str]:
    rank = int(row.get("rank") or 0)
    reasoning = row.get("reasoning", "").lower()
    errors: list[str] = []
    if rank <= 10 and not any(term in reasoning for term in STRONG_TONE):
        errors.append("top-10 reasoning is not confident enough")
    if rank >= 80 and not any(term in reasoning for term in CAUTIOUS_TONE):
        errors.append("rank 80-100 reasoning is not cautious enough")
    if rank <= 10 and any(term in reasoning for term in {"borderline", "lower-confidence"}):
        errors.append("top-10 reasoning sounds too cautious")
    return errors


def template_key(reasoning: str) -> str:
    normalized = re.sub(r"\d+(?:\.\d+)?", "<num>", reasoning.lower())
    normalized = re.sub(r"[^a-z0-9<>\s-]", " ", normalized)
    return " ".join(normalized.split()[:9])


def main() -> int:
    args = parse_args()
    rows = read_submission(args.submission)
    sampled = sample_rows(rows, args.sample_size)
    candidates = load_candidate_subset(args.candidates, {row["candidate_id"] for row in sampled})

    failures: list[str] = []
    empty = [row["candidate_id"] for row in sampled if not row.get("reasoning", "").strip()]
    if empty:
        failures.append(f"empty reasoning: {empty}")

    template_counts = Counter(template_key(row.get("reasoning", "")) for row in sampled)
    repeated = [key for key, count in template_counts.items() if key and count > 3]
    if repeated:
        failures.append(f"repeated template prefixes: {repeated}")

    for row in sampled:
        candidate_id = row["candidate_id"]
        candidate = candidates.get(candidate_id)
        if not candidate:
            failures.append(f"candidate not found for {candidate_id}")
            continue
        unsupported = unsupported_skill_claims(row.get("reasoning", ""), candidate)
        if unsupported:
            failures.append(f"{candidate_id} unsupported skill claims: {unsupported}")
        for error in tone_errors(row):
            failures.append(f"{candidate_id} rank {row.get('rank')}: {error}")

    print(f"Audited {len(sampled)} reasoning rows from {args.submission}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
