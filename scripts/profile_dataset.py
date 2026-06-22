"""Stream and profile the Redrob candidate dataset.

This script is exploratory only. It summarizes field distributions and surfaces
example candidates for manual inspection without loading candidates.jsonl into
memory.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any


LOGGER = logging.getLogger(__name__)

AI_TITLE_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning|ml|nlp|llm|data scientist|"
    r"applied scientist|research scientist|ranking|retrieval|recommendation|"
    r"search engineer|mlops|computer vision)\b",
    re.IGNORECASE,
)

AI_SKILL_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "nlp",
    "llm",
    "rag",
    "embeddings",
    "vector search",
    "recommendation systems",
    "ranking",
    "learning to rank",
    "fine-tuning llms",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "transformers",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "xgboost",
    "langchain",
}

CORE_JD_TERMS = {
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "ranker",
    "recommendation",
    "recommender",
    "search",
    "vector",
    "faiss",
    "qdrant",
    "weaviate",
    "pinecone",
    "milvus",
    "elasticsearch",
    "opensearch",
    "ndcg",
    "mrr",
    "map",
    "a/b",
    "ab test",
    "fine-tuning",
    "finetuning",
    "llm",
    "nlp",
    "python",
    "production",
}

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile candidate dataset distributions.")
    parser.add_argument(
        "--input",
        default="candidates.jsonl",
        type=Path,
        help="Path to candidates.jsonl. JSON arrays are also supported for sample files.",
    )
    parser.add_argument("--top-n", default=20, type=int, help="Rows to print per distribution.")
    parser.add_argument(
        "--examples",
        default=5,
        type=int,
        help="Number of strong, weak, suspicious, and keyword-stuffed examples to print.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def iter_candidates(path: Path) -> Iterator[dict[str, Any]]:
    """Yield candidates from JSONL without loading the whole file.

    The bundled sample file is a JSON array, so this function supports that
    shape for local EDA while preserving JSONL streaming for the full dataset.
    """
    with path.open("r", encoding="utf-8") as handle:
        first_char = ""
        while not first_char:
            first_char = handle.read(1)
            if first_char == "":
                return
            first_char = first_char.strip()
        handle.seek(0)

        if first_char == "[":
            LOGGER.warning("Input is a JSON array; this mode is intended for small samples.")
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Expected a JSON array of candidates.")
            for item in payload:
                if isinstance(item, dict):
                    yield item
                else:
                    raise ValueError("Expected each candidate to be a JSON object.")
            return

        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield item


def safe_get(candidate: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = candidate
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def bucket_years(years: Any) -> str:
    try:
        value = float(years)
    except (TypeError, ValueError):
        return "unknown"
    if value < 3:
        return "0-2"
    if value < 5:
        return "3-4"
    if value <= 9:
        return "5-9"
    if value <= 12:
        return "10-12"
    return "13+"


def bucket_number(value: Any, buckets: tuple[float, ...]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    previous: float | None = None
    for upper in buckets:
        if number <= upper:
            if previous is None:
                return f"<={upper:g}"
            return f"{previous:g}-{upper:g}"
        previous = upper
    return f">{buckets[-1]:g}"


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def months_between(start: Any, end: Any) -> int | None:
    start_date = parse_date(start)
    end_date = parse_date(end) or date.today()
    if not start_date:
        return None
    return max(0, (end_date.year - start_date.year) * 12 + end_date.month - start_date.month)


def days_since(value: Any) -> int | None:
    parsed = parse_date(value)
    if not parsed:
        return None
    return (date.today() - parsed).days


def candidate_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    parts = [
        normalized_text(profile.get("headline")),
        normalized_text(profile.get("summary")),
        normalized_text(profile.get("current_title")),
        normalized_text(profile.get("current_industry")),
    ]
    if isinstance(career, list):
        for role in career:
            if isinstance(role, dict):
                parts.extend(
                    [
                        normalized_text(role.get("title")),
                        normalized_text(role.get("industry")),
                        normalized_text(role.get("description")),
                    ]
                )
    return " ".join(part for part in parts if part)


def count_term_hits(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def education_tier(candidate: dict[str, Any]) -> str:
    tiers = [
        normalized_text(item.get("tier"))
        for item in candidate.get("education", [])
        if isinstance(item, dict)
    ]
    if not tiers:
        return "missing"
    order = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5, "": 6}
    return min(tiers, key=lambda tier: order.get(tier, 99)) or "unknown"


def ai_skill_count(candidate: dict[str, Any]) -> int:
    count = 0
    for skill in candidate.get("skills", []):
        if not isinstance(skill, dict):
            continue
        name = normalized_text(skill.get("name")).lower()
        if any(term in name for term in AI_SKILL_TERMS):
            count += 1
    return count


def expert_zero_duration_count(candidate: dict[str, Any]) -> int:
    count = 0
    for skill in candidate.get("skills", []):
        if not isinstance(skill, dict):
            continue
        if skill.get("proficiency") == "expert" and int(skill.get("duration_months") or 0) == 0:
            count += 1
    return count


def service_only_career(candidate: dict[str, Any]) -> bool:
    companies = []
    for role in candidate.get("career_history", []):
        if isinstance(role, dict):
            companies.append(normalized_text(role.get("company")).lower())
    return bool(companies) and all(company in SERVICES_COMPANIES for company in companies)


def suspicious_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience") or 0)
    career = candidate.get("career_history", [])
    career_months = 0
    current_roles = 0
    if isinstance(career, list):
        for role in career:
            if not isinstance(role, dict):
                continue
            career_months += int(role.get("duration_months") or 0)
            if role.get("is_current") is True:
                current_roles += 1
            inferred = months_between(role.get("start_date"), role.get("end_date"))
            stated = int(role.get("duration_months") or 0)
            if inferred is not None and abs(inferred - stated) > 18:
                reasons.append("role duration does not match start/end dates")
                break
    if current_roles != 1:
        reasons.append(f"{current_roles} current roles")
    if years and career_months and abs(years * 12 - career_months) > 36:
        reasons.append("career months inconsistent with years_of_experience")
    zero_experts = expert_zero_duration_count(candidate)
    if zero_experts:
        reasons.append(f"{zero_experts} expert skills with 0 months")
    salary = safe_get(candidate, "redrob_signals", "expected_salary_range_inr_lpa", default={})
    if isinstance(salary, dict) and float(salary.get("min") or 0) > float(salary.get("max") or 0):
        reasons.append("salary min greater than max")
    signals = candidate.get("redrob_signals", {})
    if isinstance(signals, dict):
        signup_date = parse_date(signals.get("signup_date"))
        last_active_date = parse_date(signals.get("last_active_date"))
        today = date.today()
        if signup_date and signup_date > today:
            reasons.append("signup date is in the future")
        if last_active_date and last_active_date > today:
            reasons.append("last active date is in the future")
        if signup_date and last_active_date and last_active_date < signup_date:
            reasons.append("last active date before signup date")
    return reasons


def keyword_stuffing_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    skills = [skill for skill in candidate.get("skills", []) if isinstance(skill, dict)]
    ai_skills = ai_skill_count(candidate)
    text = candidate_text(candidate)
    jd_hits = count_term_hits(text, CORE_JD_TERMS)
    title = normalized_text(safe_get(candidate, "profile", "current_title"))
    years = float(safe_get(candidate, "profile", "years_of_experience", default=0) or 0)

    if len(skills) >= 30:
        reasons.append(f"{len(skills)} listed skills")
    if ai_skills >= 8 and jd_hits <= 3:
        reasons.append("many AI skills but little supporting career evidence")
    if ai_skills >= 6 and not AI_TITLE_PATTERN.search(title):
        reasons.append("AI-heavy skills with non-AI current title")
    if ai_skills >= 5 and years < 3:
        reasons.append("many AI skills with low total experience")
    zero_experts = expert_zero_duration_count(candidate)
    if zero_experts >= 2:
        reasons.append("multiple expert skills with no duration")
    return reasons


def availability_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    if not isinstance(signals, dict):
        return 0.0
    score = 0.0
    score += 1.5 if signals.get("open_to_work_flag") else 0.0
    score += min(float(signals.get("recruiter_response_rate") or 0), 1.0)
    score += 1.0 if int(signals.get("notice_period_days") or 999) <= 30 else 0.0
    score += 0.5 if signals.get("willing_to_relocate") else 0.0
    score += 0.5 if signals.get("linkedin_connected") else 0.0
    score += 0.25 if signals.get("verified_email") else 0.0
    score += 0.25 if signals.get("verified_phone") else 0.0
    return score


def example_strength(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience") or 0)
    text = candidate_text(candidate)
    jd_hits = count_term_hits(text, CORE_JD_TERMS)
    score = 0.0
    score += 2.0 if 5 <= years <= 9 else 0.5 if 4 <= years <= 12 else 0.0
    score += min(jd_hits, 10) * 0.7
    score += ai_skill_count(candidate) * 0.4
    score += availability_score(candidate)
    score += 1.0 if education_tier(candidate) in {"tier_1", "tier_2"} else 0.0
    score -= len(suspicious_reasons(candidate)) * 2.0
    score -= len(keyword_stuffing_reasons(candidate)) * 1.5
    score -= 1.0 if service_only_career(candidate) else 0.0
    return score


def example_weakness(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    title = normalized_text(profile.get("current_title"))
    years = float(profile.get("years_of_experience") or 0)
    text = candidate_text(candidate)
    jd_hits = count_term_hits(text, CORE_JD_TERMS)
    score = 0.0
    score += 2.0 if not AI_TITLE_PATTERN.search(title) else 0.0
    score += 2.0 if jd_hits <= 2 else 0.0
    score += 1.0 if years < 4 or years > 12 else 0.0
    score += 1.0 if availability_score(candidate) < 1.5 else 0.0
    return score


@dataclass
class NumericSummary:
    values_seen: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    buckets: Counter[str] = field(default_factory=Counter)

    def add(self, value: Any, buckets: tuple[float, ...]) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            self.buckets["unknown"] += 1
            return
        self.values_seen += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        self.buckets[bucket_number(number, buckets)] += 1

    def average(self) -> float | None:
        if not self.values_seen:
            return None
        return self.total / self.values_seen


@dataclass
class DatasetProfile:
    total: int = 0
    current_titles: Counter[str] = field(default_factory=Counter)
    current_industries: Counter[str] = field(default_factory=Counter)
    years_buckets: Counter[str] = field(default_factory=Counter)
    countries: Counter[str] = field(default_factory=Counter)
    locations: Counter[str] = field(default_factory=Counter)
    company_sizes: Counter[str] = field(default_factory=Counter)
    education_tiers: Counter[str] = field(default_factory=Counter)
    skill_names: Counter[str] = field(default_factory=Counter)
    ai_title_counts: Counter[str] = field(default_factory=Counter)
    redrob_categorical: dict[str, Counter[str]] = field(default_factory=dict)
    redrob_numeric: dict[str, NumericSummary] = field(default_factory=dict)
    redrob_assessment_names: Counter[str] = field(default_factory=Counter)
    strong_examples: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    weak_examples: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    suspicious_examples: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    keyword_examples: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)


REDROB_NUMERIC_BUCKETS: dict[str, tuple[float, ...]] = {
    "profile_completeness_score": (50, 70, 85, 95, 100),
    "profile_views_received_30d": (0, 5, 20, 50, 100),
    "applications_submitted_30d": (0, 2, 5, 10, 25),
    "recruiter_response_rate": (0, 0.25, 0.5, 0.75, 1),
    "avg_response_time_hours": (1, 6, 24, 72, 168),
    "connection_count": (10, 50, 150, 500, 1000),
    "endorsements_received": (0, 5, 20, 50, 100),
    "notice_period_days": (0, 15, 30, 60, 90, 180),
    "github_activity_score": (-1, 0, 25, 50, 75, 100),
    "search_appearance_30d": (0, 5, 20, 50, 100),
    "saved_by_recruiters_30d": (0, 1, 3, 10, 25),
    "interview_completion_rate": (0, 0.5, 0.75, 0.9, 1),
    "offer_acceptance_rate": (-1, 0, 0.25, 0.5, 0.75, 1),
    "expected_salary_min_lpa": (10, 20, 35, 50, 75, 100),
    "expected_salary_max_lpa": (15, 30, 45, 65, 90, 120),
    "days_since_signup": (30, 90, 180, 365, 730, 1460),
    "days_since_last_active": (7, 30, 90, 180, 365),
}

REDROB_CATEGORICAL_FIELDS = {
    "open_to_work_flag",
    "preferred_work_mode",
    "willing_to_relocate",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
}


def add_heap_example(
    heap: list[tuple[float, str, dict[str, Any]]],
    score: float,
    candidate: dict[str, Any],
    limit: int,
) -> None:
    candidate_id = normalized_text(candidate.get("candidate_id"))
    item = (score, candidate_id, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def update_profile(profile: DatasetProfile, candidate: dict[str, Any], example_limit: int) -> None:
    profile.total += 1
    candidate_profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = normalized_text(candidate_profile.get("current_title")) or "missing"
    profile.current_titles[title] += 1
    profile.current_industries[normalized_text(candidate_profile.get("current_industry")) or "missing"] += 1
    profile.years_buckets[bucket_years(candidate_profile.get("years_of_experience"))] += 1
    profile.countries[normalized_text(candidate_profile.get("country")) or "missing"] += 1
    profile.locations[normalized_text(candidate_profile.get("location")) or "missing"] += 1
    profile.company_sizes[normalized_text(candidate_profile.get("current_company_size")) or "missing"] += 1
    profile.education_tiers[education_tier(candidate)] += 1

    if AI_TITLE_PATTERN.search(title):
        profile.ai_title_counts[title] += 1

    for skill in candidate.get("skills", []):
        if isinstance(skill, dict):
            profile.skill_names[normalized_text(skill.get("name")) or "missing"] += 1

    if isinstance(signals, dict):
        for field_name in REDROB_CATEGORICAL_FIELDS:
            counter = profile.redrob_categorical.setdefault(field_name, Counter())
            counter[str(signals.get(field_name, "missing"))] += 1

        for field_name, buckets in REDROB_NUMERIC_BUCKETS.items():
            summary = profile.redrob_numeric.setdefault(field_name, NumericSummary())
            if field_name == "expected_salary_min_lpa":
                value = safe_get(signals, "expected_salary_range_inr_lpa", "min")
            elif field_name == "expected_salary_max_lpa":
                value = safe_get(signals, "expected_salary_range_inr_lpa", "max")
            elif field_name == "days_since_signup":
                value = days_since(signals.get("signup_date"))
            elif field_name == "days_since_last_active":
                value = days_since(signals.get("last_active_date"))
            else:
                value = signals.get(field_name)
            summary.add(value, buckets)

        assessments = signals.get("skill_assessment_scores", {})
        if isinstance(assessments, dict):
            profile.redrob_numeric.setdefault("assessment_score", NumericSummary())
            for skill_name, score in assessments.items():
                profile.redrob_assessment_names[normalized_text(skill_name) or "missing"] += 1
                profile.redrob_numeric["assessment_score"].add(score, (40, 60, 75, 85, 95, 100))

    add_heap_example(profile.strong_examples, example_strength(candidate), candidate, example_limit)
    add_heap_example(profile.weak_examples, example_weakness(candidate), candidate, example_limit)

    suspicious = suspicious_reasons(candidate)
    if suspicious:
        add_heap_example(profile.suspicious_examples, len(suspicious), candidate, example_limit)

    stuffed = keyword_stuffing_reasons(candidate)
    if stuffed:
        add_heap_example(profile.keyword_examples, len(stuffed), candidate, example_limit)


def print_counter(title: str, counter: Counter[str], top_n: int) -> None:
    print(f"\n## {title}")
    for value, count in counter.most_common(top_n):
        print(f"{value}: {count}")


def print_numeric(title: str, summary: NumericSummary, top_n: int) -> None:
    avg = summary.average()
    avg_text = "n/a" if avg is None else f"{avg:.2f}"
    min_text = "n/a" if summary.minimum is None else f"{summary.minimum:.2f}"
    max_text = "n/a" if summary.maximum is None else f"{summary.maximum:.2f}"
    print(f"\n## {title}")
    print(f"count={summary.values_seen} mean={avg_text} min={min_text} max={max_text}")
    for value, count in summary.buckets.most_common(top_n):
        print(f"{value}: {count}")


def summarize_candidate(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = [
        normalized_text(skill.get("name"))
        for skill in candidate.get("skills", [])
        if isinstance(skill, dict)
    ][:8]
    return (
        f"{candidate.get('candidate_id')} | {profile.get('current_title')} | "
        f"{profile.get('years_of_experience')} yrs | {profile.get('location')}, "
        f"{profile.get('country')} | skills={skills} | open_to_work="
        f"{signals.get('open_to_work_flag')} response={signals.get('recruiter_response_rate')} "
        f"notice={signals.get('notice_period_days')}d"
    )


def print_examples(
    title: str,
    examples: list[tuple[float, str, dict[str, Any]]],
    reason_func: Any | None = None,
) -> None:
    print(f"\n## {title}")
    if not examples:
        print("No examples found.")
        return
    for score, _candidate_id, candidate in sorted(examples, reverse=True):
        print(f"- score={score:.2f} {summarize_candidate(candidate)}")
        if reason_func:
            reasons = reason_func(candidate)
            if reasons:
                print(f"  reasons: {'; '.join(reasons)}")


def print_report(profile: DatasetProfile, top_n: int) -> None:
    print("# Candidate Dataset Profile")
    print(f"\nTotal candidates: {profile.total}")

    print_counter("current_title distribution", profile.current_titles, top_n)
    print_counter("current_industry distribution", profile.current_industries, top_n)
    print_counter("years_of_experience buckets", profile.years_buckets, top_n)
    print_counter("country distribution", profile.countries, top_n)
    print_counter("location distribution", profile.locations, top_n)
    print_counter("current_company_size distribution", profile.company_sizes, top_n)
    print_counter("education tier distribution", profile.education_tiers, top_n)
    print_counter("skill name distribution", profile.skill_names, top_n)

    print("\n# Likely AI/ML title counts")
    if profile.ai_title_counts:
        for title, count in profile.ai_title_counts.most_common(top_n):
            print(f"{title}: {count}")
    else:
        print("No likely AI/ML titles found.")

    print("\n# Redrob Signal Distributions")
    for field_name in sorted(profile.redrob_categorical):
        print_counter(field_name, profile.redrob_categorical[field_name], top_n)
    for field_name in sorted(profile.redrob_numeric):
        print_numeric(field_name, profile.redrob_numeric[field_name], top_n)
    print_counter("skill_assessment_scores skill names", profile.redrob_assessment_names, top_n)

    print_examples("Likely strong profiles for manual inspection", profile.strong_examples)
    print_examples("Likely weak profiles for manual inspection", profile.weak_examples)
    print_examples(
        "Suspicious profiles / possible honeypots",
        profile.suspicious_examples,
        suspicious_reasons,
    )
    print_examples(
        "Possible keyword-stuffed candidates",
        profile.keyword_examples,
        keyword_stuffing_reasons,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    profile = DatasetProfile()
    for candidate in iter_candidates(args.input):
        update_profile(profile, candidate, args.examples)
        if profile.total % 10000 == 0:
            LOGGER.info("Profiled %d candidates", profile.total)

    print_report(profile, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
