"""Deterministic penalty detection for suspicious candidate profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.evidence_extractor import Evidence, EvidenceBundle, extract_evidence
from src.normalization import NormalizedCandidate


LOGGER = logging.getLogger(__name__)

REFERENCE_DATE = date(2026, 6, 22)

AI_SKILL_TERMS = {
    "ai",
    "machine learning",
    "ml",
    "nlp",
    "llm",
    "rag",
    "embedding",
    "embeddings",
    "vector",
    "retrieval",
    "ranking",
    "recommendation",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "langchain",
    "openai",
    "transformer",
    "tensorflow",
    "pytorch",
    "fine-tuning",
}

TECH_TITLE_TERMS = {
    "engineer",
    "developer",
    "scientist",
    "architect",
    "mlops",
    "data",
    "search",
    "ranking",
    "recommendation",
}

RESEARCH_ONLY_TERMS = {
    "research intern",
    "research assistant",
    "academic lab",
    "publication",
    "published papers",
    "pure research",
}

PRODUCTION_TERMS = {
    "production",
    "deployed",
    "shipped",
    "launched",
    "real users",
    "on-call",
    "owned",
    "operated",
}


@dataclass(frozen=True, slots=True)
class PenaltyResult:
    """Penalty details applied to a candidate score."""

    penalty_score: float
    penalty_reasons: tuple[str, ...]
    hard_flags: tuple[str, ...]
    soft_flags: tuple[str, ...]

    @property
    def total_penalty(self) -> float:
        """Backward-compatible alias for the initial skeleton."""
        return self.penalty_score

    @property
    def reasons(self) -> tuple[str, ...]:
        """Backward-compatible alias for the initial skeleton."""
        return self.penalty_reasons


def calculate_penalties(
    candidate: EvidenceBundle | Evidence | NormalizedCandidate | dict[str, Any],
    evidence: Evidence | None = None,
) -> PenaltyResult:
    """Calculate bounded penalties for suspicious or low-hireability profiles.

    The penalty score is normalized to 0-1 and capped at 0.75 so it can reduce a
    score materially without blindly eliminating a candidate. Downstream scoring
    can decide how aggressively to apply it.
    """
    raw = _raw_candidate(candidate)
    resolved_evidence = evidence or (candidate if isinstance(candidate, Evidence) else extract_evidence(raw))
    LOGGER.debug("Calculating penalties for candidate_id=%s", resolved_evidence.candidate_id)

    hard_flags: list[str] = []
    soft_flags: list[str] = []
    weighted_penalties: list[tuple[float, str, bool]] = []

    if raw:
        _check_expert_skill_duration(raw, weighted_penalties)
        _check_ai_keyword_stuffing(raw, resolved_evidence, weighted_penalties)
        _check_duration_consistency(raw, weighted_penalties)
        _check_profile_mismatch(raw, resolved_evidence, weighted_penalties)
        _check_pure_research(raw, resolved_evidence, weighted_penalties)
        _check_salary_expectation(raw, resolved_evidence, weighted_penalties)
        _check_identity_trust(raw, weighted_penalties)

    _check_recent_llm_demo(resolved_evidence, weighted_penalties)
    _check_services_only(resolved_evidence, weighted_penalties)
    _check_inactivity_and_response(raw, resolved_evidence, weighted_penalties)
    _check_notice_period(raw, resolved_evidence, weighted_penalties)

    for _weight, reason, is_hard in weighted_penalties:
        if is_hard:
            hard_flags.append(reason)
        else:
            soft_flags.append(reason)

    base_penalty = sum(weight for weight, _reason, _is_hard in weighted_penalties)
    severe_count = len(hard_flags)
    if severe_count >= 2:
        base_penalty += 0.08 * (severe_count - 1)

    penalty_score = round(min(0.75, base_penalty), 4)
    reasons = tuple(reason for _weight, reason, _is_hard in weighted_penalties)
    return PenaltyResult(
        penalty_score=penalty_score,
        penalty_reasons=reasons,
        hard_flags=tuple(dict.fromkeys(hard_flags)),
        soft_flags=tuple(dict.fromkeys(soft_flags)),
    )


def _raw_candidate(candidate: EvidenceBundle | Evidence | NormalizedCandidate | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, NormalizedCandidate):
        return candidate.raw_profile
    if isinstance(candidate, dict):
        return candidate
    return {}


def _check_expert_skill_duration(
    raw: dict[str, Any],
    penalties: list[tuple[float, str, bool]],
) -> None:
    expert_zero = 0
    expert_low = 0
    expert_ai_zero = 0
    for skill in _skills(raw):
        proficiency = _lower(skill.get("proficiency"))
        duration = _safe_float(skill.get("duration_months"))
        name = _text(skill.get("name"))
        if proficiency != "expert":
            continue
        if duration <= 0:
            expert_zero += 1
            if _is_ai_skill(name):
                expert_ai_zero += 1
        elif duration < 6:
            expert_low += 1

    if expert_zero:
        penalties.append((0.08 + min(0.10, expert_zero * 0.02), f"{expert_zero} expert skills have 0 months duration", True))
    if expert_low:
        penalties.append((0.04 + min(0.06, expert_low * 0.015), f"{expert_low} expert skills have under 6 months duration", False))
    if expert_ai_zero >= 2:
        penalties.append((0.10, f"{expert_ai_zero} expert AI skills have 0 months duration", True))


def _check_ai_keyword_stuffing(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    ai_skills = [skill for skill in _skills(raw) if _is_ai_skill(skill.get("name"))]
    title = _lower(_profile(raw).get("current_title"))
    has_technical_title = any(term in title for term in TECH_TITLE_TERMS)
    career_ai = max(
        evidence.value("production_ai_experience"),
        evidence.value("retrieval_search_ranking_experience"),
        evidence.value("pre_llm_ml_experience_signal"),
    )
    if len(ai_skills) >= 6 and career_ai < 0.35:
        penalties.append((0.14, f"{len(ai_skills)} AI skills but little AI career evidence", True))
    elif len(ai_skills) >= 4 and career_ai < 0.25:
        penalties.append((0.08, f"{len(ai_skills)} AI skills with weak supporting career evidence", False))
    if len(ai_skills) >= 4 and not has_technical_title:
        penalties.append((0.12, "AI keyword-heavy skills with unrelated current title", True))


def _check_duration_consistency(
    raw: dict[str, Any],
    penalties: list[tuple[float, str, bool]],
) -> None:
    profile_years = _safe_float(_profile(raw).get("years_of_experience"))
    career_months = 0.0
    current_roles = 0
    mismatched_roles = 0
    for role in _career(raw):
        duration = _safe_float(role.get("duration_months"))
        career_months += duration
        if role.get("is_current") is True:
            current_roles += 1
        inferred = _months_between(role.get("start_date"), role.get("end_date"))
        if inferred is not None and abs(inferred - duration) > 18:
            mismatched_roles += 1

    if current_roles != 1:
        penalties.append((0.08, f"{current_roles} current roles listed", True))
    if mismatched_roles:
        penalties.append((0.08, f"{mismatched_roles} role durations conflict with dates", True))
    if profile_years and career_months and abs(profile_years * 12 - career_months) > 42:
        penalties.append((0.10, "profile years_of_experience conflicts with career duration total", True))


def _check_profile_mismatch(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    profile = _profile(raw)
    current_title = _lower(profile.get("current_title"))
    summary = _lower(profile.get("summary"))
    current_role = next((role for role in _career(raw) if role.get("is_current") is True), None)
    current_role_text = ""
    if current_role:
        current_role_text = " ".join(
            [_lower(current_role.get("title")), _lower(current_role.get("description")), _lower(current_role.get("industry"))]
        )
    summary_ai = any(term in summary for term in AI_SKILL_TERMS)
    career_ai = any(term in current_role_text for term in AI_SKILL_TERMS)
    title_ai = evidence.value("ai_ml_title_strength") >= 0.5
    if summary_ai and not career_ai and not title_ai:
        penalties.append((0.08, "AI-heavy summary is not supported by current title or career history", False))
    if any(term in current_title for term in {"marketing", "hr", "sales", "finance", "accountant"}) and evidence.value("retrieval_search_ranking_experience") > 0.25:
        penalties.append((0.10, "business-function title conflicts with AI/search evidence", True))


def _check_recent_llm_demo(
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    demo_signal = evidence.value("llm_only_recent_demo_signal")
    if demo_signal >= 0.6:
        penalties.append((0.13, "recent LangChain/OpenAI demo signal without deeper ML history", True))
    elif demo_signal >= 0.25:
        penalties.append((0.06, "some recent LLM-demo signal", False))


def _check_services_only(
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    service_signal = evidence.value("services_only_penalty_signal")
    if service_signal >= 0.9:
        penalties.append((0.14, "services-only background conflicts with JD warning", True))
    elif service_signal >= 0.35:
        penalties.append((0.06, "notable IT services background", False))


def _check_pure_research(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    text = _candidate_text(raw)
    has_research = any(term in text for term in RESEARCH_ONLY_TERMS)
    has_production = any(term in text for term in PRODUCTION_TERMS) or evidence.value("production_ai_experience") >= 0.35
    if has_research and not has_production:
        penalties.append((0.14, "research-heavy profile without production deployment evidence", True))


def _check_inactivity_and_response(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    signals = _signals(raw)
    last_active = _parse_date(signals.get("last_active_date"))
    response_rate = _safe_float(signals.get("recruiter_response_rate"))
    if last_active:
        inactive_days = (REFERENCE_DATE - last_active).days
        if inactive_days > 180 and response_rate < 0.2:
            penalties.append((0.13, f"inactive for {inactive_days} days with low recruiter response", True))
        elif inactive_days > 90 and response_rate < 0.35:
            penalties.append((0.07, f"inactive for {inactive_days} days with weak recruiter response", False))
    elif evidence.value("availability_signal") < 0.25 and evidence.value("recruiter_engagement_signal") < 0.25:
        penalties.append((0.06, "weak availability and recruiter engagement signals", False))


def _check_salary_expectation(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    salary = _signals(raw).get("expected_salary_range_inr_lpa", {})
    if not isinstance(salary, dict):
        return
    min_salary = _safe_float(salary.get("min"))
    max_salary = _safe_float(salary.get("max"))
    fit = max(
        evidence.value("production_ai_experience"),
        evidence.value("retrieval_search_ranking_experience"),
        evidence.value("ai_ml_title_strength"),
    )
    if min_salary and max_salary and min_salary > max_salary:
        penalties.append((0.08, "expected salary min is greater than max", True))
    elif max_salary >= 75 and fit < 0.45:
        penalties.append((0.07, "very high salary expectation with weak role fit", False))


def _check_notice_period(
    raw: dict[str, Any],
    evidence: Evidence,
    penalties: list[tuple[float, str, bool]],
) -> None:
    notice = _safe_float(_signals(raw).get("notice_period_days"))
    if not notice:
        notice = 180 * (1.0 - evidence.value("notice_period_fit"))
    if notice > 90:
        penalties.append((0.09, f"notice period is above 90 days ({notice:g})", True))
    elif notice > 60:
        penalties.append((0.05, f"notice period is above 60 days ({notice:g})", False))


def _check_identity_trust(
    raw: dict[str, Any],
    penalties: list[tuple[float, str, bool]],
) -> None:
    signals = _signals(raw)
    completeness = _safe_float(signals.get("profile_completeness_score"))
    missing_identity = sum(
        1
        for field in ("verified_email", "verified_phone", "linkedin_connected")
        if signals.get(field) is not True
    )
    if missing_identity == 3 and completeness < 60:
        penalties.append((0.10, "no verified email/phone/linkedin with low profile completeness", True))
    elif missing_identity >= 2 and completeness < 70:
        penalties.append((0.05, "weak identity verification and profile completeness", False))


def _profile(raw: dict[str, Any]) -> dict[str, Any]:
    profile = raw.get("profile", {})
    return profile if isinstance(profile, dict) else {}


def _signals(raw: dict[str, Any]) -> dict[str, Any]:
    signals = raw.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _career(raw: dict[str, Any]) -> list[dict[str, Any]]:
    career = raw.get("career_history", [])
    return [item for item in career if isinstance(item, dict)] if isinstance(career, list) else []


def _skills(raw: dict[str, Any]) -> list[dict[str, Any]]:
    skills = raw.get("skills", [])
    return [item for item in skills if isinstance(item, dict)] if isinstance(skills, list) else []


def _candidate_text(raw: dict[str, Any]) -> str:
    profile = _profile(raw)
    parts = [_lower(profile.get("headline")), _lower(profile.get("summary")), _lower(profile.get("current_title"))]
    for role in _career(raw):
        parts.extend([_lower(role.get("title")), _lower(role.get("description")), _lower(role.get("industry"))])
    return " ".join(part for part in parts if part)


def _is_ai_skill(name: Any) -> bool:
    lowered = _lower(name)
    return any(term in lowered for term in AI_SKILL_TERMS)


def _months_between(start: Any, end: Any) -> int | None:
    start_date = _parse_date(start)
    end_date = _parse_date(end) or REFERENCE_DATE
    if not start_date:
        return None
    return max(0, (end_date.year - start_date.year) * 12 + end_date.month - start_date.month)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _lower(value: Any) -> str:
    return _text(value).lower()
