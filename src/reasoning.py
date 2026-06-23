"""Human-readable, evidence-grounded reasoning generation."""

from __future__ import annotations

import logging
from typing import Any

from src.evidence_extractor import EvidenceBundle
from src.penalties import PenaltyResult


LOGGER = logging.getLogger(__name__)

STRONG_OPENERS = (
    "High-confidence fit",
    "Strong fit",
    "Compelling fit",
    "Clear fit",
)

MID_OPENERS = (
    "Solid fit",
    "Relevant fit",
    "Credible fit",
    "Good fit",
)

CAUTIOUS_OPENERS = (
    "Cautious fit",
    "Borderline fit",
    "Lower-confidence fit",
    "Possible fit",
)

CONCERN_PREFIXES = (
    "Concern:",
    "Watch-out:",
    "Caveat:",
    "Tradeoff:",
)


def build_reasoning(
    evidence: EvidenceBundle,
    penalties: PenaltyResult,
    rank: int | None = None,
    component_scores: dict[str, float] | None = None,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Build concise recruiter-ready reasoning from supported evidence only."""
    LOGGER.debug("Building reasoning for candidate_id=%s", evidence.candidate_id)
    component_scores = component_scores or {}
    facts = _candidate_facts(candidate or {})
    opener = _opener(evidence.candidate_id, rank)

    positive_parts = _positive_parts(evidence, facts, component_scores)
    concern_parts = _concern_parts(evidence, penalties, facts, rank)

    if not positive_parts:
        positive_parts.append("profile has limited explicit evidence for the Senior AI Engineer brief")

    first_sentence = f"{opener}: {', '.join(positive_parts[:3])}."
    if concern_parts:
        prefix = CONCERN_PREFIXES[_stable_index(evidence.candidate_id, len(CONCERN_PREFIXES), offset=3)]
        second_sentence = f"{prefix} {concern_parts[0]}."
        return _csv_safe(f"{first_sentence} {second_sentence}")
    return _csv_safe(first_sentence)


def _positive_parts(
    evidence: EvidenceBundle,
    facts: dict[str, Any],
    component_scores: dict[str, float],
) -> list[str]:
    parts: list[str] = []
    title = facts.get("title")
    years = evidence.value("total_years_experience")
    if title and years:
        parts.append(f"{title} with {years:g} years")
    elif title:
        parts.append(str(title))
    elif years:
        parts.append(f"{years:g} years of experience")

    if evidence.value("retrieval_search_ranking_experience") >= 0.65:
        parts.append(_supported_domain_phrase(evidence, "retrieval_search_ranking_experience"))
    elif evidence.value("data_engineering_adjacent_strength") >= 0.65:
        parts.append("strong data/ML systems adjacency")

    if evidence.value("production_ai_experience") >= 0.65:
        parts.append("production ML/AI delivery evidence")
    elif evidence.value("recent_hands_on_coding_signal") >= 0.65:
        parts.append("recent hands-on technical work")

    if evidence.value("evaluation_framework_experience") >= 0.45:
        parts.append("ranking/evaluation framework evidence")

    if evidence.value("python_strength") >= 0.55 and _has_skill(facts, "python"):
        parts.append("Python is explicitly listed")

    product_company = facts.get("product_company")
    if evidence.value("product_company_experience") >= 0.5 and product_company:
        parts.append(f"product-company experience at {product_company}")

    if evidence.value("redrob_availability_signal", 0.0) >= 0.5:
        parts.append("strong Redrob availability")
    elif evidence.value("availability_signal") >= 0.65:
        parts.append("recent Redrob availability signal")
    elif component_scores.get("redrob_availability_score", 0.0) >= 0.65:
        parts.append("strong Redrob engagement")

    return _dedupe(parts)


def _concern_parts(
    evidence: EvidenceBundle,
    penalties: PenaltyResult,
    facts: dict[str, Any],
    rank: int | None,
) -> list[str]:
    concerns: list[str] = []
    if penalties.hard_flags:
        concerns.append(_humanize_flag(penalties.hard_flags[0]))
    elif penalties.soft_flags:
        concerns.append(_humanize_flag(penalties.soft_flags[0]))

    notice = facts.get("notice_period_days")
    if notice is not None and notice > 60:
        concerns.append(f"notice period is {notice:g} days")

    if evidence.value("llm_only_recent_demo_signal") >= 0.25:
        concerns.append("LLM evidence looks more demo-oriented than deep production ML")
    if evidence.value("services_only_penalty_signal") >= 0.35:
        concerns.append("IT-services-heavy background needs scrutiny against the JD")
    if evidence.value("recruiter_engagement_signal") < 0.3:
        concerns.append("Redrob recruiter engagement is weak")
    if rank is not None and rank >= 80 and not concerns:
        concerns.append("weaker explicit ranking or production ML evidence keeps them near the cutoff")
    return _dedupe(concerns)


def _supported_domain_phrase(evidence: EvidenceBundle, signal_name: str) -> str:
    snippets = " ".join(evidence.signals[signal_name].snippets).lower()
    if "recommend" in snippets:
        return "direct recommendation-systems evidence"
    if "ranking" in snippets or "ranker" in snippets:
        return "direct ranking-systems evidence"
    if "search" in snippets:
        return "direct search/retrieval evidence"
    if "retrieval" in snippets:
        return "direct retrieval-systems evidence"
    return "direct retrieval/search/ranking evidence"


def _opener(candidate_id: str, rank: int | None) -> str:
    if rank is not None and rank <= 10:
        options = STRONG_OPENERS
        offset = 0
    elif rank is not None and rank >= 80:
        options = CAUTIOUS_OPENERS
        offset = 7
    else:
        options = MID_OPENERS
        offset = 13
    return options[_stable_index(candidate_id, len(options), offset=offset)]


def _candidate_facts(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    signals = candidate.get("redrob_signals", {}) if isinstance(candidate.get("redrob_signals", {}), dict) else {}
    skills = candidate.get("skills", []) if isinstance(candidate.get("skills", []), list) else []
    career = candidate.get("career_history", []) if isinstance(candidate.get("career_history", []), list) else []
    skill_names = [
        str(skill.get("name", "")).strip().lower()
        for skill in skills
        if isinstance(skill, dict) and str(skill.get("name", "")).strip()
    ]
    product_company = None
    for role in career:
        if not isinstance(role, dict):
            continue
        industry = str(role.get("industry", "")).strip().lower()
        if industry in {"software", "fintech", "food delivery", "transportation", "saas", "marketplace"}:
            product_company = str(role.get("company", "")).strip() or None
            break
    return {
        "title": str(profile.get("current_title", "")).strip(),
        "skill_names": skill_names,
        "product_company": product_company,
        "notice_period_days": _safe_float_or_none(signals.get("notice_period_days")),
    }


def _has_skill(facts: dict[str, Any], skill_name: str) -> bool:
    wanted = skill_name.lower()
    return any(wanted == skill or wanted in skill for skill in facts.get("skill_names", []))


def _humanize_flag(flag: str) -> str:
    text = flag.strip()
    if not text:
        return "profile has a verification or consistency concern"
    return text[0].lower() + text[1:]


def _dedupe(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = " ".join(part.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(part)
    return result


def _stable_index(candidate_id: str, modulo: int, offset: int = 0) -> int:
    total = sum(ord(char) for char in candidate_id) + offset
    return total % modulo


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_safe(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())[:450]
