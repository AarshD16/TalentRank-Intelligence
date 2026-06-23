"""Transparent weighted scoring model for candidate ranking."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.config import RankingConfig
from src.evidence_extractor import Evidence, extract_evidence
from src.penalties import PenaltyResult, calculate_penalties
from src.reasoning import build_reasoning


LOGGER = logging.getLogger(__name__)


COMPONENT_NAMES = (
    "role_fit_score",
    "production_ml_score",
    "retrieval_ranking_score",
    "evaluation_score",
    "experience_score",
    "company_context_score",
    "skill_quality_score",
    "redrob_availability_score",
    "trust_consistency_score",
    "logistics_score",
)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Component scores and final score for one candidate."""

    candidate_id: str
    final_score: float
    component_scores: dict[str, float]
    evidence: Evidence
    penalties: PenaltyResult


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Final row written to submission.csv."""

    candidate_id: str
    rank: int
    score: float
    reasoning: str
    component_scores: dict[str, float] = field(default_factory=dict)


def score_candidate(profile: dict[str, Any], config: RankingConfig) -> float:
    """Return the deterministic 0-1 score for one candidate."""
    return score_candidate_with_components(profile, config).final_score


def score_candidate_with_components(
    profile: dict[str, Any],
    config: RankingConfig,
) -> ScoreBreakdown:
    """Score one candidate and retain transparent component scores."""
    evidence = extract_evidence(profile)
    penalties = calculate_penalties(profile, evidence=evidence)
    components = {
        "role_fit_score": _role_fit_score(evidence),
        "production_ml_score": _production_ml_score(evidence),
        "retrieval_ranking_score": _retrieval_ranking_score(evidence),
        "evaluation_score": _evaluation_score(evidence),
        "experience_score": _experience_score(evidence),
        "company_context_score": _company_context_score(evidence),
        "skill_quality_score": _skill_quality_score(evidence),
        "redrob_availability_score": _redrob_availability_score(evidence),
        "trust_consistency_score": _trust_consistency_score(evidence),
        "logistics_score": _logistics_score(evidence),
    }
    final_score = _combine_components(components, config, penalties)
    LOGGER.debug(
        "Scored candidate_id=%s final_score=%.6f components=%s",
        evidence.candidate_id,
        final_score,
        components,
    )
    return ScoreBreakdown(
        candidate_id=evidence.candidate_id,
        final_score=final_score,
        component_scores=components,
        evidence=evidence,
        penalties=penalties,
    )


def rank_candidates(
    candidates: Sequence[dict[str, Any]],
    config: RankingConfig,
) -> list[RankedCandidate]:
    """Rank candidates and return exactly config.top_k rows."""
    LOGGER.info("Ranking %d candidates", len(candidates))
    scored = [score_candidate_with_components(candidate, config) for candidate in candidates]
    scored.sort(key=lambda item: (-item.final_score, item.candidate_id))
    top = scored[: config.top_k]
    if len(top) != config.top_k:
        raise ValueError(f"Need at least {config.top_k} candidates to produce a valid submission.")

    ranked: list[RankedCandidate] = []
    for index, item in enumerate(top, start=1):
        ranked.append(
            RankedCandidate(
                candidate_id=item.candidate_id,
                rank=index,
                score=item.final_score,
                reasoning=build_reasoning(
                    evidence=item.evidence,
                    penalties=item.penalties,
                    rank=index,
                    component_scores=item.component_scores,
                    candidate=_candidate_by_id(candidates, item.candidate_id),
                ),
                component_scores=item.component_scores,
            )
        )
    return ranked


def _role_fit_score(evidence: Evidence) -> float:
    """Role match from title relevance, seniority, and hands-on signal."""
    value = (
        evidence.value("ai_ml_title_strength") * 0.48
        + evidence.value("seniority_strength") * 0.24
        + evidence.value("recent_hands_on_coding_signal") * 0.20
        + evidence.value("nlp_ir_relevance") * 0.08
    )
    value -= evidence.value("llm_only_recent_demo_signal") * 0.18
    value -= evidence.value("services_only_penalty_signal") * 0.08
    return _clamp(value)


def _production_ml_score(evidence: Evidence) -> float:
    """Production ML systems depth, not generic AI curiosity."""
    value = (
        evidence.value("production_ai_experience") * 0.62
        + evidence.value("pre_llm_ml_experience_signal") * 0.18
        + evidence.value("data_engineering_adjacent_strength") * 0.12
        + evidence.value("recent_hands_on_coding_signal") * 0.08
    )
    value -= evidence.value("llm_only_recent_demo_signal") * 0.22
    return _clamp(value)


def _retrieval_ranking_score(evidence: Evidence) -> float:
    """Search, ranking, recommendation, retrieval, and vector infra evidence."""
    value = (
        evidence.value("retrieval_search_ranking_experience") * 0.55
        + evidence.value("vector_database_experience") * 0.25
        + evidence.value("nlp_ir_relevance") * 0.20
    )
    # If retrieval-like terms only appear in weak role contexts, cap their impact.
    if evidence.value("ai_ml_title_strength") < 0.35 and evidence.value("production_ai_experience") < 0.35:
        value = min(value, 0.35)
    return _clamp(value)


def _evaluation_score(evidence: Evidence) -> float:
    """Ranking evaluation framework experience."""
    value = evidence.value("evaluation_framework_experience")
    if value > 0 and evidence.value("production_ai_experience") >= 0.5:
        value = min(1.0, value + 0.12)
    return _clamp(value)


def _experience_score(evidence: Evidence) -> float:
    """Fit to the JD's 5-9 year preference with room for strong adjacent evidence."""
    years = evidence.value("total_years_experience")
    core_evidence = max(
        evidence.value("production_ai_experience"),
        evidence.value("retrieval_search_ranking_experience"),
    )
    if 5 <= years <= 9:
        value = 1.0
    elif 4 <= years < 5:
        value = 0.72 + core_evidence * 0.15
    elif 9 < years <= 12:
        value = 0.72 + core_evidence * 0.12
    elif 12 < years <= 16:
        value = 0.42 + core_evidence * 0.18
    elif 3 <= years < 4:
        value = 0.35 + core_evidence * 0.15
    else:
        value = 0.2 + core_evidence * 0.1
    return _clamp(value)


def _company_context_score(evidence: Evidence) -> float:
    """Product-company context, with services-only career down-weighting."""
    value = (
        evidence.value("product_company_experience") * 0.75
        + (1.0 - evidence.value("services_only_penalty_signal")) * 0.25
    )
    return _clamp(value)


def _skill_quality_score(evidence: Evidence) -> float:
    """Quality of relevant technical skills without rewarding raw skill count."""
    value = (
        evidence.value("python_strength") * 0.32
        + evidence.value("production_ai_experience") * 0.22
        + evidence.value("retrieval_search_ranking_experience") * 0.18
        + evidence.value("vector_database_experience") * 0.12
        + evidence.value("data_engineering_adjacent_strength") * 0.10
        + evidence.value("open_source_or_github_signal") * 0.06
    )
    value -= evidence.value("llm_only_recent_demo_signal") * 0.12
    return _clamp(value)


def _redrob_availability_score(evidence: Evidence) -> float:
    """Practical hireability from Redrob behavioral signals."""
    return _clamp(
        evidence.value("availability_signal") * 0.45
        + evidence.value("recruiter_engagement_signal") * 0.35
        + evidence.value("notice_period_fit") * 0.10
        + evidence.value("profile_trust_signal") * 0.10
    )


def _trust_consistency_score(evidence: Evidence) -> float:
    """Profile reliability and anti-honeypot/anti-keyword-stuffing pressure."""
    value = (
        evidence.value("profile_trust_signal") * 0.55
        + (1.0 - evidence.value("llm_only_recent_demo_signal")) * 0.20
        + (1.0 - evidence.value("services_only_penalty_signal")) * 0.15
        + evidence.value("recent_hands_on_coding_signal") * 0.10
    )
    return _clamp(value)


def _logistics_score(evidence: Evidence) -> float:
    """Location, notice period, and availability fit."""
    return _clamp(
        evidence.value("location_fit") * 0.55
        + evidence.value("notice_period_fit") * 0.35
        + evidence.value("availability_signal") * 0.10
    )


def _combine_components(
    components: dict[str, float],
    config: RankingConfig,
    penalties: PenaltyResult | None = None,
) -> float:
    weights = config.resolved_scoring_weights()
    missing = set(COMPONENT_NAMES) - set(weights)
    if missing:
        raise ValueError(f"Missing scoring weights for: {sorted(missing)}")
    score = sum(components[name] * weights[name] for name in COMPONENT_NAMES)
    if penalties is not None:
        score *= 1.0 - min(0.75, penalties.penalty_score) * 0.55
    return round(_clamp(score), 6)


def _build_scoring_reasoning(score: ScoreBreakdown) -> str:
    top_components = sorted(
        score.component_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    component_text = ", ".join(f"{name.replace('_score', '')}={value:.2f}" for name, value in top_components)
    retrieval = score.evidence.signals["retrieval_search_ranking_experience"].snippets[:1]
    production = score.evidence.signals["production_ai_experience"].snippets[:1]
    snippet = (retrieval or production or ("Transparent evidence score from profile, career, skills, and Redrob signals.",))[0]
    return f"{component_text}. Evidence: {snippet}"


def _candidate_by_id(candidates: Sequence[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if str(candidate.get("candidate_id", "")).strip() == candidate_id:
            return candidate
    return {}


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
