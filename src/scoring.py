"""Transparent weighted scoring model for candidate ranking."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.candidate_evidence import CandidateEvidence, extract_candidate_evidence
from src.config import RankingConfig
from src.evidence_extractor import Evidence, extract_evidence
from src.penalties import PenaltyResult, calculate_penalties
from src.reasoning import build_reasoning, build_rolespec_reasoning
from src.rolespec import RoleSpec, default_redrob_ai_engineer_rolespec


LOGGER = logging.getLogger(__name__)


COMPONENT_NAMES = (
    "capability_score",
    "redrob_hireability_score",
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
    candidate_evidence: CandidateEvidence | None = None
    competency_scores: dict[str, float] = field(default_factory=dict)
    missing_must_have: tuple[str, ...] = ()
    score_flags: tuple[str, ...] = ()


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
    rolespec: RoleSpec | None = None,
) -> ScoreBreakdown:
    """Score one candidate and retain transparent component scores."""
    rolespec = rolespec or default_redrob_ai_engineer_rolespec()
    candidate_evidence = extract_candidate_evidence(profile)
    evidence = candidate_evidence.legacy_evidence
    penalties = calculate_penalties(profile, evidence=evidence)
    capability = _capability_score(profile, candidate_evidence, rolespec)
    hireability = _redrob_hireability_score(profile, evidence)
    trust = _redrob_trust_consistency_score(profile, evidence, penalties)
    logistics = _logistics_score(evidence)
    components = {
        "capability_score": capability,
        "redrob_hireability_score": hireability,
        "trust_consistency_score": trust,
        "logistics_score": logistics,
    }
    competency_scores = {
        competency: candidate_evidence.score(competency)
        for competency in rolespec.scoring_weights
    }
    final_score, score_flags = _final_capability_hireability_score(
        profile,
        candidate_evidence,
        rolespec,
        components,
        penalties,
    )
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
        candidate_evidence=candidate_evidence,
        competency_scores=competency_scores,
        missing_must_have=_missing_must_have(candidate_evidence, rolespec),
        score_flags=score_flags,
    )


def rank_candidates(
    candidates: Sequence[dict[str, Any]],
    config: RankingConfig,
) -> list[RankedCandidate]:
    """Rank candidates and return exactly config.top_k rows."""
    LOGGER.info("Ranking %d candidates", len(candidates))
    rolespec = default_redrob_ai_engineer_rolespec()
    scored = [score_candidate_with_components(candidate, config, rolespec=rolespec) for candidate in candidates]
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
                reasoning=(
                    build_rolespec_reasoning(
                        candidate_evidence=item.candidate_evidence,
                        rolespec=rolespec,
                        penalties=item.penalties,
                        rank=index,
                        candidate=_candidate_by_id(candidates, item.candidate_id),
                    )
                    if item.candidate_evidence is not None
                    else build_reasoning(
                        evidence=item.evidence,
                        penalties=item.penalties,
                        rank=index,
                        component_scores=item.component_scores,
                        candidate=_candidate_by_id(candidates, item.candidate_id),
                    )
                ),
                component_scores=item.component_scores,
            )
        )
    return ranked


def _capability_score(profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    """Primary JD capability fit score; Redrob behavior is deliberately excluded."""
    evidence = candidate_evidence.legacy_evidence
    title = _role_title_match_score(profile, candidate_evidence, rolespec)
    experience = _capability_experience_score(candidate_evidence.years_experience, evidence)
    competency_match = _weighted_competency_match(candidate_evidence, rolespec)
    evidence_fit = _evidence_requirement_fit(candidate_evidence, rolespec)
    scale = _production_scale_score(profile)
    seniority = _seniority_match_score(candidate_evidence, rolespec)

    value = (
        competency_match * 0.52
        + title * 0.16
        + experience * 0.12
        + evidence_fit * 0.12
        + seniority * 0.04
        + scale * 0.04
    )
    if title < 0.35 and evidence_fit < 0.75:
        value = min(value, 0.48)
    if candidate_evidence.score("keyword_stuffing") >= 0.5 and _career_ai_search_evidence(evidence) < 0.45:
        value = min(value, 0.38)
    if _rolespec_recall_gate(candidate_evidence, rolespec):
        value = max(value, 0.88)
    elif _exceptional_near_band(candidate_evidence, rolespec):
        value = max(value, 0.80)
    return _clamp(value)


def _final_capability_hireability_score(
    profile: dict[str, Any],
    candidate_evidence: CandidateEvidence,
    rolespec: RoleSpec,
    components: dict[str, float],
    penalties: PenaltyResult,
) -> tuple[float, tuple[str, ...]]:
    evidence = candidate_evidence.legacy_evidence
    capability = components["capability_score"]
    hireability = components["redrob_hireability_score"]
    trust = components["trust_consistency_score"]
    logistics = components["logistics_score"]
    hireability_min = float(rolespec.redrob_signal_policy.get("hireability_multiplier_min", 0.85))
    hireability_max = float(rolespec.redrob_signal_policy.get("hireability_multiplier_max", 1.15))
    trust_min = float(rolespec.redrob_signal_policy.get("trust_multiplier_min", 0.95))
    trust_max = float(rolespec.redrob_signal_policy.get("trust_multiplier_max", 1.05))
    hireability_multiplier = hireability_min + hireability * (hireability_max - hireability_min)
    trust_multiplier = trust_min + trust * (trust_max - trust_min)
    risk_penalty = _risk_penalty(profile, candidate_evidence, rolespec, penalties)
    raw_score = capability * hireability_multiplier * trust_multiplier - risk_penalty
    complete_match_score = _complete_match_bundle_score(profile, candidate_evidence, rolespec)
    high_recall_gate = _high_recall_safety_gate(profile, candidate_evidence, rolespec)
    if complete_match_score >= 0.98:
        raw_score = max(raw_score + 0.055, 0.965 + min(0.025, complete_match_score * 0.025))
    elif high_recall_gate:
        raw_score = max(raw_score + 0.030, 0.925)

    flags: list[str] = []
    capped_score, cap_flags = _apply_score_caps(profile, candidate_evidence, rolespec, raw_score)
    flags.extend(cap_flags)
    if hireability_multiplier > 1.05:
        flags.append("hireability_boost")
    if _inactive_low_response(profile):
        flags.append("low_activity_response_demotion")
    if _rolespec_recall_gate(candidate_evidence, rolespec):
        flags.append("recall_safety_gate")
    if high_recall_gate:
        flags.append("high_recall_safety_gate")
    if complete_match_score >= 0.98:
        flags.append("complete_match_bundle")
    if capped_score > 0.95:
        capped_score = 0.95 + (capped_score - 0.95) * 0.35
    capped_score = min(capped_score, 0.9975)
    return round(_clamp(capped_score), 6), tuple(flags)


def _redrob_hireability_score(profile: dict[str, Any], evidence: Evidence) -> float:
    signals = _signals(profile)
    response_rate = _safe_float(signals.get("recruiter_response_rate"), default=0.0)
    response_time = _safe_float(signals.get("avg_response_time_hours"), default=168.0)
    interview_completion = _safe_float(signals.get("interview_completion_rate"), default=0.0)
    offer_acceptance = _safe_float(signals.get("offer_acceptance_rate"), default=-1.0)
    applications = _safe_float(signals.get("applications_submitted_30d"), default=0.0)
    active_score = _last_active_score(signals.get("last_active_date"))
    notice_score = _notice_score(_safe_float(signals.get("notice_period_days"), default=90.0))
    response_time_score = 1.0 if response_time <= 24 else 0.78 if response_time <= 72 else 0.45 if response_time <= 168 else 0.15
    open_to_work = 1.0 if signals.get("open_to_work_flag") is True else 0.35
    offer_score = offer_acceptance if offer_acceptance >= 0 else 0.50
    application_score = 1.0 - min(1.0, math.log1p(applications) / math.log1p(40.0)) * 0.35
    market_tie_breaker = _market_validation_score(signals) * 0.05
    return _clamp(
        response_rate * 0.24
        + response_time_score * 0.16
        + active_score * 0.16
        + open_to_work * 0.10
        + notice_score * 0.12
        + interview_completion * 0.10
        + offer_score * 0.07
        + application_score * 0.05
        + market_tie_breaker
    )


def _redrob_trust_consistency_score(
    profile: dict[str, Any],
    evidence: Evidence,
    penalties: PenaltyResult,
) -> float:
    signals = _signals(profile)
    completeness = min(1.0, _safe_float(signals.get("profile_completeness_score"), default=0.0) / 100.0)
    verified = sum(1 for field in ("verified_email", "verified_phone", "linkedin_connected") if signals.get(field) is True) / 3.0
    github = min(1.0, max(0.0, _safe_float(signals.get("github_activity_score"), default=0.0)) / 80.0)
    assessment = _skill_assessment_score(signals.get("skill_assessment_scores"))
    endorsements = min(1.0, math.log1p(_safe_float(signals.get("endorsements_received"), default=0.0)) / math.log1p(180.0))
    inconsistency = min(1.0, penalties.penalty_score + evidence.value("llm_only_recent_demo_signal") * 0.35)
    return _clamp(
        completeness * 0.26
        + verified * 0.24
        + github * 0.14
        + assessment * 0.16
        + endorsements * 0.10
        + (1.0 - inconsistency) * 0.10
    )


def _risk_penalty(
    profile: dict[str, Any],
    candidate_evidence: CandidateEvidence,
    rolespec: RoleSpec,
    penalties: PenaltyResult,
) -> float:
    evidence = candidate_evidence.legacy_evidence
    risk = min(0.18, penalties.penalty_score * 0.12)
    for signal in rolespec.negative_signals:
        risk += min(0.06, _negative_signal_score(signal, profile, candidate_evidence, rolespec) * 0.06)
    if _inactive_low_response(profile):
        risk += 0.035
    return min(0.22, risk)


def _apply_score_caps(
    profile: dict[str, Any],
    candidate_evidence: CandidateEvidence,
    rolespec: RoleSpec,
    score: float,
) -> tuple[float, tuple[str, ...]]:
    flags: list[str] = []
    capped = score
    for cap in rolespec.score_caps:
        if _score_cap_applies(cap.rule, profile, candidate_evidence, rolespec):
            capped = min(capped, cap.max_score)
            flags.append(f"cap_{cap.rule}")
    return capped, tuple(flags)


def _capability_experience_score(years: float, evidence: Evidence) -> float:
    exceptional = min(
        evidence.value("semantic_production_retrieval_score"),
        max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")),
    )
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0:
        return 0.72 if exceptional >= 0.85 else 0.48
    if 9.0 < years <= 12.0:
        return 0.68
    if 3.0 <= years < 4.0:
        return 0.28 if exceptional >= 0.9 else 0.18
    if years < 3.0:
        return 0.08
    return 0.38


def _weighted_competency_match(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    if not rolespec.scoring_weights:
        return 0.0
    total = 0.0
    for competency, weight in rolespec.scoring_weights.items():
        total += _role_aware_competency_score(candidate_evidence, competency, rolespec) * weight
    return _clamp(total)


def _role_aware_competency_score(candidate_evidence: CandidateEvidence, competency: str, rolespec: RoleSpec) -> float:
    base = candidate_evidence.score(competency)
    evidence = candidate_evidence.legacy_evidence
    if competency == "ai_ml_engineering":
        return _clamp(max(base, evidence.value("ai_ml_title_strength"), evidence.value("production_ai_experience") * 0.85))
    if competency in {"retrieval_search", "ranking_recommendation"}:
        return _retrieval_ranking_depth(evidence)
    if competency == "production_ml_systems":
        return _production_retrieval_depth(evidence)
    if competency == "ranking_evaluation":
        return _ranking_evaluation_depth(evidence)
    if competency == "vector_hybrid_search":
        return _career_backed_vector_depth(evidence)
    if competency == "python_engineering":
        return _python_ml_engineering_depth(evidence)
    if competency == "backend_systems":
        return _backend_depth(candidate_evidence)
    if competency == "product_management":
        return _product_management_depth(candidate_evidence, rolespec)
    return base


def _evidence_requirement_fit(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    if not rolespec.evidence_requirements:
        return 1.0
    met = 0.0
    for requirement in rolespec.evidence_requirements:
        item = candidate_evidence.competencies.get(requirement.name)
        score = _role_aware_competency_score(candidate_evidence, requirement.name, rolespec)
        if item is None:
            continue
        source_ok = _requirement_source_ok(requirement.name, requirement.required_sources, candidate_evidence, item.source_field)
        if score >= requirement.min_score and source_ok:
            met += 1.0
        elif score >= requirement.min_score:
            met += 0.65
        else:
            met += min(0.5, score / max(requirement.min_score, 0.01) * 0.5)
    return _clamp(met / len(rolespec.evidence_requirements))


def _requirement_source_ok(
    competency: str,
    required_sources: tuple[str, ...],
    candidate_evidence: CandidateEvidence,
    source_field: str,
) -> bool:
    if not required_sources:
        return True
    if any(required in source_field for required in required_sources):
        return True
    if "career_history" not in required_sources:
        return False
    evidence = candidate_evidence.legacy_evidence
    if competency in {"ranking_recommendation", "retrieval_search"}:
        return _career_backed_score(evidence, ("retrieval_search_ranking_experience",)) >= 0.35
    if competency == "ranking_evaluation":
        return _career_backed_score(evidence, ("evaluation_framework_experience",)) >= 0.35
    if competency == "production_ml_systems":
        return _production_career_evidence(evidence) >= 0.35
    if competency == "vector_hybrid_search":
        return _source_quality_signal(evidence, ("vector_database_experience", "retrieval_search_ranking_experience")) >= 0.35
    return False


def _role_title_match_score(profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    title = _profile_title(profile)
    role_text = f"{rolespec.role_title} {rolespec.role_family}".lower()
    if not title:
        return 0.0
    title_lower = title.lower()
    if any(term in title_lower for term in _role_title_terms(role_text)):
        return 1.0
    if rolespec.role_family in {"ai_ml_search", "ai_ml", "search"}:
        return _relevant_title_score(candidate_evidence)
    if "backend" in role_text and any(term in title_lower for term in ("backend", "software engineer", "api engineer", "platform engineer")):
        return 0.95
    if "product" in role_text and "product" in title_lower and "manager" in title_lower:
        return 0.95
    if rolespec.role_family in title_lower:
        return 0.85
    return min(0.65, candidate_evidence.score(_primary_role_competency(rolespec)) * 0.75)


def _role_title_terms(role_text: str) -> tuple[str, ...]:
    terms = []
    for phrase in (
        "senior ai engineer",
        "lead ai engineer",
        "recommendation systems engineer",
        "search engineer",
        "applied ml engineer",
        "machine learning engineer",
        "nlp engineer",
        "backend engineer",
        "software engineer",
        "product manager",
    ):
        if phrase in role_text:
            terms.append(phrase.replace("senior ", "").replace("lead ", ""))
    return tuple(terms)


def _seniority_match_score(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    target = rolespec.target_seniority.lower()
    level = candidate_evidence.seniority_level
    if target in level:
        return 1.0
    if target == "senior" and level == "lead_plus":
        return 0.85
    if target in {"lead", "lead_plus"} and level in {"senior", "lead_plus"}:
        return 0.85
    if target == "mid" and level in {"mid", "senior"}:
        return 0.85
    return 0.45


def _primary_role_competency(rolespec: RoleSpec) -> str:
    if rolespec.must_have_competencies:
        return rolespec.must_have_competencies[0]
    if rolespec.scoring_weights:
        return max(rolespec.scoring_weights, key=rolespec.scoring_weights.get)
    return "backend_systems"


def _backend_depth(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    text_score = _text_signal_score(evidence, ("api", "backend", "microservice", "distributed", "database", "latency"))
    return _clamp(max(candidate_evidence.score("backend_systems"), text_score, evidence.value("recent_hands_on_coding_signal") * 0.85))


def _product_management_depth(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    evidence = candidate_evidence.legacy_evidence
    text_score = _text_signal_score(evidence, ("roadmap", "product", "stakeholder", "metrics", "experiment", "launch"))
    return _clamp(max(candidate_evidence.score("product_management"), text_score, evidence.value("product_company_experience") * 0.75))


def _text_signal_score(evidence: Evidence, terms: tuple[str, ...]) -> float:
    best = 0.0
    for signal in evidence.signals.values():
        snippets = " ".join(signal.snippets).lower()
        if any(term in snippets for term in terms):
            best = max(best, signal.value)
    return best


def _rolespec_recall_gate(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> bool:
    if _role_title_match_score({"profile": {"current_title": _title_text(candidate_evidence)}}, candidate_evidence, rolespec) < 0.70:
        return False
    if not rolespec.experience_range[0] <= candidate_evidence.years_experience <= rolespec.experience_range[1]:
        return False
    return _evidence_requirement_fit(candidate_evidence, rolespec) >= 0.80


def _score_cap_applies(rule: str, profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> bool:
    evidence = candidate_evidence.legacy_evidence
    if rule in {"missing_must_have_production_evidence", "no_production_evidence"}:
        return any(
            competency in {"production_ml_systems", "backend_systems"} and _role_aware_competency_score(candidate_evidence, competency, rolespec) < 0.35
            for competency in rolespec.must_have_competencies
        ) or _production_career_evidence(evidence) < 0.30
    if rule == "skills_only_match":
        return _skills_only_match(candidate_evidence, rolespec)
    if rule in {"unrelated_title", "unrelated_title_and_history"}:
        return _role_title_match_score(profile, candidate_evidence, rolespec) < 0.35 and _weighted_competency_match(candidate_evidence, rolespec) < 0.35
    if rule in {"inactive_candidate", "inactive_low_response"}:
        return _inactive_low_response(profile)
    if rule == "no_ranking_search_recommendation_evidence":
        return _ranking_search_recommendation_evidence(candidate_evidence) < 0.30
    if rule == "no_ranking_evaluation_evidence":
        return _ranking_evaluation_depth(evidence) < 0.25
    if rule == "generic_ml_without_retrieval_or_evaluation":
        return (
            _role_aware_competency_score(candidate_evidence, "ai_ml_engineering", rolespec) >= 0.55
            and _ranking_search_recommendation_evidence(candidate_evidence) < 0.35
            and _ranking_evaluation_depth(evidence) < 0.30
        )
    if rule == "recent_llm_demo_only":
        return evidence.value("llm_only_recent_demo_signal") >= 0.50 and _production_career_evidence(evidence) < 0.35
    if rule == "pure_research_without_deployment":
        return candidate_evidence.score("research") >= 0.45 and _production_career_evidence(evidence) < 0.25
    if rule == "consulting_only_background":
        return candidate_evidence.score("consulting_services") >= 0.8
    return False


def _ranking_search_recommendation_evidence(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    career = _career_backed_score(
        evidence,
        ("retrieval_search_ranking_experience", "evaluation_framework_experience"),
    )
    return max(
        _role_aware_competency_score(candidate_evidence, "ranking_recommendation", default_redrob_ai_engineer_rolespec()),
        _role_aware_competency_score(candidate_evidence, "retrieval_search", default_redrob_ai_engineer_rolespec()),
        career,
        evidence.value("semantic_production_retrieval_score") * 0.75,
    )


def _skills_only_match(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> bool:
    matched = [
        item
        for name, item in candidate_evidence.competencies.items()
        if name in rolespec.scoring_weights and item.score >= 0.35
    ]
    return bool(matched) and all(item.source_field.startswith("skills") or "assessment" in item.source_field for item in matched)


def _negative_signal_score(signal: str, profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    if signal == "keyword_stuffing":
        return candidate_evidence.score("keyword_stuffing")
    if signal == "skills_only_match":
        return 1.0 if _skills_only_match(candidate_evidence, rolespec) else 0.0
    if signal == "unrelated_title":
        return 1.0 if candidate_evidence.title_family == "other" else 0.0
    if signal == "inactive_low_response":
        return 1.0 if _inactive_low_response(profile) else 0.0
    if signal == "pure_research_without_deployment":
        return max(0.0, candidate_evidence.score("research") - candidate_evidence.score("production_ml_systems") * 0.5)
    if signal == "consulting_only_background":
        return candidate_evidence.score("consulting_services")
    return 0.0


def _profile_title(profile: dict[str, Any]) -> str:
    raw = profile.get("profile", {})
    return str(raw.get("current_title", "")).strip() if isinstance(raw, dict) else ""


def _production_retrieval_depth(evidence: Evidence) -> float:
    career_backed = _career_backed_score(
        evidence,
        ("retrieval_search_ranking_experience", "production_ai_experience"),
    )
    semantic = evidence.value("semantic_production_retrieval_score")
    deployment = evidence.value("semantic_product_shipping_score")
    return _clamp(max(career_backed, semantic * 0.92, deployment * 0.72))


def _retrieval_ranking_depth(evidence: Evidence) -> float:
    career = _career_backed_score(evidence, ("retrieval_search_ranking_experience",))
    source_quality = _source_quality_signal(evidence, ("retrieval_search_ranking_experience",))
    semantic = evidence.value("semantic_production_retrieval_score")
    if career < 0.30:
        return _clamp(max(source_quality, semantic * 0.62))
    return _clamp(max(career, source_quality, semantic * 0.88))


def _ranking_evaluation_depth(evidence: Evidence) -> float:
    career_eval = _career_backed_score(evidence, ("evaluation_framework_experience",))
    source_quality = _source_quality_signal(evidence, ("evaluation_framework_experience",))
    semantic = evidence.value("semantic_evaluation_score")
    if career_eval < 0.25:
        return _clamp(max(source_quality, semantic * 0.68))
    return _clamp(max(career_eval, source_quality, semantic * 0.90))


def _career_backed_vector_depth(evidence: Evidence) -> float:
    career_vector = _career_backed_score(evidence, ("vector_database_experience",))
    career_retrieval = _career_backed_score(evidence, ("retrieval_search_ranking_experience",))
    source_quality = _source_quality_signal(evidence, ("vector_database_experience",))
    semantic = evidence.value("semantic_vector_hybrid_search_score")
    if career_vector < 0.20 and career_retrieval < 0.45:
        return min(0.45, max(source_quality, semantic))
    return _clamp(max(career_vector, source_quality, semantic * 0.90))


def _python_ml_engineering_depth(evidence: Evidence) -> float:
    return _clamp(
        evidence.value("python_strength") * 0.45
        + evidence.value("recent_hands_on_coding_signal") * 0.25
        + evidence.value("production_ai_experience") * 0.20
        + evidence.value("data_engineering_adjacent_strength") * 0.10
    )


def _production_scale_score(profile: dict[str, Any]) -> float:
    text = _candidate_text(profile)
    scale_terms = (
        "30m+",
        "50m+",
        "10m+",
        "30m ",
        "50m ",
        "10m ",
        "million users",
        "millions of users",
        "million items",
        "millions of items",
        "million queries",
        "millions of queries",
        "queries/month",
        "queries per month",
        "candidate corpus",
        "document corpus",
        "500k",
        "500k+",
        "8k qps",
        "qps",
        "p95",
        "sub-200ms",
        "sub 200ms",
        "latency",
        "engagement lift",
        "recruiter engagement lift",
        "time-to-shortlist",
        "ctr",
        "conversion",
        "revenue",
        "deployed system",
        "production scale",
        "real users",
    )
    return min(1.0, sum(0.12 for term in scale_terms if term in text))


def _market_validation_score(signals: dict[str, Any]) -> float:
    views = _safe_float(signals.get("profile_views_received_30d"), default=0.0)
    appearances = _safe_float(signals.get("search_appearance_30d"), default=0.0)
    saved = _safe_float(signals.get("saved_by_recruiters_30d"), default=0.0)
    return _clamp(
        _log_norm(views, 250.0) * 0.30
        + _log_norm(appearances, 800.0) * 0.35
        + _log_norm(saved, 30.0) * 0.35
    )


def _last_active_score(value: Any) -> float:
    active = _parse_date(value)
    if active is None:
        return 0.25
    days = (date.today() - active).days
    if days <= 14:
        return 1.0
    if days <= 30:
        return 0.85
    if days <= 90:
        return 0.62
    if days <= 180:
        return 0.35
    return 0.10


def _notice_score(days: float) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.78
    if days <= 90:
        return 0.58
    return 0.25


def _skill_assessment_score(value: Any) -> float:
    if not isinstance(value, dict) or not value:
        return 0.35
    scores = [_safe_float(score, default=0.0) for score in value.values()]
    return _clamp(sum(scores) / max(1, len(scores)) / 100.0)


def _inactive_low_response(profile: dict[str, Any]) -> bool:
    signals = _signals(profile)
    return _last_active_score(signals.get("last_active_date")) <= 0.35 and _safe_float(
        signals.get("recruiter_response_rate"),
        default=0.0,
    ) < 0.20


def _ai_only_in_skills(evidence: Evidence) -> bool:
    career_ai = _career_ai_search_evidence(evidence)
    skill_sources = []
    for name in ("retrieval_search_ranking_experience", "vector_database_experience", "production_ai_experience"):
        signal = evidence.signals.get(name)
        if signal is not None:
            skill_sources.extend(field for field in signal.source_fields if field.startswith("skills") or "assessment" in field)
    return bool(skill_sources) and career_ai < 0.25


def _career_ai_search_evidence(evidence: Evidence) -> float:
    return max(
        _career_backed_score(evidence, ("retrieval_search_ranking_experience", "production_ai_experience", "evaluation_framework_experience")),
        evidence.value("pre_llm_ml_experience_signal") * 0.75,
    )


def _production_career_evidence(evidence: Evidence) -> float:
    return max(
        _career_backed_score(evidence, ("production_ai_experience", "retrieval_search_ranking_experience")),
        evidence.value("semantic_product_shipping_score") * 0.45,
    )


def _complete_match_bundle_score(profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> float:
    if not _is_redrob_senior_ai_rolespec(rolespec):
        return 0.0
    evidence = candidate_evidence.legacy_evidence
    title = _relevant_title_score(candidate_evidence)
    experience = _experience_band_match(candidate_evidence, strict=True)
    production = _career_backed_score(evidence, ("retrieval_search_ranking_experience", "production_ai_experience"))
    evaluation = _career_backed_score(evidence, ("evaluation_framework_experience",))
    vector = _source_quality_signal(evidence, ("vector_database_experience",))
    scale = _production_scale_score(profile)
    if min(title, experience, production, evaluation, vector) < 0.70 or scale < 0.12:
        return 0.0
    return _clamp(
        title * 0.16
        + experience * 0.14
        + production * 0.25
        + evaluation * 0.20
        + vector * 0.15
        + scale * 0.10
    )


def _high_recall_safety_gate(profile: dict[str, Any], candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> bool:
    if not _is_redrob_senior_ai_rolespec(rolespec):
        return False
    evidence = candidate_evidence.legacy_evidence
    if _relevant_title_score(candidate_evidence) < 0.70:
        return False
    years = candidate_evidence.years_experience
    exceptional = (
        _career_backed_score(evidence, ("retrieval_search_ranking_experience",)) >= 0.65
        and _career_backed_score(evidence, ("evaluation_framework_experience",)) >= 0.55
    )
    if not (4.5 <= years <= 9.5 or (4.0 <= years < 4.5 and exceptional)):
        return False
    production = _career_backed_score(evidence, ("retrieval_search_ranking_experience", "production_ai_experience"))
    if production < 0.45:
        return False
    facets = 0
    if _career_backed_score(evidence, ("evaluation_framework_experience",)) >= 0.35:
        facets += 1
    if _source_quality_signal(evidence, ("vector_database_experience",)) >= 0.35:
        facets += 1
    if _production_scale_score(profile) >= 0.12:
        facets += 1
    signals = _signals(profile)
    response_rate = _safe_float(signals.get("recruiter_response_rate"), default=0.0)
    notice = _safe_float(signals.get("notice_period_days"), default=999.0)
    if response_rate >= 0.55 and notice <= 90:
        facets += 1
    return facets >= 2


def _experience_band_match(candidate_evidence: CandidateEvidence, strict: bool = False) -> float:
    years = candidate_evidence.years_experience
    if 5.0 <= years <= 9.0:
        return 1.0
    if not strict and 4.5 <= years <= 9.5:
        return 0.82
    evidence = candidate_evidence.legacy_evidence
    exceptional = (
        _career_backed_score(evidence, ("retrieval_search_ranking_experience",)) >= 0.65
        and _career_backed_score(evidence, ("evaluation_framework_experience",)) >= 0.55
    )
    if 4.0 <= years < 5.0 and exceptional:
        return 0.72
    if 9.0 < years <= 9.5 and exceptional:
        return 0.72
    return 0.0


def _is_redrob_senior_ai_rolespec(rolespec: RoleSpec) -> bool:
    role_text = f"{rolespec.role_title} {rolespec.role_family}".lower()
    return "ai" in role_text and any(term in role_text for term in ("senior", "search", "ml", "machine", "engineer"))


def _signals(profile: dict[str, Any]) -> dict[str, Any]:
    signals = profile.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _candidate_text(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    raw_profile = profile.get("profile", {})
    if isinstance(raw_profile, dict):
        parts.extend(str(raw_profile.get(field, "")) for field in ("current_title", "headline", "summary"))
    career = profile.get("career_history", [])
    if isinstance(career, list):
        for role in career:
            if isinstance(role, dict):
                parts.extend(str(role.get(field, "")) for field in ("title", "description", "industry"))
    return " ".join(parts).lower()


def _log_norm(value: float, high: float) -> float:
    return _clamp(math.log1p(max(0.0, value)) / math.log1p(high))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _role_fit_score(evidence: Evidence) -> float:
    """Role match from title relevance, seniority, and hands-on signal."""
    value = (
        evidence.value("ai_ml_title_strength") * 0.42
        + evidence.value("seniority_strength") * 0.20
        + evidence.value("recent_hands_on_coding_signal") * 0.16
        + evidence.value("nlp_ir_relevance") * 0.07
        + evidence.value("semantic_senior_ai_engineering_score") * 0.15
    )
    value -= evidence.value("llm_only_recent_demo_signal") * 0.18
    value -= evidence.value("services_only_penalty_signal") * 0.08
    return _clamp(value)


def _production_ml_score(evidence: Evidence) -> float:
    """Production ML systems depth, not generic AI curiosity."""
    value = (
        evidence.value("production_ai_experience") * 0.43
        + evidence.value("semantic_product_shipping_score") * 0.18
        + evidence.value("semantic_production_retrieval_score") * 0.16
        + evidence.value("pre_llm_ml_experience_signal") * 0.12
        + evidence.value("data_engineering_adjacent_strength") * 0.07
        + evidence.value("recent_hands_on_coding_signal") * 0.04
    )
    value -= evidence.value("llm_only_recent_demo_signal") * 0.22
    return _clamp(value)


def _retrieval_ranking_score(evidence: Evidence) -> float:
    """Search, ranking, recommendation, retrieval, and vector infra evidence."""
    value = (
        evidence.value("retrieval_search_ranking_experience") * 0.36
        + evidence.value("semantic_production_retrieval_score") * 0.26
        + evidence.value("semantic_vector_hybrid_search_score") * 0.18
        + evidence.value("vector_database_experience") * 0.12
        + evidence.value("nlp_ir_relevance") * 0.08
    )
    # If retrieval-like terms only appear in weak role contexts, cap their impact.
    if evidence.value("ai_ml_title_strength") < 0.35 and evidence.value("production_ai_experience") < 0.35:
        value = min(value, 0.35)
    return _clamp(value)


def _evaluation_score(evidence: Evidence) -> float:
    """Ranking evaluation framework experience."""
    value = (
        evidence.value("evaluation_framework_experience") * 0.55
        + evidence.value("semantic_evaluation_score") * 0.45
    )
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


def _combine_rolespec_score(
    candidate_evidence: CandidateEvidence,
    rolespec: RoleSpec,
    penalties: PenaltyResult,
) -> float:
    evidence = candidate_evidence.legacy_evidence
    adjusted_scores = _adjusted_rolespec_scores(candidate_evidence)
    weighted = 0.0
    for competency, weight in rolespec.scoring_weights.items():
        weighted += adjusted_scores.get(competency, candidate_evidence.score(competency)) * weight

    full_match_score = _full_match_pattern_score(candidate_evidence)
    safety_gate = _recall_safety_gate(candidate_evidence)
    experience_adjustment = _experience_role_adjustment(candidate_evidence)
    vector_penalty = _generic_vector_inflation_penalty(evidence)
    title_penalty = _title_mismatch_penalty(candidate_evidence)

    missing = _missing_must_have(candidate_evidence, rolespec)
    if safety_gate:
        missing = tuple(item for item in missing if item != "python_engineering")
    missing_penalty = min(0.18, 0.045 * len(missing))
    risk_penalty = 0.0
    for signal in rolespec.negative_signals:
        risk_penalty += min(0.08, candidate_evidence.score(signal) * 0.08)
    penalty = min(0.32, penalties.penalty_score * 0.24 + missing_penalty + risk_penalty + vector_penalty + title_penalty)
    bonus = min(0.07, full_match_score * 0.038 + (0.016 if safety_gate else 0.0) + experience_adjustment * 0.45)
    raw_score = weighted + bonus - penalty
    if safety_gate:
        raw_score = max(raw_score, 0.965 + full_match_score * 0.012 - penalties.penalty_score * 0.015 - title_penalty * 0.80)
    elif _exceptional_near_band(candidate_evidence):
        raw_score = max(raw_score, 0.956 + full_match_score * 0.008 - penalties.penalty_score * 0.01)
    if raw_score > 0.985:
        raw_score = 0.985 + (raw_score - 0.985) * 0.15
    return round(_clamp(raw_score), 6)


def _missing_must_have(candidate_evidence: CandidateEvidence, rolespec: RoleSpec) -> tuple[str, ...]:
    return tuple(
        competency
        for competency in rolespec.must_have_competencies
        if candidate_evidence.score(competency) < 0.35
    )


def _adjusted_rolespec_scores(candidate_evidence: CandidateEvidence) -> dict[str, float]:
    evidence = candidate_evidence.legacy_evidence
    scores = {name: item.score for name, item in candidate_evidence.competencies.items()}

    career_retrieval = _career_backed_score(
        evidence,
        ("retrieval_search_ranking_experience", "evaluation_framework_experience", "production_ai_experience"),
    )
    semantic_retrieval = evidence.value("semantic_production_retrieval_score")
    explicit_retrieval = evidence.value("retrieval_search_ranking_experience")
    scores["retrieval_search"] = _clamp(max(explicit_retrieval, semantic_retrieval * (0.82 + career_retrieval * 0.18)))
    scores["ranking_recommendation"] = _clamp(
        max(scores.get("ranking_recommendation", 0.0), semantic_retrieval * (0.80 + career_retrieval * 0.20))
    )

    career_vector = _career_backed_score(evidence, ("vector_database_experience", "retrieval_search_ranking_experience"))
    vector_score = max(evidence.value("semantic_vector_hybrid_search_score"), evidence.value("vector_database_experience"))
    if career_vector < 0.25 and career_retrieval < 0.45:
        vector_score = min(vector_score, 0.42)
    scores["vector_hybrid_search"] = _clamp(vector_score)

    eval_score = max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience"))
    if _career_backed_score(evidence, ("evaluation_framework_experience",)) < 0.25 and eval_score < 0.75:
        eval_score *= 0.82
    scores["ranking_evaluation"] = _clamp(eval_score)

    production_score = max(
        evidence.value("production_ai_experience"),
        evidence.value("semantic_product_shipping_score"),
        evidence.value("semantic_production_retrieval_score") * 0.78,
    )
    scores["mlops_production"] = _clamp(production_score)
    scores["ai_ml"] = _clamp(max(scores.get("ai_ml", 0.0), _relevant_title_score(candidate_evidence)))
    return scores


def _full_match_pattern_score(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    title = _relevant_title_score(candidate_evidence)
    years = candidate_evidence.years_experience
    if 5.0 <= years <= 9.0:
        experience = 1.0
    elif 4.0 <= years < 5.0:
        exceptional = min(
            evidence.value("semantic_production_retrieval_score"),
            max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")),
        )
        experience = 0.68 if exceptional >= 0.85 else 0.38
    elif 9.0 < years <= 12.0:
        experience = 0.55
    elif 3.0 <= years < 4.0:
        experience = 0.18
    else:
        experience = 0.08

    production_ranking = max(
        _career_backed_score(evidence, ("retrieval_search_ranking_experience", "production_ai_experience")),
        evidence.value("semantic_production_retrieval_score") * 0.88,
    )
    evaluation_or_deploy = max(
        evidence.value("evaluation_framework_experience"),
        evidence.value("semantic_evaluation_score"),
        evidence.value("semantic_product_shipping_score") * 0.72,
    )
    return _clamp(title * 0.25 + experience * 0.22 + production_ranking * 0.33 + evaluation_or_deploy * 0.20)


def _recall_safety_gate(candidate_evidence: CandidateEvidence) -> bool:
    evidence = candidate_evidence.legacy_evidence
    title_text = _title_text(candidate_evidence)
    if any(term in title_text for term in ("junior", "research engineer")):
        return False
    if _relevant_title_score(candidate_evidence) < 0.70:
        return False
    if not 4.5 <= candidate_evidence.years_experience <= 9.5:
        return False
    production_search = max(
        _career_backed_score(evidence, ("retrieval_search_ranking_experience",)),
        evidence.value("semantic_production_retrieval_score"),
    )
    evaluation_or_deployment = max(
        evidence.value("evaluation_framework_experience"),
        evidence.value("semantic_evaluation_score"),
        evidence.value("semantic_product_shipping_score"),
        evidence.value("production_ai_experience"),
    )
    return production_search >= 0.78 and evaluation_or_deployment >= 0.55


def _experience_role_adjustment(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    years = candidate_evidence.years_experience
    extraordinary = (
        _relevant_title_score(candidate_evidence) >= 0.75
        and evidence.value("semantic_production_retrieval_score") >= 0.9
        and max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")) >= 0.8
    )
    if 5.0 <= years <= 9.0:
        return 0.035
    if 4.0 <= years < 5.0:
        return 0.012 if extraordinary else -0.035
    if 3.0 <= years < 4.0:
        return -0.075 if extraordinary else -0.14
    if years < 3.0:
        return -0.18
    if 9.0 < years <= 12.0:
        return -0.025
    return -0.065


def _exceptional_near_band(candidate_evidence: CandidateEvidence, rolespec: RoleSpec | None = None) -> bool:
    evidence = candidate_evidence.legacy_evidence
    years = candidate_evidence.years_experience
    min_years = rolespec.experience_range[0] if rolespec is not None else 5.0
    if not max(0.0, min_years - 1.0) <= years < min_years:
        return False
    if rolespec is not None:
        title_ok = _role_title_match_score({"profile": {"current_title": _title_text(candidate_evidence)}}, candidate_evidence, rolespec) >= 0.70
        evidence_ok = _evidence_requirement_fit(candidate_evidence, rolespec) >= 0.75
        return title_ok and evidence_ok
    if _relevant_title_score(candidate_evidence) < 0.70:
        return False
    if evidence.value("semantic_production_retrieval_score") < 0.90:
        return False
    return max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")) >= 0.78


def _generic_vector_inflation_penalty(evidence: Evidence) -> float:
    vector_score = max(evidence.value("semantic_vector_hybrid_search_score"), evidence.value("vector_database_experience"))
    career_retrieval = _career_backed_score(evidence, ("retrieval_search_ranking_experience", "production_ai_experience"))
    if vector_score < 0.55 or career_retrieval >= 0.45:
        return 0.0
    vector_signal = evidence.signals.get("vector_database_experience")
    vector_sources = vector_signal.source_fields if vector_signal is not None else ()
    skill_only_vector = bool(vector_sources) and all(
        field.startswith("skills") or "assessment" in field for field in vector_sources
    )
    return 0.075 if skill_only_vector else 0.045


def _title_mismatch_penalty(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    title = _title_text(candidate_evidence)
    if _relevant_title_score(candidate_evidence) < 0.55:
        return 0.055
    if "junior" in title:
        return 0.08
    if "research engineer" in title:
        return 0.045
    if "applied scientist" in title:
        return 0.035
    if "staff" in title:
        return 0.025
    if "senior data scientist" in title:
        strong_ranking = min(
            evidence.value("semantic_production_retrieval_score"),
            max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")),
        )
        return 0.0 if strong_ranking >= 0.78 else 0.055
    return 0.0


def _title_text(candidate_evidence: CandidateEvidence) -> str:
    title_signal = candidate_evidence.legacy_evidence.signals.get("ai_ml_title_strength")
    return " ".join(title_signal.snippets if title_signal is not None else ()).lower()


def _career_backed_score(evidence: Evidence, signal_names: tuple[str, ...]) -> float:
    best = 0.0
    for name in signal_names:
        signal = evidence.signals.get(name)
        if signal is None:
            continue
        if any(field.startswith("career_history") for field in signal.source_fields):
            best = max(best, signal.value)
    return best


def _source_quality_signal(evidence: Evidence, signal_names: tuple[str, ...]) -> float:
    best = 0.0
    for name in signal_names:
        signal = evidence.signals.get(name)
        if signal is None:
            continue
        source_multiplier = 0.0
        for field in signal.source_fields:
            if field.startswith("career_history"):
                source_multiplier = max(source_multiplier, 1.0)
            elif field.startswith("profile"):
                source_multiplier = max(source_multiplier, 0.55)
            elif field.startswith("skills") or "assessment" in field:
                source_multiplier = max(source_multiplier, 0.15)
            elif field.startswith("semantic_intents"):
                source_multiplier = max(source_multiplier, 0.50)
        best = max(best, signal.value * source_multiplier)
    return _clamp(best)


def _relevant_title_score(candidate_evidence: CandidateEvidence) -> float:
    evidence = candidate_evidence.legacy_evidence
    title_signal = evidence.signals.get("ai_ml_title_strength")
    title = " ".join(title_signal.snippets if title_signal is not None else ()).lower()
    base = evidence.value("ai_ml_title_strength")
    if title_signal is not None and "profile.current_title" not in title_signal.source_fields:
        base = min(base, 0.68)
    if "staff" in title:
        base = max(min(base, 0.86), 0.78)
    if "senior applied scientist" in title or "applied scientist" in title:
        base = max(min(base, 0.88), 0.78)
    if "research engineer" in title:
        base = min(base, 0.72)
    if "senior data scientist" in title:
        strong_ranking = min(
            evidence.value("semantic_production_retrieval_score"),
            max(evidence.value("semantic_evaluation_score"), evidence.value("evaluation_framework_experience")),
        )
        return 0.82 if strong_ranking >= 0.78 else 0.55
    return base


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
