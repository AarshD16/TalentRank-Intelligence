"""Generic reusable candidate evidence model."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.evidence_extractor import Evidence, extract_evidence
from src.rolespec import COMPETENCY_TAXONOMY


@dataclass(frozen=True, slots=True)
class CompetencyEvidence:
    """Explainable evidence for one competency."""

    competency: str
    score: float
    evidence_snippet: str
    source_field: str


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Generic evidence categories reusable across job descriptions."""

    candidate_id: str
    title_family: str
    seniority_level: str
    years_experience: float
    domain_experience: dict[str, CompetencyEvidence]
    system_experience: dict[str, CompetencyEvidence]
    production_depth: CompetencyEvidence
    evaluation_depth: CompetencyEvidence
    scale_depth: CompetencyEvidence
    technical_skills: dict[str, CompetencyEvidence]
    company_context: dict[str, CompetencyEvidence]
    behavioral_availability: CompetencyEvidence
    trust_consistency: CompetencyEvidence
    risk_flags: tuple[str, ...]
    competencies: dict[str, CompetencyEvidence]
    legacy_evidence: Evidence

    def score(self, competency: str) -> float:
        item = self.competencies.get(competency)
        return 0.0 if item is None else item.score


def extract_candidate_evidence(candidate: dict[str, Any]) -> CandidateEvidence:
    """Extract generic career evidence from a raw candidate profile."""
    legacy = extract_evidence(candidate)
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    competencies = _competencies_from_legacy(legacy)
    for name in COMPETENCY_TAXONOMY:
        competencies.setdefault(name, CompetencyEvidence(name, 0.0, "", ""))
    return CandidateEvidence(
        candidate_id=legacy.candidate_id,
        title_family=_title_family(str(profile.get("current_title", ""))),
        seniority_level=_seniority_level(str(profile.get("current_title", "")), legacy.value("total_years_experience")),
        years_experience=legacy.value("total_years_experience"),
        domain_experience={
            key: competencies[key]
            for key in (
                "ai_ml_engineering",
                "retrieval_search",
                "ranking_recommendation",
                "vector_hybrid_search",
                "data_engineering",
                "product_management",
                "sales",
                "marketing",
                "design",
            )
        },
        system_experience={
            key: competencies[key]
            for key in ("production_ml_systems", "backend_systems", "devops_cloud", "frontend_engineering")
        },
        production_depth=competencies["production_ml_systems"],
        evaluation_depth=competencies["ranking_evaluation"],
        scale_depth=_scale_depth(candidate, legacy),
        technical_skills={
            key: competencies[key]
            for key in ("python_engineering", "data_engineering", "backend_systems", "frontend_engineering", "devops_cloud")
        },
        company_context={
            "product_management": competencies["product_management"],
            "consulting_services": competencies["consulting_services"],
        },
        behavioral_availability=competencies["behavioral_availability"],
        trust_consistency=competencies["trust_consistency"],
        risk_flags=tuple(
            name
            for name in ("research", "consulting_services", "keyword_stuffing")
            if competencies[name].score >= 0.5
        ),
        competencies=competencies,
        legacy_evidence=legacy,
    )


def _competencies_from_legacy(legacy: Evidence) -> dict[str, CompetencyEvidence]:
    competencies = {
        "ai_ml_engineering": _from_signals(legacy, "ai_ml_engineering", ("ai_ml_title_strength", "semantic_senior_ai_engineering_score", "production_ai_experience")),
        "retrieval_search": _from_signals(legacy, "retrieval_search", ("semantic_production_retrieval_score", "retrieval_search_ranking_experience", "nlp_ir_relevance")),
        "ranking_recommendation": _from_signals(legacy, "ranking_recommendation", ("semantic_production_retrieval_score", "retrieval_search_ranking_experience")),
        "vector_hybrid_search": _from_signals(legacy, "vector_hybrid_search", ("semantic_vector_hybrid_search_score", "vector_database_experience")),
        "ranking_evaluation": _from_signals(legacy, "ranking_evaluation", ("semantic_evaluation_score", "evaluation_framework_experience")),
        "production_ml_systems": _from_signals(legacy, "production_ml_systems", ("production_ai_experience", "semantic_product_shipping_score", "recent_hands_on_coding_signal")),
        "python_engineering": _from_signals(legacy, "python_engineering", ("python_strength", "recent_hands_on_coding_signal")),
        "product_management": _from_signals(legacy, "product_management", ("product_company_experience",)),
        "data_engineering": _from_signals(legacy, "data_engineering", ("data_engineering_adjacent_strength",)),
        "backend_systems": _derived("backend_systems", min(1.0, legacy.value("recent_hands_on_coding_signal") * 0.7 + legacy.value("data_engineering_adjacent_strength") * 0.3), legacy),
        "frontend_engineering": _keyword_competency(legacy, "frontend_engineering", ("frontend", "react", "vue", "angular", "typescript", "css", "html")),
        "devops_cloud": _from_signals(legacy, "devops_cloud", ("open_source_or_github_signal",)),
        "management": _derived("management", max(0.0, legacy.value("seniority_strength") - legacy.value("recent_hands_on_coding_signal") * 0.4), legacy),
        "research": _derived("research", max(0.0, legacy.value("llm_only_recent_demo_signal") - legacy.value("production_ai_experience") * 0.4), legacy),
        "consulting_services": _from_signals(legacy, "consulting_services", ("services_only_penalty_signal",)),
        "keyword_stuffing": _from_signals(legacy, "keyword_stuffing", ("llm_only_recent_demo_signal",)),
        "behavioral_availability": _from_signals(legacy, "behavioral_availability", ("availability_signal", "recruiter_engagement_signal", "notice_period_fit")),
        "trust_consistency": _from_signals(legacy, "trust_consistency", ("profile_trust_signal",)),
    }
    competencies.update(
        {
            "sales": _keyword_competency(legacy, "sales", ("sales", "account executive", "pipeline", "quota", "crm")),
            "marketing": _keyword_competency(legacy, "marketing", ("marketing", "campaign", "seo", "content", "growth")),
            "design": _keyword_competency(legacy, "design", ("design", "figma", "ux", "ui", "prototype")),
            "open_source_validation": _from_signals(legacy, "open_source_validation", ("open_source_or_github_signal",)),
        }
    )
    # Backward-compatible aliases for older tests and debug scripts.
    competencies["ai_ml"] = competencies["ai_ml_engineering"]
    competencies["mlops_production"] = competencies["production_ml_systems"]
    competencies["frontend"] = competencies["frontend_engineering"]
    competencies["devops"] = competencies["devops_cloud"]
    competencies["product_company"] = competencies["product_management"]
    competencies["services_only"] = competencies["consulting_services"]
    competencies["research_only"] = competencies["research"]
    return competencies


def _from_signals(legacy: Evidence, competency: str, signal_names: tuple[str, ...]) -> CompetencyEvidence:
    best_name = max(signal_names, key=lambda name: legacy.value(name))
    signal = legacy.signals.get(best_name)
    if signal is None:
        return CompetencyEvidence(competency, 0.0, "", "")
    return CompetencyEvidence(
        competency=competency,
        score=signal.value,
        evidence_snippet=signal.snippets[0] if signal.snippets else "",
        source_field=signal.source_fields[0] if signal.source_fields else best_name,
    )


def _derived(competency: str, score: float, legacy: Evidence) -> CompetencyEvidence:
    signal = legacy.signals.get("recent_hands_on_coding_signal")
    return CompetencyEvidence(
        competency=competency,
        score=round(max(0.0, min(score, 1.0)), 4),
        evidence_snippet=signal.snippets[0] if signal and signal.snippets else "",
        source_field=signal.source_fields[0] if signal and signal.source_fields else "derived",
    )


def _keyword_competency(legacy: Evidence, competency: str, terms: tuple[str, ...]) -> CompetencyEvidence:
    best_score = 0.0
    best_snippet = ""
    best_source = ""
    for signal in legacy.signals.values():
        for source, snippet in zip(signal.source_fields, signal.snippets, strict=False):
            lowered = snippet.lower()
            if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) for term in terms):
                score = signal.value * (1.0 if source.startswith("career_history") else 0.75)
                if score > best_score:
                    best_score = score
                    best_snippet = snippet
                    best_source = source
    return CompetencyEvidence(competency, round(max(0.0, min(best_score, 1.0)), 4), best_snippet, best_source)


def _scale_depth(candidate: dict[str, Any], legacy: Evidence) -> CompetencyEvidence:
    text = str(candidate).lower()
    scale_terms = ("10m", "500k", "scale", "real users", "production load", "revenue", "retention")
    score = min(1.0, sum(0.18 for term in scale_terms if term in text))
    return CompetencyEvidence("scale_depth", score, "", "career_history[].description")


def _title_family(title: str) -> str:
    lowered = title.lower()
    if any(term in lowered for term in ("machine learning", "ml", "ai", "nlp", "data scientist", "recommendation", "search")):
        return "ai_ml"
    if "data" in lowered:
        return "data"
    if "product" in lowered and "manager" in lowered:
        return "product_management"
    if any(term in lowered for term in ("sales", "account executive")):
        return "sales"
    if "marketing" in lowered:
        return "marketing"
    if "design" in lowered:
        return "design"
    if any(term in lowered for term in ("backend", "software", "developer", "engineer")):
        return "software_engineering"
    if "manager" in lowered:
        return "management"
    return "other"


def _seniority_level(title: str, years: float) -> str:
    lowered = title.lower()
    if any(term in lowered for term in ("staff", "principal", "lead", "head")):
        return "lead_plus"
    if "senior" in lowered or years >= 5:
        return "senior"
    if years >= 3:
        return "mid"
    return "junior"
