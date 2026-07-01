from __future__ import annotations

import pytest

from src.config import RankingConfig
from src.rolespec import EvidenceRequirement, RoleSpec, ScoreCap, rolespec_from_mapping, rolespec_to_mapping
from src.scoring import score_candidate_with_components


def _signals(**overrides):
    data = {
        "profile_completeness_score": 92,
        "last_active_date": "2026-06-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.75,
        "avg_response_time_hours": 18,
        "saved_by_recruiters_30d": 8,
        "profile_views_received_30d": 90,
        "search_appearance_30d": 160,
        "interview_completion_rate": 0.8,
        "notice_period_days": 30,
        "willing_to_relocate": True,
        "github_activity_score": 42,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "offer_acceptance_rate": 0.55,
        "applications_submitted_30d": 4,
        "endorsements_received": 24,
        "skill_assessment_scores": {"Python": 88},
    }
    data.update(overrides)
    return data


def _candidate(candidate_id: str, title: str, description: str, skills: list[str], years: float = 6.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": title,
            "summary": description,
            "location": "Pune",
            "country": "India",
            "years_of_experience": years,
            "current_title": title,
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": title,
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": int(years * 12),
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": description,
            }
        ],
        "education": [],
        "skills": [{"name": skill, "proficiency": "advanced", "endorsements": 5, "duration_months": 36} for skill in skills],
        "redrob_signals": _signals(),
    }


def _ai_rolespec() -> RoleSpec:
    from src.rolespec import default_redrob_ai_engineer_rolespec

    return default_redrob_ai_engineer_rolespec()


def _backend_rolespec() -> RoleSpec:
    return RoleSpec(
        role_title="Backend Engineer",
        role_family="backend_engineering",
        target_seniority="senior",
        experience_range=(4, 10),
        must_have_competencies=("backend_systems", "python_engineering"),
        preferred_competencies=("data_engineering", "devops_cloud"),
        negative_signals=("skills_only_match", "unrelated_title"),
        evidence_requirements=(EvidenceRequirement("backend_systems", 0.4, ("career_history",)),),
        score_caps=(ScoreCap("skills_only_match", 0.6), ScoreCap("unrelated_title", 0.5)),
        redrob_signal_policy={
            "hireability_multiplier_min": 0.85,
            "hireability_multiplier_max": 1.15,
            "trust_multiplier_min": 0.95,
            "trust_multiplier_max": 1.05,
        },
        reasoning_priorities=("backend_systems", "python_engineering"),
        scoring_weights={"backend_systems": 0.55, "python_engineering": 0.25, "data_engineering": 0.1, "devops_cloud": 0.1},
    )


def _pm_rolespec() -> RoleSpec:
    return RoleSpec(
        role_title="Product Manager",
        role_family="product_management",
        target_seniority="senior",
        experience_range=(4, 12),
        must_have_competencies=("product_management", "management"),
        preferred_competencies=("data_engineering", "design"),
        negative_signals=("unrelated_title",),
        evidence_requirements=(EvidenceRequirement("product_management", 0.35, ("career_history", "profile")),),
        score_caps=(ScoreCap("unrelated_title", 0.55),),
        redrob_signal_policy={
            "hireability_multiplier_min": 0.85,
            "hireability_multiplier_max": 1.15,
            "trust_multiplier_min": 0.95,
            "trust_multiplier_max": 1.05,
        },
        reasoning_priorities=("product_management", "management"),
        scoring_weights={"product_management": 0.55, "management": 0.2, "data_engineering": 0.15, "design": 0.1},
    )


def test_same_candidate_scores_differently_for_different_rolespecs() -> None:
    candidate = _candidate(
        "CAND_FAKE001",
        "Backend Engineer",
        "Built Python APIs, backend microservices, database integrations, Kafka pipelines, and production services.",
        ["Python", "FastAPI", "Postgres", "Kafka"],
    )

    backend_score = score_candidate_with_components(candidate, RankingConfig(), rolespec=_backend_rolespec()).final_score
    pm_score = score_candidate_with_components(candidate, RankingConfig(), rolespec=_pm_rolespec()).final_score

    assert backend_score > pm_score


def test_three_rolespecs_prefer_different_candidates() -> None:
    ai_candidate = _candidate(
        "CAND_FAKE_AI",
        "Senior AI Engineer",
        "Owned a production recommendation system with BM25 + dense retrieval, NDCG/MRR evaluation, A/B testing, and deployed ML monitoring.",
        ["Python", "FAISS", "Pinecone", "Learning to Rank"],
    )
    backend_candidate = _candidate(
        "CAND_FAKE_BE",
        "Backend Engineer",
        "Built backend APIs, microservices, Redis caching, Postgres schemas, p95 latency improvements, and deployment pipelines.",
        ["Python", "FastAPI", "Redis", "Postgres"],
    )
    pm_candidate = _candidate(
        "CAND_FAKE_PM",
        "Senior Product Manager",
        "Owned product roadmap, stakeholder alignment, launch metrics, experimentation, user research, and prioritization.",
        ["Roadmapping", "Analytics", "Experimentation", "User Research"],
    )

    config = RankingConfig()
    assert score_candidate_with_components(ai_candidate, config, _ai_rolespec()).final_score > score_candidate_with_components(backend_candidate, config, _ai_rolespec()).final_score
    assert score_candidate_with_components(backend_candidate, config, _backend_rolespec()).final_score > score_candidate_with_components(pm_candidate, config, _backend_rolespec()).final_score
    assert score_candidate_with_components(pm_candidate, config, _pm_rolespec()).final_score > score_candidate_with_components(ai_candidate, config, _pm_rolespec()).final_score


def test_redrob_signals_modify_but_do_not_dominate_rolespec_score() -> None:
    strong = _candidate(
        "CAND_FAKE_STRONG",
        "Senior AI Engineer",
        "Shipped production search ranking and recommendation systems with A/B testing and NDCG evaluation.",
        ["Python", "Elasticsearch", "Learning to Rank"],
    )
    weak = _candidate(
        "CAND_FAKE_WEAK",
        "Marketing Manager",
        "Used AI tools and wrote marketing campaigns.",
        ["RAG", "Pinecone", "Prompt Engineering"],
    )
    weak["redrob_signals"] = _signals(recruiter_response_rate=1.0, avg_response_time_hours=1, interview_completion_rate=1.0)

    config = RankingConfig()
    assert score_candidate_with_components(strong, config, _ai_rolespec()).final_score > score_candidate_with_components(weak, config, _ai_rolespec()).final_score


def test_rolespec_validation_rejects_unsupported_competencies() -> None:
    payload = rolespec_to_mapping(_backend_rolespec())
    payload["must_have_competencies"] = ["backend_systems", "unsupported_magic"]

    with pytest.raises(ValueError, match="Unsupported competencies"):
        rolespec_from_mapping(payload)


def test_llm_generated_rolespec_payload_validates_before_use() -> None:
    payload = rolespec_to_mapping(_pm_rolespec())
    validated = rolespec_from_mapping(payload)

    assert validated.role_title == "Product Manager"
    assert validated.scoring_weights["product_management"] > validated.scoring_weights["design"]


def test_final_scoring_is_deterministic() -> None:
    candidate = _candidate(
        "CAND_FAKE_DET",
        "Backend Engineer",
        "Built Python backend services, APIs, microservices, and database-backed systems.",
        ["Python", "FastAPI", "Postgres"],
    )

    first = score_candidate_with_components(candidate, RankingConfig(), _backend_rolespec()).final_score
    second = score_candidate_with_components(candidate, RankingConfig(), _backend_rolespec()).final_score

    assert first == second
