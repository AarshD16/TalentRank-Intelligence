from __future__ import annotations

import copy
import json
from pathlib import Path

from src.penalties import PenaltyResult, calculate_penalties


def _sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(candidate_id: str) -> dict:
    for candidate in _sample_candidates():
        if candidate["candidate_id"] == candidate_id:
            return copy.deepcopy(candidate)
    raise AssertionError(f"Missing sample candidate {candidate_id}")


def _base_candidate() -> dict:
    return {
        "candidate_id": "CAND_9999000",
        "profile": {
            "headline": "Senior AI Engineer",
            "summary": "Built production search and ranking systems with Python.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior AI Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": "Senior AI Engineer",
                "start_date": "2021-06-01",
                "end_date": None,
                "duration_months": 61,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Shipped production recommendation and retrieval systems to real users using Python.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 20, "duration_months": 60},
            {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "saved_by_recruiters_30d": 8,
            "profile_views_received_30d": 80,
            "interview_completion_rate": 0.9,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "github_activity_score": 35,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 55},
            "skill_assessment_scores": {"Python": 86},
        },
    }


def test_keyword_stuffed_marketing_profile_gets_hard_flags() -> None:
    candidate = _base_candidate()
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["profile"]["headline"] = "AI RAG LLM vector search growth leader"
    candidate["profile"]["summary"] = "Built tutorials with LangChain, OpenAI API, RAG, Pinecone, FAISS, and embeddings."
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][0]["description"] = "Managed campaigns, content calendars, paid media, and brand reporting."
    candidate["career_history"][0]["industry"] = "Marketing"
    candidate["skills"] = [
        {"name": name, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for name in ["RAG", "Pinecone", "FAISS", "LangChain", "Embeddings", "Fine-tuning LLMs"]
    ]

    result = calculate_penalties(candidate)

    assert isinstance(result, PenaltyResult)
    assert result.penalty_score >= 0.35
    assert any("AI keyword" in reason or "AI skills" in reason for reason in result.hard_flags)
    assert any("expert AI skills" in reason for reason in result.hard_flags)


def test_expert_skills_with_zero_months_are_penalized() -> None:
    candidate = _base_candidate()
    candidate["skills"].extend(
        [
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        ]
    )

    result = calculate_penalties(candidate)

    assert result.penalty_score >= 0.18
    assert any("expert skills have 0 months" in reason for reason in result.penalty_reasons)
    assert any("expert AI skills" in reason for reason in result.hard_flags)


def test_inactive_perfect_on_paper_candidate_gets_availability_penalty() -> None:
    candidate = _base_candidate()
    candidate["redrob_signals"]["last_active_date"] = "2025-01-01"
    candidate["redrob_signals"]["recruiter_response_rate"] = 0.05

    result = calculate_penalties(candidate)

    assert result.penalty_score >= 0.13
    assert any("inactive" in reason for reason in result.hard_flags)


def test_strong_candidate_with_one_minor_concern_is_not_over_penalized() -> None:
    candidate = _candidate("CAND_0000031")

    result = calculate_penalties(candidate)

    assert result.penalty_score < 0.15
    assert not result.hard_flags
    assert result.total_penalty == result.penalty_score
    assert result.reasons == result.penalty_reasons
