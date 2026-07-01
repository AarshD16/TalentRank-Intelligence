from __future__ import annotations

import json
from pathlib import Path

from src.config import RankingConfig
from src.scoring import COMPONENT_NAMES, score_candidate, score_candidate_with_components


def _sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(candidate_id: str) -> dict:
    for candidate in _sample_candidates():
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise AssertionError(f"Missing sample candidate {candidate_id}")


def test_scoring_weights_are_configured_for_all_components() -> None:
    weights = RankingConfig().resolved_scoring_weights()

    assert set(weights) == set(COMPONENT_NAMES)
    assert abs(sum(weights.values()) - 1.0) < 0.0001


def test_strong_ai_engineer_scores_above_keyword_stuffed_non_engineer() -> None:
    config = RankingConfig()
    strong = score_candidate_with_components(_candidate("CAND_0000031"), config)
    keyword_stuffed = score_candidate_with_components(_candidate("CAND_0000021"), config)

    assert strong.final_score > keyword_stuffed.final_score + 0.4
    assert strong.component_scores["capability_score"] > 0.8
    assert 0.85 <= 0.85 + strong.component_scores["redrob_hireability_score"] * 0.30 <= 1.15
    assert keyword_stuffed.component_scores["capability_score"] < 0.4


def test_marketing_manager_with_many_ai_keywords_still_scores_low() -> None:
    config = RankingConfig()
    strong = score_candidate_with_components(_candidate("CAND_0000031"), config)
    stuffed_marketer = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "headline": "AI RAG LLM vector search enthusiast",
            "summary": "Built tutorial projects with LangChain, OpenAI API, RAG, Pinecone, FAISS, embeddings, and prompt engineering.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Marketing Manager",
            "current_company": "Generic Ads",
            "current_company_size": "201-500",
            "current_industry": "Marketing",
        },
        "career_history": [
            {
                "company": "Generic Ads",
                "title": "Marketing Manager",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 52,
                "is_current": True,
                "industry": "Marketing",
                "company_size": "201-500",
                "description": "Managed campaigns, content calendars, paid media, and brand reporting.",
            }
        ],
        "education": [],
        "skills": [
            {"name": name, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
            for name in ["RAG", "Pinecone", "FAISS", "LangChain", "Embeddings", "Fine-tuning LLMs"]
        ],
        "redrob_signals": {
            "profile_completeness_score": 80,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "saved_by_recruiters_30d": 5,
            "profile_views_received_30d": 50,
            "interview_completion_rate": 0.8,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "github_activity_score": -1,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "skill_assessment_scores": {},
        },
    }

    scored = score_candidate_with_components(stuffed_marketer, config)

    assert scored.final_score < strong.final_score
    assert scored.component_scores["capability_score"] < 0.35
    assert "cap_unrelated_title" in scored.score_flags


def test_generic_llm_demo_profile_does_not_dominate_core_fit() -> None:
    config = RankingConfig()
    strong = score_candidate_with_components(_candidate("CAND_0000031"), config)
    demo_profile = score_candidate_with_components(_candidate("CAND_0000043"), config)

    assert demo_profile.evidence.value("llm_only_recent_demo_signal") > 0.5
    assert demo_profile.final_score < strong.final_score
    assert demo_profile.component_scores["capability_score"] < 0.5


def test_score_candidate_returns_normalized_float() -> None:
    score = score_candidate(_candidate("CAND_0000031"), RankingConfig())

    assert 0.0 <= score <= 1.0
