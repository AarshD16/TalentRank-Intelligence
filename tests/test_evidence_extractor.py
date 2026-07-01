from __future__ import annotations

import json
from pathlib import Path

from src.evidence_extractor import Evidence, SignalEvidence, extract_evidence


EXPECTED_SIGNALS = {
    "total_years_experience",
    "ai_ml_title_strength",
    "seniority_strength",
    "production_ai_experience",
    "retrieval_search_ranking_experience",
    "vector_database_experience",
    "evaluation_framework_experience",
    "python_strength",
    "product_company_experience",
    "services_only_penalty_signal",
    "recent_hands_on_coding_signal",
    "llm_only_recent_demo_signal",
    "pre_llm_ml_experience_signal",
    "nlp_ir_relevance",
    "data_engineering_adjacent_strength",
    "open_source_or_github_signal",
    "education_strength",
    "location_fit",
    "notice_period_fit",
    "availability_signal",
    "recruiter_engagement_signal",
    "profile_trust_signal",
    "semantic_jd_intent_score",
    "semantic_production_retrieval_score",
    "semantic_vector_hybrid_search_score",
    "semantic_evaluation_score",
    "semantic_senior_ai_engineering_score",
    "semantic_product_shipping_score",
}


def _sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(candidate_id: str) -> dict:
    for candidate in _sample_candidates():
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise AssertionError(f"Missing sample candidate {candidate_id}")


def test_extract_evidence_returns_all_expected_signals() -> None:
    evidence = extract_evidence(_candidate("CAND_0000031"))

    assert isinstance(evidence, Evidence)
    assert evidence.candidate_id == "CAND_0000031"
    assert set(evidence.signals) == EXPECTED_SIGNALS
    assert all(isinstance(signal, SignalEvidence) for signal in evidence.signals.values())


def test_strong_recommendation_systems_candidate_has_core_ai_evidence() -> None:
    evidence = extract_evidence(_candidate("CAND_0000031"))

    assert evidence.value("total_years_experience") == 6.0
    assert evidence.value("ai_ml_title_strength") >= 0.95
    assert evidence.value("production_ai_experience") >= 0.5
    assert evidence.value("retrieval_search_ranking_experience") >= 0.7
    assert evidence.value("vector_database_experience") >= 0.5
    assert evidence.value("nlp_ir_relevance") >= 0.5
    assert evidence.value("availability_signal") >= 0.5
    assert evidence.signals["retrieval_search_ranking_experience"].snippets
    assert evidence.signals["retrieval_search_ranking_experience"].source_fields


def test_backend_data_candidate_gets_adjacent_data_engineering_signal() -> None:
    evidence = extract_evidence(_candidate("CAND_0000001"))

    assert evidence.value("data_engineering_adjacent_strength") >= 0.5
    assert evidence.value("services_only_penalty_signal") > 0.0
    assert evidence.value("ai_ml_title_strength") < 0.5
    assert evidence.signals["data_engineering_adjacent_strength"].snippets


def test_recent_llm_demo_candidate_is_flagged_without_strong_production_ai() -> None:
    evidence = extract_evidence(_candidate("CAND_0000043"))

    assert evidence.value("llm_only_recent_demo_signal") >= 0.25
    assert evidence.value("recruiter_engagement_signal") < 0.5
    assert evidence.value("open_source_or_github_signal") == 0.0
    assert evidence.signals["llm_only_recent_demo_signal"].snippets
