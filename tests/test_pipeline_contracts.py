from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from rank import CandidateScore, _sort_best_first
from scripts.audit_reasoning import unsupported_skill_claims
from src.config import RankingConfig
from src.evidence_extractor import extract_evidence
from src.io import iter_candidates_jsonl, write_submission_csv
from src.penalties import calculate_penalties
from src.reasoning import build_reasoning
from src.scoring import RankedCandidate, score_candidate_with_components
from validate_submission import validate_submission


def _sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(candidate_id: str) -> dict:
    for candidate in _sample_candidates():
        if candidate["candidate_id"] == candidate_id:
            return json.loads(json.dumps(candidate))
    raise AssertionError(f"Missing sample candidate {candidate_id}")


def _valid_rows() -> list[RankedCandidate]:
    return [
        RankedCandidate(
            candidate_id=f"CAND_{index:07d}",
            rank=index,
            score=1.0 - index / 1000.0,
            reasoning=f"Supported reasoning for candidate {index}.",
        )
        for index in range(1, 101)
    ]


def test_candidate_loading_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    rows = [_candidate("CAND_0000031"), _candidate("CAND_0000001")]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loaded = list(iter_candidates_jsonl(path))

    assert [row["candidate_id"] for row in loaded] == ["CAND_0000031", "CAND_0000001"]


def test_evidence_extraction_correctness_for_strong_candidate() -> None:
    evidence = extract_evidence(_candidate("CAND_0000031"))

    assert evidence.value("total_years_experience") == 6.0
    assert evidence.value("retrieval_search_ranking_experience") >= 0.8
    assert evidence.value("production_ai_experience") >= 0.7
    assert evidence.signals["retrieval_search_ranking_experience"].source_fields


def test_strong_ai_engineer_ranks_above_keyword_stuffed_non_engineer() -> None:
    config = RankingConfig()
    strong = score_candidate_with_components(_candidate("CAND_0000031"), config)
    stuffed = score_candidate_with_components(_candidate("CAND_0000021"), config)

    assert strong.final_score > stuffed.final_score
    assert strong.component_scores["capability_score"] > stuffed.component_scores["capability_score"]


def test_behavioral_signals_modify_but_do_not_dominate_technical_fit() -> None:
    config = RankingConfig()
    strong = score_candidate_with_components(_candidate("CAND_0000031"), config)
    weak = _candidate("CAND_0000043")
    weak["redrob_signals"].update(
        {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-20",
            "recruiter_response_rate": 1.0,
            "avg_response_time_hours": 1,
            "saved_by_recruiters_30d": 25,
            "profile_views_received_30d": 200,
            "interview_completion_rate": 1.0,
            "notice_period_days": 0,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        }
    )
    weak_score = score_candidate_with_components(weak, config)

    assert weak_score.component_scores["redrob_hireability_score"] > 0.8
    assert strong.final_score > weak_score.final_score


def test_honeypot_like_profile_receives_penalties() -> None:
    candidate = _candidate("CAND_0000043")
    candidate["skills"].extend(
        [
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ]
    )

    result = calculate_penalties(candidate)

    assert result.penalty_score >= 0.2
    assert result.hard_flags


def test_reasoning_contains_supported_facts_only() -> None:
    candidate = _candidate("CAND_0000043")
    score = score_candidate_with_components(candidate, RankingConfig())
    reasoning = build_reasoning(score.evidence, score.penalties, rank=90, component_scores=score.component_scores, candidate=candidate)

    assert reasoning
    assert unsupported_skill_claims(reasoning, candidate) == []


def test_csv_contract_exactly_100_rows_ranks_and_non_increasing_scores(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    write_submission_csv(path, _valid_rows())

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 100
    assert [int(row["rank"]) for row in rows] == list(range(1, 101))
    scores = [float(row["score"]) for row in rows]
    assert scores == sorted(scores, reverse=True)
    validate_submission(path)


def test_tie_break_is_deterministic_by_candidate_id() -> None:
    items = [
        CandidateScore(candidate={}, breakdown=SimpleNamespace(final_score=0.5, candidate_id="CAND_0000003")),
        CandidateScore(candidate={}, breakdown=SimpleNamespace(final_score=0.5, candidate_id="CAND_0000001")),
        CandidateScore(candidate={}, breakdown=SimpleNamespace(final_score=0.5, candidate_id="CAND_0000002")),
    ]

    sorted_items = _sort_best_first(items)

    assert [item.breakdown.candidate_id for item in sorted_items] == [
        "CAND_0000001",
        "CAND_0000002",
        "CAND_0000003",
    ]


def test_validate_submission_script_passes(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    write_submission_csv(path, _valid_rows())

    completed = subprocess.run(
        [sys.executable, "validate_submission.py", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS" in completed.stdout
