from __future__ import annotations

import copy
import json
from pathlib import Path

from src.config import RankingConfig
from src.penalties import calculate_penalties
from src.reasoning import build_reasoning
from src.scoring import score_candidate_with_components


def _sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(candidate_id: str) -> dict:
    for candidate in _sample_candidates():
        if candidate["candidate_id"] == candidate_id:
            return copy.deepcopy(candidate)
    raise AssertionError(f"Missing sample candidate {candidate_id}")


def test_top_rank_reasoning_is_confident_and_specific() -> None:
    candidate = _candidate("CAND_0000031")
    score = score_candidate_with_components(candidate, RankingConfig())

    reasoning = build_reasoning(
        evidence=score.evidence,
        penalties=score.penalties,
        rank=1,
        component_scores=score.component_scores,
        candidate=candidate,
    )

    assert any(word in reasoning for word in ("High-confidence", "Strong", "Compelling", "Clear fit"))
    assert "Recommendation Systems Engineer" in reasoning
    assert "6" in reasoning
    assert any(term in reasoning.lower() for term in ("recommendation", "retrieval", "search", "ranking"))


def test_low_rank_reasoning_is_cautious_and_mentions_concern() -> None:
    candidate = _candidate("CAND_0000043")
    score = score_candidate_with_components(candidate, RankingConfig())

    reasoning = build_reasoning(
        evidence=score.evidence,
        penalties=score.penalties,
        rank=90,
        component_scores=score.component_scores,
        candidate=candidate,
    )

    assert any(word in reasoning for word in ("Cautious", "Borderline", "Lower-confidence", "Possible fit"))
    assert any(word in reasoning for word in ("Concern", "Watch-out", "Caveat", "Tradeoff"))
    assert "Python" not in reasoning


def test_reasoning_acknowledges_penalty_concerns() -> None:
    candidate = _candidate("CAND_0000031")
    candidate["redrob_signals"]["notice_period_days"] = 120
    score = score_candidate_with_components(candidate, RankingConfig())
    penalties = calculate_penalties(candidate)

    reasoning = build_reasoning(
        evidence=score.evidence,
        penalties=penalties,
        rank=15,
        component_scores=score.component_scores,
        candidate=candidate,
    )

    assert "120" in reasoning
    assert "notice" in reasoning.lower()
