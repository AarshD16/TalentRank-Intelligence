"""Compare current scoring with semantic intent features ablated."""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RankingConfig
from src.evidence_extractor import Evidence, SignalEvidence
from src.io import iter_candidates_jsonl
from src.scoring import (
    COMPONENT_NAMES,
    _combine_components,
    _company_context_score,
    _evaluation_score,
    _experience_score,
    _logistics_score,
    _production_ml_score,
    _redrob_availability_score,
    _retrieval_ranking_score,
    _role_fit_score,
    _skill_quality_score,
    _trust_consistency_score,
    score_candidate_with_components,
)


SEMANTIC_SIGNALS = (
    "semantic_jd_intent_score",
    "semantic_production_retrieval_score",
    "semantic_vector_hybrid_search_score",
    "semantic_evaluation_score",
    "semantic_senior_ai_engineering_score",
    "semantic_product_shipping_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic feature ablation diagnostics.")
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--limit", default=5000, type=int)
    parser.add_argument("--top-k", default=100, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RankingConfig()
    current_heap: list[tuple[float, str, str]] = []
    ablated_heap: list[tuple[float, str, str]] = []
    deltas: list[tuple[float, str, float, float]] = []
    processed = 0

    for candidate in iter_candidates_jsonl(args.candidates):
        processed += 1
        scored = score_candidate_with_components(candidate, config)
        ablated_score = _score_without_semantic(scored, config)
        _push_top(current_heap, args.top_k, scored.final_score, scored.candidate_id)
        _push_top(ablated_heap, args.top_k, ablated_score, scored.candidate_id)
        deltas.append((scored.final_score - ablated_score, scored.candidate_id, scored.final_score, ablated_score))
        if args.limit and processed >= args.limit:
            break

    current_ids = {cid for _score, _reverse, cid in current_heap}
    ablated_ids = {cid for _score, _reverse, cid in ablated_heap}
    print("# Semantic Ablation")
    print(f"Candidates processed: {processed}")
    print(f"Top-{args.top_k} overlap current vs semantic-disabled: {len(current_ids & ablated_ids)}")
    print("\nLargest semantic lifts:")
    for delta, cid, current, ablated in sorted(deltas, reverse=True)[:25]:
        print(f"- {cid}: +{delta:.4f} current={current:.4f} ablated={ablated:.4f}")
    return 0


def _score_without_semantic(scored, config: RankingConfig) -> float:
    signals = dict(scored.evidence.signals)
    zero = SignalEvidence(value=0.0, snippets=(), source_fields=("semantic_ablation",))
    for name in SEMANTIC_SIGNALS:
        signals[name] = zero
    evidence = Evidence(candidate_id=scored.evidence.candidate_id, signals=signals)
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
    return _combine_components(components, config, scored.penalties)


def _push_top(heap: list[tuple[float, str, str]], top_k: int, score: float, candidate_id: str) -> None:
    item = (score, "".join(chr(255 - ord(char) % 255) for char in candidate_id), candidate_id)
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


if __name__ == "__main__":
    raise SystemExit(main())
