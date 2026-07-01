"""Human-readable, evidence-grounded reasoning generation."""

from __future__ import annotations

import logging
from typing import Any

from src.candidate_evidence import CandidateEvidence
from src.evidence_extractor import EvidenceBundle
from src.penalties import PenaltyResult
from src.rolespec import RoleSpec


LOGGER = logging.getLogger(__name__)

STRONG_OPENERS = (
    "High-confidence fit",
    "Strong fit",
    "Compelling fit",
    "Clear fit",
)

MID_OPENERS = (
    "Solid fit",
    "Relevant fit",
    "Credible fit",
    "Good fit",
)

CAUTIOUS_OPENERS = (
    "Cautious fit",
    "Borderline fit",
    "Lower-confidence fit",
    "Possible fit",
)

CONCERN_PREFIXES = (
    "Concern:",
    "Watch-out:",
    "Caveat:",
    "Tradeoff:",
)


def build_reasoning(
    evidence: EvidenceBundle,
    penalties: PenaltyResult,
    rank: int | None = None,
    component_scores: dict[str, float] | None = None,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Build concise recruiter-ready reasoning from supported evidence only."""
    LOGGER.debug("Building reasoning for candidate_id=%s", evidence.candidate_id)
    component_scores = component_scores or {}
    facts = _candidate_facts(candidate or {})
    opener = _opener(evidence.candidate_id, rank)

    positive_parts = _positive_parts(evidence, facts, component_scores)
    concern_parts = _concern_parts(evidence, penalties, facts, rank)

    if not positive_parts:
        positive_parts.append("profile has limited explicit evidence for the Senior AI Engineer brief")

    first_sentence = f"{opener}: {', '.join(positive_parts[:3])}."
    if concern_parts:
        prefix = CONCERN_PREFIXES[_stable_index(evidence.candidate_id, len(CONCERN_PREFIXES), offset=3)]
        second_sentence = f"{prefix} {concern_parts[0]}."
        return _csv_safe(f"{first_sentence} {second_sentence}")
    return _csv_safe(first_sentence)


def build_rolespec_reasoning(
    candidate_evidence: CandidateEvidence,
    rolespec: RoleSpec,
    penalties: PenaltyResult,
    rank: int | None = None,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Build concise JD-aware reasoning from RoleSpec priorities."""
    facts = _candidate_facts(candidate or {})
    opener = _opener(candidate_evidence.candidate_id, rank)
    title = facts.get("title") or candidate_evidence.title_family
    years = candidate_evidence.years_experience
    positive_parts = [f"{title} with {years:g} years"]
    positive_parts.extend(_exact_evidence_parts(candidate_evidence.legacy_evidence))

    for competency in rolespec.reasoning_priorities:
        item = candidate_evidence.competencies.get(competency)
        if item is None or item.score < 0.35:
            continue
        phrase = _evidence_phrase_from_snippet(item.evidence_snippet) or _humanize_competency(competency)
        if phrase:
            positive_parts.append(phrase)
        if len(positive_parts) >= 3:
            break

    missing = [
        _humanize_competency(name)
        for name in rolespec.must_have_competencies
        if candidate_evidence.score(name) < 0.35
    ]
    concerns = _concern_parts(candidate_evidence.legacy_evidence, penalties, facts, rank)
    if missing:
        concerns.insert(0, f"missing stronger evidence for {', '.join(missing[:2])}")

    first_sentence = f"{opener} for {rolespec.role_title}: {', '.join(_dedupe(positive_parts)[:3])}."
    if concerns:
        prefix = CONCERN_PREFIXES[_stable_index(candidate_evidence.candidate_id, len(CONCERN_PREFIXES), offset=3)]
        return _csv_safe(f"{first_sentence} {prefix} {concerns[0]}.")
    return _csv_safe(first_sentence)


def _positive_parts(
    evidence: EvidenceBundle,
    facts: dict[str, Any],
    component_scores: dict[str, float],
) -> list[str]:
    parts: list[str] = []
    title = facts.get("title")
    years = evidence.value("total_years_experience")
    if title and years:
        parts.append(f"{title} with {years:g} years")
    elif title:
        parts.append(str(title))
    elif years:
        parts.append(f"{years:g} years of experience")

    if evidence.value("retrieval_search_ranking_experience") >= 0.65:
        parts.append(_supported_domain_phrase(evidence, "retrieval_search_ranking_experience"))
    elif evidence.value("semantic_production_retrieval_score") >= 0.55:
        parts.append(_supported_domain_phrase(evidence, "semantic_production_retrieval_score"))
    elif evidence.value("data_engineering_adjacent_strength") >= 0.65:
        parts.append("strong data/ML systems adjacency")

    if evidence.value("semantic_vector_hybrid_search_score") >= 0.55:
        vector_phrase = _snippet_backed_phrase(evidence, "semantic_vector_hybrid_search_score", "")
        if vector_phrase:
            parts.append(vector_phrase)

    if evidence.value("production_ai_experience") >= 0.65:
        parts.append("production ML/AI delivery evidence")
    elif evidence.value("recent_hands_on_coding_signal") >= 0.65:
        parts.append("recent hands-on technical work")

    if evidence.value("evaluation_framework_experience") >= 0.45 or evidence.value("semantic_evaluation_score") >= 0.55:
        parts.append(_snippet_backed_phrase(evidence, "semantic_evaluation_score", "ranking/evaluation framework evidence"))

    if evidence.value("python_strength") >= 0.55 and _has_skill(facts, "python"):
        parts.append("Python is explicitly listed")

    product_company = facts.get("product_company")
    if evidence.value("product_company_experience") >= 0.5 and product_company:
        parts.append(f"product-company experience at {product_company}")

    response_rate = facts.get("recruiter_response_rate")
    if response_rate is not None and response_rate >= 0.75 and component_scores.get("redrob_hireability_score", 0.0) >= 0.70:
        parts.append(f"recruiter response rate {response_rate:.0%}")

    return _dedupe(parts)


def _concern_parts(
    evidence: EvidenceBundle,
    penalties: PenaltyResult,
    facts: dict[str, Any],
    rank: int | None,
) -> list[str]:
    concerns: list[str] = []
    if penalties.hard_flags:
        concerns.append(_humanize_flag(penalties.hard_flags[0]))
    elif penalties.soft_flags:
        concerns.append(_humanize_flag(penalties.soft_flags[0]))

    notice = facts.get("notice_period_days")
    if notice is not None and notice > 60:
        concerns.append(f"notice period is {notice:g} days")

    if evidence.value("llm_only_recent_demo_signal") >= 0.25:
        concerns.append("LLM evidence looks more demo-oriented than deep production ML")
    if evidence.value("services_only_penalty_signal") >= 0.35:
        concerns.append("IT-services-heavy background needs scrutiny against the JD")
    response_rate = facts.get("recruiter_response_rate")
    if response_rate is not None and response_rate < 0.2:
        concerns.append(f"recruiter response rate is {response_rate:.0%}")
    if rank is not None and rank >= 80 and not concerns:
        concerns.append("weaker explicit ranking or production ML evidence keeps them near the cutoff")
    return _dedupe(concerns)


def _supported_domain_phrase(evidence: EvidenceBundle, signal_name: str) -> str:
    snippets = " ".join(evidence.signals[signal_name].snippets).lower()
    exact = _exact_evidence_phrase(snippets)
    if exact:
        return exact
    if "recommend" in snippets:
        return ""
    if "ranking" in snippets or "ranker" in snippets:
        return ""
    if "search" in snippets:
        return ""
    if "retrieval" in snippets:
        return ""
    return ""


def _snippet_backed_phrase(evidence: EvidenceBundle, signal_name: str, fallback: str) -> str:
    signal = evidence.signals.get(signal_name)
    if not signal or not signal.snippets:
        return fallback
    snippets = " ".join(signal.snippets).lower()
    exact = _exact_evidence_phrase(snippets)
    if exact:
        return exact
    for phrase in (
        "discovery feed",
        "learning-to-rank",
        "learning to rank",
        "semantic search",
        "hybrid search",
        "dense retrieval",
        "offline-online",
        "a/b test",
        "a/b testing",
        "ndcg",
        "mrr",
        "map",
        "query expansion",
        "faiss",
        "qdrant",
        "pinecone",
        "weaviate",
        "milvus",
        "opensearch",
        "elasticsearch",
    ):
        if phrase in snippets:
            return f"{fallback} ({phrase})" if fallback else phrase
    return fallback


def _exact_evidence_phrase(snippets: str) -> str:
    phrase_groups = (
        (("50m", "queries"), "ranking pipeline serving 50M+ queries/month"),
        (("30m", "candidate corpus"), "30M+ candidate corpus"),
        (("8k qps",), "8K QPS serving scale"),
        (("sub-200ms", "p95"), "sub-200ms p95 latency"),
        (("sub 200ms", "p95"), "sub-200ms p95 latency"),
        (("faiss hnsw",), "FAISS HNSW retrieval"),
        (("bm25", "dense retrieval"), "BM25 + dense retrieval"),
        (("bm25", "faiss"), "BM25 + FAISS retrieval"),
        (("bm25", "elasticsearch"), "BM25/Elasticsearch retrieval"),
        (("ndcg", "mrr", "recall"), "NDCG/MRR/recall@K evaluation"),
        (("ndcg", "mrr"), "NDCG/MRR evaluation"),
        (("mrr", "map"), "MRR/MAP evaluation"),
        (("recall@k",), "recall@K evaluation"),
        (("precision@k",), "precision@K evaluation"),
        (("a/b engagement metrics",), "A/B engagement metrics"),
        (("offline-online", "a/b"), "offline-online A/B correlation"),
        (("offline online", "a/b"), "offline-online A/B correlation"),
        (("offline to online",), "offline-online correlation"),
        (("a/b testing",), "A/B testing"),
        (("a/b test",), "A/B testing"),
        (("ab testing",), "A/B testing"),
        (("recommendation system",), "production recommendation system"),
        (("discovery feed",), "discovery feed ranking"),
        (("semantic search",), "semantic search"),
        (("hybrid search",), "hybrid search"),
        (("dense retrieval",), "dense retrieval"),
        (("learning-to-rank",), "learning-to-rank"),
        (("learning to rank",), "learning to rank"),
        (("relevance judgments",), "relevance judgments"),
        (("human relevance judgments",), "human relevance judgments"),
        (("relevance labeling",), "relevance labeling"),
        (("relevance labels",), "relevance labels"),
        (("500k", "semantic search"), "semantic search over 500K documents"),
        (("500k", "documents"), "semantic search over 500K documents"),
        (("embedding versioning",), "embedding versioning"),
        (("index versioning",), "index versioning"),
        (("rollback paths",), "rollback paths"),
        (("time-to-shortlist",), "time-to-shortlist improvement"),
        (("recruiter engagement lift",), "recruiter engagement lift"),
    )
    for required_terms, label in phrase_groups:
        if all(term in snippets for term in required_terms):
            return label
    return ""


def _evidence_phrase_from_snippet(snippet: str) -> str:
    lowered = snippet.lower()
    return _exact_evidence_phrase(lowered)


def _exact_evidence_parts(evidence: EvidenceBundle, limit: int = 3) -> list[str]:
    ordered_signals = (
        "evaluation_framework_experience",
        "retrieval_search_ranking_experience",
        "vector_database_experience",
        "semantic_evaluation_score",
        "semantic_production_retrieval_score",
        "semantic_vector_hybrid_search_score",
    )
    parts: list[str] = []
    for signal_name in ordered_signals:
        signal = evidence.signals.get(signal_name)
        if signal is None or signal.value < 0.30:
            continue
        phrase = _exact_evidence_phrase(" ".join(signal.snippets).lower())
        if phrase:
            parts.append(phrase)
        if len(_dedupe(parts)) >= limit:
            break
    return _dedupe(parts)[:limit]


def _humanize_competency(name: str) -> str:
    return name.replace("_", " ")


def _opener(candidate_id: str, rank: int | None) -> str:
    if rank is not None and rank <= 10:
        options = STRONG_OPENERS
        offset = 0
    elif rank is not None and rank >= 80:
        options = CAUTIOUS_OPENERS
        offset = 7
    else:
        options = MID_OPENERS
        offset = 13
    return options[_stable_index(candidate_id, len(options), offset=offset)]


def _candidate_facts(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    signals = candidate.get("redrob_signals", {}) if isinstance(candidate.get("redrob_signals", {}), dict) else {}
    skills = candidate.get("skills", []) if isinstance(candidate.get("skills", []), list) else []
    career = candidate.get("career_history", []) if isinstance(candidate.get("career_history", []), list) else []
    skill_names = [
        str(skill.get("name", "")).strip().lower()
        for skill in skills
        if isinstance(skill, dict) and str(skill.get("name", "")).strip()
    ]
    product_company = None
    for role in career:
        if not isinstance(role, dict):
            continue
        industry = str(role.get("industry", "")).strip().lower()
        if industry in {
            "software",
            "fintech",
            "food delivery",
            "transportation",
            "saas",
            "marketplace",
            "edtech",
            "adtech",
            "insurance tech",
            "healthtech",
            "internet",
            "ai/ml",
            "ai",
            "ml",
            "e-commerce",
            "ecommerce",
        }:
            product_company = str(role.get("company", "")).strip() or None
            break
    return {
        "title": str(profile.get("current_title", "")).strip(),
        "skill_names": skill_names,
        "product_company": product_company,
        "notice_period_days": _safe_float_or_none(signals.get("notice_period_days")),
        "recruiter_response_rate": _safe_float_or_none(signals.get("recruiter_response_rate")),
    }


def _has_skill(facts: dict[str, Any], skill_name: str) -> bool:
    wanted = skill_name.lower()
    return any(wanted == skill or wanted in skill for skill in facts.get("skill_names", []))


def _humanize_flag(flag: str) -> str:
    text = flag.strip()
    if not text:
        return "profile has a verification or consistency concern"
    return text[0].lower() + text[1:]


def _dedupe(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = " ".join(part.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(part)
    return result


def _stable_index(candidate_id: str, modulo: int, offset: int = 0) -> int:
    total = sum(ord(char) for char in candidate_id) + offset
    return total % modulo


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_safe(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())[:450]
