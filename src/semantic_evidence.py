"""Lightweight semantic/JD intent evidence without external model calls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)?")

JD_INTENT_BLOCKS: dict[str, tuple[str, ...]] = {
    "production_retrieval_ranking": (
        "production retrieval ranking recommendation search discovery feed personalization learning to rank",
        "owned shipped operated ranking layer recommendation system discovery feed search product real users",
        "click through dwell time engagement revenue per search downstream conversion relevance labeling",
        "rag based ranking pipeline recruiter feedback loop time to shortlist recruiter engagement lift",
    ),
    "vector_hybrid_search": (
        "semantic search dense retrieval hybrid search vector database embedding based retrieval",
        "faiss pinecone weaviate qdrant milvus opensearch elasticsearch pgvector vector index index refresh",
        "sentence transformers bge e5 nearest neighbor query expansion embedding drift",
        "bm25 dense retrieval faiss hnsw embedding versioning index versioning rollback paths",
    ),
    "evaluation_frameworks": (
        "ndcg mrr map precision recall offline online evaluation a/b testing ab testing",
        "offline online correlation ranking metrics human judgments relevance labels evaluation workflow",
        "benchmarks recruiter feedback experiments test interpretation model monitoring drift checks",
        "recall@k precision@k human relevance judgments relevance labeling a/b engagement metrics",
    ),
    "senior_ai_engineering": (
        "senior ai engineer machine learning engineer applied ml engineer nlp engineer recommendation systems engineer",
        "mentored owned led designed architecture production python ml systems feature pipeline",
    ),
    "product_shipping_mindset": (
        "shipped launched product pm users production load improved retention conversion engagement",
        "scrappy product engineering real users operational monitoring retraining schedules alerting",
    ),
}

INTENT_WEIGHTS: dict[str, float] = {
    "production_retrieval_ranking": 0.32,
    "vector_hybrid_search": 0.23,
    "evaluation_frameworks": 0.22,
    "senior_ai_engineering": 0.13,
    "product_shipping_mindset": 0.10,
}


@dataclass(frozen=True, slots=True)
class SemanticIntentResult:
    """BM25-like intent evidence for one candidate."""

    intent_scores: dict[str, float]
    snippets: dict[str, tuple[str, ...]]

    @property
    def overall_score(self) -> float:
        return round(
            sum(self.intent_scores.get(name, 0.0) * weight for name, weight in INTENT_WEIGHTS.items()),
            4,
        )


def extract_semantic_intents(candidate: dict[str, Any]) -> SemanticIntentResult:
    """Compute transparent BM25-like similarity against JD intent blocks.

    This is intentionally local and deterministic. It is not an embedding model;
    it scores saturated term overlap between candidate career text and each JD
    intent block, with snippets retained for explanation.
    """
    field_texts = _candidate_career_texts(candidate)
    full_text = " ".join(text for _field, text in field_texts).lower()
    doc_tokens = _tokenize(full_text)
    token_counts: dict[str, int] = {}
    for token in doc_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    doc_len = max(1, len(doc_tokens))

    scores: dict[str, float] = {}
    snippets: dict[str, tuple[str, ...]] = {}
    for intent_name, blocks in JD_INTENT_BLOCKS.items():
        intent_terms = _intent_terms(blocks)
        raw_score = _bm25_like(token_counts, doc_len, intent_terms)
        phrase_boost, phrase_snippets = _phrase_evidence(field_texts, blocks)
        scores[intent_name] = round(min(1.0, raw_score + phrase_boost), 4)
        snippets[intent_name] = tuple(phrase_snippets[:4])
    return SemanticIntentResult(intent_scores=scores, snippets=snippets)


def optional_local_embedding_scores(
    candidate_texts: list[str],
    intent_texts: list[str],
    model_path: str | None = None,
) -> list[float] | None:
    """Optional local sentence-transformer scoring for offline experiments.

    This is deliberately not called by the final ranking path. It performs no
    network access and only works when ``sentence_transformers`` and a local
    model path are available.
    """
    if not model_path:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return None
    model = SentenceTransformer(model_path, local_files_only=True)
    candidate_embeddings = model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False)
    intent_embeddings = model.encode(intent_texts, normalize_embeddings=True, show_progress_bar=False)
    scores: list[float] = []
    for candidate_embedding in candidate_embeddings:
        best = max(float(candidate_embedding @ intent_embedding) for intent_embedding in intent_embeddings)
        scores.append(best)
    return scores


def _candidate_career_texts(candidate: dict[str, Any]) -> list[tuple[str, str]]:
    profile = candidate.get("profile", {}) if isinstance(candidate.get("profile", {}), dict) else {}
    texts = [
        ("profile.current_title", str(profile.get("current_title", "")).strip()),
        ("profile.headline", str(profile.get("headline", "")).strip()),
        ("profile.summary", str(profile.get("summary", "")).strip()),
    ]
    career = candidate.get("career_history", [])
    if isinstance(career, list):
        for index, role in enumerate(career):
            if not isinstance(role, dict):
                continue
            texts.extend(
                [
                    (f"career_history[{index}].title", str(role.get("title", "")).strip()),
                    (f"career_history[{index}].description", str(role.get("description", "")).strip()),
                    (f"career_history[{index}].industry", str(role.get("industry", "")).strip()),
                ]
            )
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        skill_text = " ".join(str(skill.get("name", "")).strip() for skill in skills if isinstance(skill, dict))
        texts.append(("skills[].name", skill_text))
    return [(field, text) for field, text in texts if text]


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(token)
        if "-" in token or "_" in token or "/" in token:
            tokens.extend(part for part in re.split(r"[-_/]", token) if part)
    return tokens


def _intent_terms(blocks: tuple[str, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for block in blocks:
        for token in _tokenize(block):
            if len(token) > 1:
                seen[token] = None
    return tuple(seen)


def _bm25_like(token_counts: dict[str, int], doc_len: int, query_terms: tuple[str, ...]) -> float:
    if not query_terms:
        return 0.0
    k1 = 1.2
    b = 0.75
    avg_doc_len = 180.0
    score = 0.0
    for term in query_terms:
        tf = token_counts.get(term, 0)
        if not tf:
            continue
        idf = 1.0 + math.log(1.0 + 1.0 / (1.0 + min(tf, 5)))
        denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
        score += idf * (tf * (k1 + 1.0)) / denom
    return min(0.82, score / max(6.0, len(query_terms) * 0.42))


def _phrase_evidence(field_texts: list[tuple[str, str]], blocks: tuple[str, ...]) -> tuple[float, list[str]]:
    phrases = _important_phrases(blocks)
    boost = 0.0
    snippets: list[str] = []
    for _field, text in field_texts:
        lowered = text.lower()
        for phrase in phrases:
            if phrase in lowered:
                boost += 0.08 if "career_history" in _field else 0.04
                snippets.append(_snippet(text, phrase))
                break
    return min(0.32, boost), snippets


def _important_phrases(blocks: tuple[str, ...]) -> tuple[str, ...]:
    phrases = {
        "semantic search",
        "dense retrieval",
        "hybrid search",
        "vector database",
        "embedding-based retrieval",
        "embedding based retrieval",
        "discovery feed",
        "learning-to-rank",
        "learning to rank",
        "offline-online",
        "offline online",
        "a/b test",
        "a/b testing",
        "ab testing",
        "ndcg",
        "mrr",
        "map",
        "recommendation system",
        "ranking layer",
        "query expansion",
        "index refresh",
        "embedding drift",
        "embedding versioning",
        "index versioning",
        "rollback paths",
        "faiss hnsw",
        "bm25 + dense retrieval",
        "human relevance judgments",
        "relevance labeling",
        "a/b engagement metrics",
        "offline to online",
        "recruiter feedback loop",
        "time-to-shortlist",
        "recruiter engagement lift",
    }
    block_text = " ".join(blocks).lower()
    return tuple(phrase for phrase in phrases if phrase in block_text)


def _snippet(text: str, phrase: str, max_chars: int = 190) -> str:
    clean = " ".join(text.split())
    index = clean.lower().find(phrase)
    if index < 0:
        return clean[:max_chars]
    start = max(0, index - 70)
    end = min(len(clean), index + len(phrase) + 100)
    return clean[start:end]
