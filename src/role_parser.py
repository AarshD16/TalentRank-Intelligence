"""Deterministic job-description to RoleSpec parser."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from src.rolespec import RoleSpec, default_redrob_ai_engineer_rolespec


COMPETENCY_PHRASES: dict[str, tuple[str, ...]] = {
    "ai_ml": ("ai engineer", "machine learning", "ml systems", "applied ml", "nlp"),
    "retrieval_search": ("retrieval", "search", "semantic search", "embeddings-based retrieval"),
    "ranking_recommendation": ("ranking", "recommendation", "matching systems", "ranker"),
    "vector_hybrid_search": ("vector database", "hybrid search", "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch", "elasticsearch"),
    "ranking_evaluation": ("ndcg", "mrr", "map", "evaluation framework", "a/b testing", "offline", "online"),
    "mlops_production": ("production", "deployed", "real users", "mlflow", "model monitoring", "drift"),
    "python_engineering": ("python", "code quality"),
    "product_company": ("product", "users", "pm", "marketplace"),
    "data_engineering": ("data pipeline", "feature pipeline", "spark", "airflow"),
    "backend_systems": ("backend", "distributed systems", "infrastructure"),
    "management": ("mentoring", "growing the team"),
    "research_only": ("pure research", "academic labs", "research-only"),
    "services_only": ("consulting firms", "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"),
    "keyword_stuffing": ("langchain tutorials", "hot framework", "ai keywords"),
}


def parse_rolespec_from_docx(path: Path) -> RoleSpec:
    """Parse a JD docx into a deterministic RoleSpec."""
    text = _extract_docx_text(path)
    lowered = text.lower()
    base = default_redrob_ai_engineer_rolespec()
    title = _parse_title_clean(text) or base.role_title
    experience_range = _parse_experience_range_clean(lowered) or base.experience_range
    present = {
        competency
        for competency, phrases in COMPETENCY_PHRASES.items()
        if any(phrase in lowered for phrase in phrases)
    }
    must_have = tuple(
        item
        for item in (
            "retrieval_search",
            "ranking_recommendation",
            "vector_hybrid_search",
            "ranking_evaluation",
            "python_engineering",
            "mlops_production",
        )
        if item in present
    )
    preferred = tuple(
        item
        for item in ("product_company", "data_engineering", "backend_systems", "ai_ml", "behavioral_availability")
        if item in present or item == "behavioral_availability"
    )
    negative = tuple(item for item in ("research_only", "services_only", "keyword_stuffing") if item in present)
    weights = _weights_from_present(present)
    return RoleSpec(
        role_title=title,
        target_seniority="senior" if "senior" in lowered or "5-9" in lowered else base.target_seniority,
        experience_range=experience_range,
        must_have_competencies=must_have or base.must_have_competencies,
        preferred_competencies=preferred or base.preferred_competencies,
        negative_signals=negative or base.negative_signals,
        logistics_preferences=base.logistics_preferences,
        scoring_weights=weights,
    )


def _extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _parse_title(text: str) -> str | None:
    for line in text.splitlines()[:10]:
        if "Job Description:" in line:
            return line.split("Job Description:", 1)[1].strip().split("—", 1)[0].strip()
    return None


def _parse_experience_range(text: str) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*years", text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def _parse_title_robust(text: str) -> str | None:
    for line in text.splitlines()[:10]:
        if "Job Description:" in line:
            title_part = line.split("Job Description:", 1)[1].strip()
            dash_pattern = "|".join(("-", "--", chr(8211), chr(8212), "â€“", "â€”"))
            return re.split(rf"\s+(?:{dash_pattern})\s+", title_part, maxsplit=1)[0].strip()
    return None


def _parse_experience_range_robust(text: str) -> tuple[float, float] | None:
    dash_pattern = "|".join(("-", chr(8211), "â€“"))
    match = re.search(rf"(\d+(?:\.\d+)?)\s*(?:{dash_pattern})\s*(\d+(?:\.\d+)?)\s*years", text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def _parse_title_clean(text: str) -> str | None:
    for line in text.splitlines()[:10]:
        if "Job Description:" in line:
            title_part = line.split("Job Description:", 1)[1].strip()
            separators = ("--", "-", chr(8211), chr(8212))
            pattern = "|".join(re.escape(separator) for separator in separators)
            return re.split(rf"\s+(?:{pattern})\s+", title_part, maxsplit=1)[0].strip()
    return None


def _parse_experience_range_clean(text: str) -> tuple[float, float] | None:
    separators = ("-", chr(8211))
    pattern = "|".join(re.escape(separator) for separator in separators)
    match = re.search(rf"(\d+(?:\.\d+)?)\s*(?:{pattern})\s*(\d+(?:\.\d+)?)\s*years", text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def _weights_from_present(present: set[str]) -> dict[str, float]:
    weights = {
        "retrieval_search": 0.16,
        "ranking_recommendation": 0.16,
        "ranking_evaluation": 0.13,
        "mlops_production": 0.13,
        "python_engineering": 0.10,
        "ai_ml": 0.08,
        "product_company": 0.08,
        "vector_hybrid_search": 0.07,
        "behavioral_availability": 0.05,
        "data_engineering": 0.02,
        "backend_systems": 0.01,
        "trust_consistency": 0.01,
    }
    for competency in present:
        if competency in weights:
            weights[competency] *= 1.08
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}
