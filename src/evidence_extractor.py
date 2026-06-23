"""Deterministic evidence extraction for explainable candidate ranking."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.normalization import NormalizedCandidate


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SignalEvidence:
    """One transparent evidence signal extracted from a candidate profile."""

    value: float
    snippets: tuple[str, ...]
    source_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    """Structured career evidence used by scoring and reasoning."""

    candidate_id: str
    signals: dict[str, SignalEvidence]

    def value(self, signal_name: str, default: float = 0.0) -> float:
        signal = self.signals.get(signal_name)
        return default if signal is None else signal.value


# Backward-compatible name for modules created in the initial skeleton.
EvidenceBundle = Evidence


AI_ML_TITLE_TERMS = {
    "ai engineer": 1.0,
    "machine learning engineer": 1.0,
    "ml engineer": 1.0,
    "applied scientist": 0.95,
    "data scientist": 0.85,
    "recommendation systems engineer": 1.0,
    "search engineer": 0.95,
    "nlp engineer": 0.95,
    "mlops engineer": 0.85,
    "data engineer": 0.45,
    "backend engineer": 0.30,
    "cloud engineer": 0.25,
}

SENIORITY_TERMS = {
    "founding": 1.0,
    "principal": 0.95,
    "staff": 0.9,
    "lead": 0.85,
    "senior": 0.8,
    "sr.": 0.8,
    "architect": 0.65,
}

PRODUCTION_TERMS = {
    "production",
    "deployed",
    "shipped",
    "launched",
    "owned",
    "operated",
    "scaled",
    "real users",
    "on-call",
}

AI_TERMS = {
    "machine learning",
    "ml",
    "ai",
    "model",
    "models",
    "llm",
    "nlp",
    "transformer",
    "recommendation",
    "ranking",
    "retrieval",
    "embedding",
}

RETRIEVAL_SEARCH_RANKING_TERMS = {
    "retrieval",
    "ranking",
    "ranker",
    "recommendation",
    "recommender",
    "search",
    "hybrid search",
    "bm25",
    "semantic search",
    "information retrieval",
}

VECTOR_DATABASE_TERMS = {
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "vector database",
    "vector db",
    "vector index",
    "ann index",
}

EVALUATION_TERMS = {
    "ndcg",
    "mrr",
    "map",
    "precision@",
    "recall@",
    "offline evaluation",
    "a/b",
    "ab test",
    "a/b test",
    "online experiment",
    "evaluation framework",
    "ranking metrics",
}

PYTHON_TERMS = {"python", "pandas", "numpy", "scikit-learn", "sklearn", "fastapi"}

NLP_IR_TERMS = {
    "nlp",
    "natural language",
    "information retrieval",
    "ir",
    "retrieval",
    "search",
    "semantic search",
    "embedding",
    "sentence transformers",
    "transformers",
}

DATA_ENGINEERING_TERMS = {
    "spark",
    "pyspark",
    "airflow",
    "kafka",
    "data pipeline",
    "data pipelines",
    "etl",
    "warehouse",
    "snowflake",
    "dbt",
    "apache beam",
    "feature pipeline",
    "feature engineering",
}

RECENT_LLM_DEMO_TERMS = {
    "langchain",
    "openai api",
    "anthropic api",
    "small rag side project",
    "side project",
    "tutorial",
    "self-learner",
    "online courses",
    "played with",
}

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
}

PRODUCT_INDUSTRIES = {
    "software",
    "fintech",
    "food delivery",
    "transportation",
    "e-commerce",
    "saas",
    "marketplace",
}

TARGET_LOCATIONS = {
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "mumbai",
    "bangalore",
    "bengaluru",
}


def extract_evidence(candidate: NormalizedCandidate | dict[str, Any]) -> Evidence:
    """Extract deterministic evidence used by scoring and reasoning.

    The extractor intentionally uses transparent dictionaries and phrase matching
    rather than opaque ML models. It accepts a ``NormalizedCandidate`` from the
    pipeline, but also tolerates raw candidate dictionaries for tests and EDA.
    """
    raw = _raw_candidate(candidate)
    candidate_id = str(raw.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("Candidate profile is missing candidate_id.")

    LOGGER.debug("Extracting evidence for candidate_id=%s", candidate_id)
    signals = {
        "total_years_experience": _total_years(raw),
        "ai_ml_title_strength": _ai_ml_title_strength(raw),
        "seniority_strength": _seniority_strength(raw),
        "production_ai_experience": _production_ai_experience(raw),
        "retrieval_search_ranking_experience": _keyword_signal(
            raw,
            "retrieval_search_ranking_experience",
            RETRIEVAL_SEARCH_RANKING_TERMS,
            ("profile.summary", "profile.headline", "career_history[].description", "skills[].name"),
        ),
        "vector_database_experience": _keyword_signal(
            raw,
            "vector_database_experience",
            VECTOR_DATABASE_TERMS,
            ("profile.summary", "career_history[].description", "skills[].name", "redrob_signals.skill_assessment_scores"),
        ),
        "evaluation_framework_experience": _keyword_signal(
            raw,
            "evaluation_framework_experience",
            EVALUATION_TERMS,
            ("profile.summary", "career_history[].description", "skills[].name"),
        ),
        "python_strength": _python_strength(raw),
        "product_company_experience": _product_company_experience(raw),
        "services_only_penalty_signal": _services_only_penalty_signal(raw),
        "recent_hands_on_coding_signal": _recent_hands_on_coding_signal(raw),
        "llm_only_recent_demo_signal": _llm_only_recent_demo_signal(raw),
        "pre_llm_ml_experience_signal": _pre_llm_ml_experience_signal(raw),
        "nlp_ir_relevance": _keyword_signal(
            raw,
            "nlp_ir_relevance",
            NLP_IR_TERMS,
            ("profile.summary", "profile.headline", "career_history[].description", "skills[].name"),
        ),
        "data_engineering_adjacent_strength": _keyword_signal(
            raw,
            "data_engineering_adjacent_strength",
            DATA_ENGINEERING_TERMS,
            ("profile.summary", "career_history[].description", "skills[].name"),
        ),
        "open_source_or_github_signal": _open_source_or_github_signal(raw),
        "education_strength": _education_strength(raw),
        "location_fit": _location_fit(raw),
        "notice_period_fit": _notice_period_fit(raw),
        "availability_signal": _availability_signal(raw),
        "recruiter_engagement_signal": _recruiter_engagement_signal(raw),
        "profile_trust_signal": _profile_trust_signal(raw),
    }
    return Evidence(candidate_id=candidate_id, signals=signals)


def _raw_candidate(candidate: NormalizedCandidate | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, NormalizedCandidate):
        return candidate.raw_profile
    return candidate


def _signal(value: float, snippets: list[str], source_fields: list[str]) -> SignalEvidence:
    return SignalEvidence(
        value=round(float(max(0.0, min(value, 1.0))), 4),
        snippets=tuple(snippets[:5]),
        source_fields=tuple(dict.fromkeys(source_fields)),
    )


def _unbounded_signal(value: float, snippets: list[str], source_fields: list[str]) -> SignalEvidence:
    return SignalEvidence(
        value=round(float(max(0.0, value)), 4),
        snippets=tuple(snippets[:5]),
        source_fields=tuple(dict.fromkeys(source_fields)),
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _profile(raw: dict[str, Any]) -> dict[str, Any]:
    profile = raw.get("profile", {})
    return profile if isinstance(profile, dict) else {}


def _signals(raw: dict[str, Any]) -> dict[str, Any]:
    signals = raw.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _career(raw: dict[str, Any]) -> list[dict[str, Any]]:
    career = raw.get("career_history", [])
    return [item for item in career if isinstance(item, dict)] if isinstance(career, list) else []


def _education(raw: dict[str, Any]) -> list[dict[str, Any]]:
    education = raw.get("education", [])
    return [item for item in education if isinstance(item, dict)] if isinstance(education, list) else []


def _skills(raw: dict[str, Any]) -> list[dict[str, Any]]:
    skills = raw.get("skills", [])
    return [item for item in skills if isinstance(item, dict)] if isinstance(skills, list) else []


def _skill_names(raw: dict[str, Any]) -> list[str]:
    return [_text(skill.get("name")) for skill in _skills(raw) if _text(skill.get("name"))]


def _snippet(text: str, term: str, max_chars: int = 180) -> str:
    clean = " ".join(text.split())
    if not clean:
        return ""
    index = clean.lower().find(term.lower())
    if index < 0:
        return clean[:max_chars]
    start = max(0, index - 60)
    end = min(len(clean), index + len(term) + 90)
    return clean[start:end]


def _field_texts(raw: dict[str, Any]) -> list[tuple[str, str]]:
    profile = _profile(raw)
    fields = [
        ("profile.current_title", _text(profile.get("current_title"))),
        ("profile.headline", _text(profile.get("headline"))),
        ("profile.summary", _text(profile.get("summary"))),
        ("profile.current_industry", _text(profile.get("current_industry"))),
    ]
    for index, role in enumerate(_career(raw)):
        fields.extend(
            [
                (f"career_history[{index}].title", _text(role.get("title"))),
                (f"career_history[{index}].company", _text(role.get("company"))),
                (f"career_history[{index}].industry", _text(role.get("industry"))),
                (f"career_history[{index}].description", _text(role.get("description"))),
            ]
        )
    for index, skill in enumerate(_skills(raw)):
        fields.append((f"skills[{index}].name", _text(skill.get("name"))))
    assessments = _signals(raw).get("skill_assessment_scores", {})
    if isinstance(assessments, dict):
        for name in assessments:
            fields.append(("redrob_signals.skill_assessment_scores", _text(name)))
    return [(field, text) for field, text in fields if text]


def _match_terms(raw: dict[str, Any], terms: set[str]) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    for field, text in _field_texts(raw):
        for term in terms:
            if _contains_term(text, term):
                matches.append((field, term, _snippet(text, term)))
                break
    return matches


def _contains_term(text: str, term: str) -> bool:
    lowered_text = text.lower()
    lowered_term = term.lower()
    if lowered_term not in lowered_text:
        return False
    if len(lowered_term) <= 3:
        return re.search(rf"(?<!\w){re.escape(lowered_term)}(?!\w)", lowered_text) is not None
    return True


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _source_matches(pattern: str, field: str) -> bool:
    if "[]" not in pattern:
        return field == pattern or field.startswith(pattern)
    prefix, suffix = pattern.split("[]", 1)
    return field.startswith(prefix + "[") and field.endswith(suffix)


def _keyword_signal(
    raw: dict[str, Any],
    _name: str,
    terms: set[str],
    preferred_sources: tuple[str, ...],
) -> SignalEvidence:
    matches = _match_terms(raw, terms)
    preferred = [
        item
        for item in matches
        if any(_source_matches(source, item[0]) for source in preferred_sources)
    ]
    selected = preferred or matches
    role_hits = sum(1 for field, _term, _snippet_text in selected if field.startswith("career_history"))
    skill_hits = sum(1 for field, _term, _snippet_text in selected if field.startswith("skills") or "assessment" in field)
    profile_hits = sum(1 for field, _term, _snippet_text in selected if field.startswith("profile"))
    value = min(1.0, role_hits * 0.35 + skill_hits * 0.18 + profile_hits * 0.22)
    return _signal(
        value,
        [snippet for _field, _term, snippet in selected],
        [field for field, _term, _snippet_text in selected],
    )


def _total_years(raw: dict[str, Any]) -> SignalEvidence:
    years = _safe_float(_profile(raw).get("years_of_experience"))
    return _unbounded_signal(years, [f"{years:g} years of experience"], ["profile.years_of_experience"])


def _ai_ml_title_strength(raw: dict[str, Any]) -> SignalEvidence:
    profile = _profile(raw)
    titles = [("profile.current_title", _text(profile.get("current_title")))]
    titles.extend((f"career_history[{i}].title", _text(role.get("title"))) for i, role in enumerate(_career(raw)))
    best = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    for field, title in titles:
        lowered = title.lower()
        for term, weight in AI_ML_TITLE_TERMS.items():
            if term in lowered and weight > best:
                best = weight
                snippets = [title]
                sources = [field]
    return _signal(best, snippets, sources)


def _seniority_strength(raw: dict[str, Any]) -> SignalEvidence:
    title = _text(_profile(raw).get("current_title"))
    lowered = title.lower()
    value = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    for term, weight in SENIORITY_TERMS.items():
        if term in lowered:
            value = max(value, weight)
            snippets.append(title)
            sources.append("profile.current_title")
    years = _safe_float(_profile(raw).get("years_of_experience"))
    if 5 <= years <= 9:
        value = max(value, 0.75)
        snippets.append(f"{years:g} years aligns with the JD target band")
        sources.append("profile.years_of_experience")
    elif 9 < years <= 12:
        value = max(value, 0.6)
        snippets.append(f"{years:g} years is above target but plausibly senior")
        sources.append("profile.years_of_experience")
    elif 3.5 <= years < 5:
        value = max(value, 0.45)
        snippets.append(f"{years:g} years is slightly below the target band")
        sources.append("profile.years_of_experience")
    return _signal(value, snippets, sources)


def _production_ai_experience(raw: dict[str, Any]) -> SignalEvidence:
    snippets: list[str] = []
    sources: list[str] = []
    value = 0.0
    for field, text in _field_texts(raw):
        lowered = text.lower()
        if _contains_any(lowered, PRODUCTION_TERMS) and _contains_any(lowered, AI_TERMS):
            value += 0.35 if field.startswith("career_history") else 0.22
            snippets.append(_snippet(text, "production" if "production" in lowered else "model"))
            sources.append(field)
    for skill in _skill_names(raw):
        if _contains_any(skill, {"mlops", "bentoml", "kubeflow", "mlflow"}):
            value += 0.12
            snippets.append(skill)
            sources.append("skills[].name")
    return _signal(value, snippets, sources)


def _python_strength(raw: dict[str, Any]) -> SignalEvidence:
    matches = _match_terms(raw, PYTHON_TERMS)
    value = 0.0
    snippets = [snippet for _field, _term, snippet in matches]
    sources = [field for field, _term, _snippet_text in matches]
    for skill in _skills(raw):
        if _lower(skill.get("name")) == "python":
            proficiency = _lower(skill.get("proficiency"))
            months = _safe_float(skill.get("duration_months"))
            value = max(value, {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6}.get(proficiency, 0.35))
            if months >= 24:
                value = min(1.0, value + 0.1)
            snippets.append(f"Python skill listed as {proficiency or 'unknown'} for {months:g} months")
            sources.append("skills[].name")
    if matches:
        value = max(value, min(1.0, 0.25 + len(matches) * 0.15))
    return _signal(value, snippets, sources)


def _product_company_experience(raw: dict[str, Any]) -> SignalEvidence:
    snippets: list[str] = []
    sources: list[str] = []
    months = 0
    for index, role in enumerate(_career(raw)):
        industry = _lower(role.get("industry"))
        company = _text(role.get("company"))
        if industry in PRODUCT_INDUSTRIES or any(term in industry for term in PRODUCT_INDUSTRIES):
            duration = int(_safe_float(role.get("duration_months")))
            months += duration
            snippets.append(f"{company} in {role.get('industry')} for {duration} months")
            sources.append(f"career_history[{index}].industry")
    value = min(1.0, months / 48)
    return _signal(value, snippets, sources)


def _services_only_penalty_signal(raw: dict[str, Any]) -> SignalEvidence:
    career = _career(raw)
    if not career:
        return _signal(0.0, [], [])
    service_roles = []
    for index, role in enumerate(career):
        company = _lower(role.get("company"))
        industry = _lower(role.get("industry"))
        is_service = company in SERVICES_COMPANIES or industry == "it services"
        service_roles.append(is_service)
    if all(service_roles):
        snippets = [
            f"{role.get('company')} / {role.get('industry')}"
            for role in career[:5]
        ]
        return _signal(1.0, snippets, ["career_history[].company", "career_history[].industry"])
    if any(service_roles):
        return _signal(0.4, ["Some career history is in IT services"], ["career_history[].industry"])
    return _signal(0.0, [], [])


def _recent_hands_on_coding_signal(raw: dict[str, Any]) -> SignalEvidence:
    career = _career(raw)
    current = [role for role in career if role.get("is_current") is True]
    role = current[0] if current else (career[0] if career else {})
    text = " ".join([_text(role.get("title")), _text(role.get("description"))])
    lowered = text.lower()
    coding_terms = {"implemented", "built", "developed", "coded", "python", "api", "pipeline", "deployed"}
    manager_terms = {"manager", "director", "head of", "program manager", "project manager"}
    value = 0.0
    if any(term in lowered for term in coding_terms):
        value += 0.7
    if any(term in lowered for term in {"engineer", "developer", "scientist"}):
        value += 0.2
    if any(term in lowered for term in manager_terms) and not any(term in lowered for term in coding_terms):
        value = min(value, 0.25)
    source = "career_history[].description" if role else ""
    return _signal(value, [_snippet(text, "built") or text], [source] if source else [])


def _llm_only_recent_demo_signal(raw: dict[str, Any]) -> SignalEvidence:
    snippets: list[str] = []
    sources: list[str] = []
    value = 0.0
    matches = _match_terms(raw, RECENT_LLM_DEMO_TERMS)
    has_production_ai = _production_ai_experience(raw).value >= 0.5
    has_pre_llm = _pre_llm_ml_experience_signal(raw).value >= 0.5
    if matches and not has_production_ai and not has_pre_llm:
        value = min(1.0, 0.35 + len(matches) * 0.15)
        snippets = [snippet for _field, _term, snippet in matches]
        sources = [field for field, _term, _snippet_text in matches]
    elif matches:
        value = 0.25
        snippets = [snippet for _field, _term, snippet in matches[:2]]
        sources = [field for field, _term, _snippet_text in matches[:2]]
    return _signal(value, snippets, sources)


def _pre_llm_ml_experience_signal(raw: dict[str, Any]) -> SignalEvidence:
    snippets: list[str] = []
    sources: list[str] = []
    value = 0.0
    for index, role in enumerate(_career(raw)):
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) or date.today()
        text = " ".join([_text(role.get("title")), _text(role.get("description"))])
        lowered = text.lower()
        if start and start.year < 2022 and _contains_any(lowered, AI_TERMS):
            duration = int(_safe_float(role.get("duration_months")))
            value += 0.35 if duration >= 18 else 0.2
            snippets.append(_snippet(text, "machine learning") or _snippet(text, "model"))
            sources.append(f"career_history[{index}].description")
        elif end.year < 2022 and _contains_any(lowered, AI_TERMS):
            value += 0.25
            snippets.append(_snippet(text, "model"))
            sources.append(f"career_history[{index}].description")
    return _signal(value, snippets, sources)


def _open_source_or_github_signal(raw: dict[str, Any]) -> SignalEvidence:
    signals = _signals(raw)
    github_score = _safe_float(signals.get("github_activity_score"))
    text_matches = _match_terms(raw, {"open-source", "open source", "oss", "contribution", "pull request"})
    value = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    if github_score >= 0:
        value = min(1.0, github_score / 75)
        snippets.append(f"GitHub activity score {github_score:g}")
        sources.append("redrob_signals.github_activity_score")
    if text_matches:
        value = max(value, 0.5)
        snippets.extend(snippet for _field, _term, snippet in text_matches)
        sources.extend(field for field, _term, _snippet_text in text_matches)
    return _signal(value, snippets, sources)


def _education_strength(raw: dict[str, Any]) -> SignalEvidence:
    best = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    tier_weights = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.4, "tier_4": 0.2}
    for index, item in enumerate(_education(raw)):
        tier = _lower(item.get("tier"))
        field = _lower(item.get("field_of_study"))
        degree = _lower(item.get("degree"))
        value = tier_weights.get(tier, 0.1)
        if _contains_any(field, {"computer", "machine learning", "data science", "ai", "statistics"}):
            value = min(1.0, value + 0.2)
        if any(term in degree for term in {"master", "m.tech", "m.s", "phd"}):
            value = min(1.0, value + 0.1)
        if value > best:
            best = value
        snippets.append(f"{item.get('degree')} in {item.get('field_of_study')} at {item.get('institution')} ({item.get('tier')})")
        sources.append(f"education[{index}]")
    return _signal(best, snippets, sources)


def _location_fit(raw: dict[str, Any]) -> SignalEvidence:
    profile = _profile(raw)
    signals = _signals(raw)
    location = " ".join([_lower(profile.get("location")), _lower(profile.get("country"))])
    value = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    if "india" in location:
        value = 0.45
        snippets.append(f"{profile.get('location')}, {profile.get('country')}")
        sources.extend(["profile.location", "profile.country"])
    if any(city in location for city in TARGET_LOCATIONS):
        value = max(value, 0.85)
    if any(city in location for city in {"pune", "noida"}):
        value = 1.0
    if signals.get("willing_to_relocate") is True:
        value = min(1.0, value + 0.2)
        snippets.append("Willing to relocate")
        sources.append("redrob_signals.willing_to_relocate")
    return _signal(value, snippets, sources)


def _notice_period_fit(raw: dict[str, Any]) -> SignalEvidence:
    days = _safe_float(_signals(raw).get("notice_period_days"))
    if days <= 30:
        value = 1.0
    elif days <= 60:
        value = 0.65
    elif days <= 90:
        value = 0.35
    else:
        value = 0.1
    return _signal(value, [f"{days:g} day notice period"], ["redrob_signals.notice_period_days"])


def _availability_signal(raw: dict[str, Any]) -> SignalEvidence:
    signals = _signals(raw)
    value = 0.0
    snippets: list[str] = []
    sources: list[str] = []
    if signals.get("open_to_work_flag") is True:
        value += 0.35
        snippets.append("Open to work")
        sources.append("redrob_signals.open_to_work_flag")
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days = (date.today() - last_active).days
        if days <= 30:
            value += 0.35
        elif days <= 90:
            value += 0.22
        elif days <= 180:
            value += 0.1
        snippets.append(f"Last active {days} days ago")
        sources.append("redrob_signals.last_active_date")
    value += _notice_period_fit(raw).value * 0.2
    if signals.get("willing_to_relocate") is True:
        value += 0.1
        sources.append("redrob_signals.willing_to_relocate")
    return _signal(value, snippets, sources)


def _recruiter_engagement_signal(raw: dict[str, Any]) -> SignalEvidence:
    signals = _signals(raw)
    response_rate = _safe_float(signals.get("recruiter_response_rate"))
    response_time = _safe_float(signals.get("avg_response_time_hours"))
    saved = _safe_float(signals.get("saved_by_recruiters_30d"))
    views = _safe_float(signals.get("profile_views_received_30d"))
    interview_completion = _safe_float(signals.get("interview_completion_rate"))
    value = 0.0
    value += min(1.0, response_rate) * 0.35
    value += (1.0 if response_time <= 24 else 0.65 if response_time <= 72 else 0.25 if response_time <= 168 else 0.05) * 0.2
    value += min(1.0, saved / 10) * 0.2
    value += min(1.0, views / 100) * 0.1
    value += min(1.0, interview_completion) * 0.15
    snippets = [
        f"response_rate={response_rate:g}",
        f"avg_response_time_hours={response_time:g}",
        f"saved_by_recruiters_30d={saved:g}",
        f"profile_views_received_30d={views:g}",
        f"interview_completion_rate={interview_completion:g}",
    ]
    return _signal(
        value,
        snippets,
        [
            "redrob_signals.recruiter_response_rate",
            "redrob_signals.avg_response_time_hours",
            "redrob_signals.saved_by_recruiters_30d",
            "redrob_signals.profile_views_received_30d",
            "redrob_signals.interview_completion_rate",
        ],
    )


def _profile_trust_signal(raw: dict[str, Any]) -> SignalEvidence:
    signals = _signals(raw)
    completeness = _safe_float(signals.get("profile_completeness_score"))
    value = min(1.0, completeness / 100) * 0.45
    snippets = [f"profile_completeness_score={completeness:g}"]
    sources = ["redrob_signals.profile_completeness_score"]
    for field in ("verified_email", "verified_phone", "linkedin_connected"):
        if signals.get(field) is True:
            value += 0.18 if field != "linkedin_connected" else 0.19
            snippets.append(f"{field}=true")
            sources.append(f"redrob_signals.{field}")
    return _signal(value, snippets, sources)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None
