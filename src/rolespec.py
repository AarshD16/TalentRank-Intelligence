"""Validated RoleSpec model and YAML persistence.

The RoleSpec is the only artifact an LLM is allowed to generate. It is data,
not executable code, and ranking remains deterministic after this file is
loaded.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COMPETENCY_TAXONOMY = (
    "ai_ml_engineering",
    "retrieval_search",
    "ranking_recommendation",
    "ranking_evaluation",
    "vector_hybrid_search",
    "production_ml_systems",
    "python_engineering",
    "backend_systems",
    "data_engineering",
    "frontend_engineering",
    "devops_cloud",
    "product_management",
    "sales",
    "marketing",
    "design",
    "management",
    "research",
    "consulting_services",
    "open_source_validation",
)

NEGATIVE_SIGNALS = (
    "keyword_stuffing",
    "skills_only_match",
    "unrelated_title",
    "inactive_low_response",
    "pure_research_without_deployment",
    "consulting_only_background",
)

SCORE_CAP_RULES = (
    "missing_must_have_production_evidence",
    "no_production_evidence",
    "skills_only_match",
    "unrelated_title",
    "unrelated_title_and_history",
    "inactive_candidate",
    "inactive_low_response",
    "pure_research_without_deployment",
    "consulting_only_background",
    "no_ranking_search_recommendation_evidence",
    "no_ranking_evaluation_evidence",
    "generic_ml_without_retrieval_or_evaluation",
    "recent_llm_demo_only",
)

REDROB_SIGNAL_NAMES = (
    "recruiter_response_rate",
    "avg_response_time_hours",
    "last_active_date",
    "open_to_work_flag",
    "notice_period_days",
    "interview_completion_rate",
    "offer_acceptance_rate",
    "profile_completeness_score",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
    "github_activity_score",
    "saved_by_recruiters_30d",
    "search_appearance_30d",
)


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """Minimum evidence thresholds for one competency or capability."""

    name: str
    min_score: float = 0.35
    required_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoreCap:
    """A validated score cap rule from RoleSpec."""

    rule: str
    max_score: float
    applies_to: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Structured representation of a job description."""

    role_title: str
    role_family: str
    target_seniority: str
    experience_range: tuple[float, float]
    must_have_competencies: tuple[str, ...]
    preferred_competencies: tuple[str, ...]
    negative_signals: tuple[str, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...]
    score_caps: tuple[ScoreCap, ...]
    redrob_signal_policy: dict[str, Any]
    reasoning_priorities: tuple[str, ...]
    scoring_weights: dict[str, float] = field(default_factory=dict)
    logistics_preferences: dict[str, Any] = field(default_factory=dict)


LEGACY_COMPETENCY_ALIASES = {
    "ai_ml": "ai_ml_engineering",
    "mlops_production": "production_ml_systems",
    "frontend": "frontend_engineering",
    "devops": "devops_cloud",
    "research_only": "research",
    "services_only": "consulting_services",
    "behavioral_availability": "open_source_validation",
    "trust_consistency": "open_source_validation",
    "product_company": "product_management",
}


def default_redrob_ai_engineer_rolespec() -> RoleSpec:
    """Default RoleSpec matching the supplied Redrob Senior AI Engineer JD."""
    return RoleSpec(
        role_title="Senior AI Engineer",
        role_family="ai_ml_search",
        target_seniority="senior",
        experience_range=(5.0, 9.0),
        must_have_competencies=(
            "ai_ml_engineering",
            "retrieval_search",
            "ranking_recommendation",
            "ranking_evaluation",
            "production_ml_systems",
            "python_engineering",
        ),
        preferred_competencies=(
            "vector_hybrid_search",
            "backend_systems",
            "data_engineering",
            "open_source_validation",
        ),
        negative_signals=("keyword_stuffing", "skills_only_match", "consulting_only_background"),
        evidence_requirements=(
            EvidenceRequirement("production_ml_systems", 0.45, ("career_history", "profile")),
            EvidenceRequirement("retrieval_search", 0.45, ("career_history", "profile")),
            EvidenceRequirement("ranking_evaluation", 0.35, ("career_history", "profile")),
        ),
        score_caps=(
            ScoreCap("no_production_evidence", 0.70, ("production_ml_systems",)),
            ScoreCap("skills_only_match", 0.60),
            ScoreCap("unrelated_title", 0.50),
            ScoreCap("unrelated_title_and_history", 0.50),
            ScoreCap("inactive_low_response", 0.75),
            ScoreCap("no_ranking_search_recommendation_evidence", 0.78),
            ScoreCap("generic_ml_without_retrieval_or_evaluation", 0.82),
            ScoreCap("pure_research_without_deployment", 0.68),
            ScoreCap("recent_llm_demo_only", 0.72),
            ScoreCap("consulting_only_background", 0.82),
        ),
        redrob_signal_policy={
            "hireability_multiplier_min": 0.85,
            "hireability_multiplier_max": 1.15,
            "trust_multiplier_min": 0.95,
            "trust_multiplier_max": 1.05,
            "market_validation_weight": 0.05,
        },
        reasoning_priorities=(
            "ranking_recommendation",
            "ranking_evaluation",
            "retrieval_search",
            "vector_hybrid_search",
            "production_ml_systems",
            "python_engineering",
        ),
        scoring_weights={
            "ranking_recommendation": 0.22,
            "ranking_evaluation": 0.18,
            "retrieval_search": 0.16,
            "vector_hybrid_search": 0.12,
            "production_ml_systems": 0.12,
            "python_engineering": 0.08,
            "ai_ml_engineering": 0.06,
            "product_management": 0.04,
            "open_source_validation": 0.02,
        },
        logistics_preferences={
            "preferred_locations": ["Pune", "Noida", "Delhi NCR", "Hyderabad", "Mumbai", "Bangalore"],
            "max_preferred_notice_days": 30,
            "max_acceptable_notice_days": 90,
            "hybrid_preferred": True,
        },
    )


def save_rolespec(path: Path, rolespec: RoleSpec) -> None:
    """Persist a RoleSpec as simple YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rolespec_to_mapping(rolespec)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_rolespec(path: Path | None) -> RoleSpec:
    """Load and validate a RoleSpec from YAML."""
    if path is None:
        return default_redrob_ai_engineer_rolespec()
    if not path.exists():
        raise FileNotFoundError(f"RoleSpec file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        data = json.loads(text)
        return rolespec_from_mapping(data)
    try:
        import yaml  # type: ignore
    except ImportError:
        data = _parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"RoleSpec must be a mapping: {path}")
    return rolespec_from_mapping(data)


def rolespec_from_mapping(data: dict[str, Any]) -> RoleSpec:
    """Validate untrusted RoleSpec data from YAML/JSON."""
    role_title = _required_str(data, "role_title")
    role_family = _required_str(data, "role_family")
    target_seniority = _required_str(data, "target_seniority")
    exp = data.get("experience_range")
    if not isinstance(exp, (list, tuple)) or len(exp) != 2:
        raise ValueError("experience_range must contain exactly [min, max]")
    experience_range = (float(exp[0]), float(exp[1]))
    if experience_range[0] < 0 or experience_range[1] < experience_range[0]:
        raise ValueError("experience_range must be non-negative and ordered")

    must_have = _validate_competencies(data.get("must_have_competencies", ()), "must_have_competencies")
    preferred = _validate_competencies(data.get("preferred_competencies", ()), "preferred_competencies")
    negative = _validate_negative_signals(data.get("negative_signals", ()))
    weights = _validate_weights(data.get("scoring_weights", {}), must_have + preferred)
    requirements = _validate_requirements(data.get("evidence_requirements", ()))
    caps = _validate_caps(data.get("score_caps", ()))
    redrob_policy = dict(data.get("redrob_signal_policy", {}))
    _validate_redrob_policy(redrob_policy)
    priorities = _validate_competencies(data.get("reasoning_priorities", must_have), "reasoning_priorities")
    return RoleSpec(
        role_title=role_title,
        role_family=role_family,
        target_seniority=target_seniority,
        experience_range=experience_range,
        must_have_competencies=must_have,
        preferred_competencies=preferred,
        negative_signals=negative,
        evidence_requirements=requirements,
        score_caps=caps,
        redrob_signal_policy=redrob_policy,
        reasoning_priorities=priorities,
        scoring_weights=weights,
        logistics_preferences=dict(data.get("logistics_preferences", {})),
    )


def rolespec_to_mapping(rolespec: RoleSpec) -> dict[str, Any]:
    return {
        "role_title": rolespec.role_title,
        "role_family": rolespec.role_family,
        "target_seniority": rolespec.target_seniority,
        "experience_range": list(rolespec.experience_range),
        "must_have_competencies": list(rolespec.must_have_competencies),
        "preferred_competencies": list(rolespec.preferred_competencies),
        "negative_signals": list(rolespec.negative_signals),
        "evidence_requirements": [
            {"name": item.name, "min_score": item.min_score, "required_sources": list(item.required_sources)}
            for item in rolespec.evidence_requirements
        ],
        "score_caps": [
            {"rule": item.rule, "max_score": item.max_score, "applies_to": list(item.applies_to)}
            for item in rolespec.score_caps
        ],
        "redrob_signal_policy": dict(rolespec.redrob_signal_policy),
        "reasoning_priorities": list(rolespec.reasoning_priorities),
        "scoring_weights": dict(rolespec.scoring_weights),
        "logistics_preferences": dict(rolespec.logistics_preferences),
    }


def _required_str(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"RoleSpec missing required field: {key}")
    return value


def _canonical_competency(name: Any) -> str:
    text = str(name).strip()
    return LEGACY_COMPETENCY_ALIASES.get(text, text)


def _validate_competencies(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    result = tuple(dict.fromkeys(_canonical_competency(item) for item in values))
    unsupported = sorted(set(result) - set(COMPETENCY_TAXONOMY))
    if unsupported:
        raise ValueError(f"Unsupported competencies in {field_name}: {unsupported}")
    return result


def _validate_negative_signals(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("negative_signals must be a list")
    result = tuple(dict.fromkeys(str(item).strip() for item in values))
    unsupported = sorted(set(result) - set(NEGATIVE_SIGNALS))
    if unsupported:
        raise ValueError(f"Unsupported negative_signals: {unsupported}")
    return result


def _validate_weights(values: Any, fallback: tuple[str, ...]) -> dict[str, float]:
    if not values:
        if not fallback:
            raise ValueError("RoleSpec requires scoring_weights or competencies")
        equal = 1.0 / len(fallback)
        return {name: equal for name in fallback}
    if not isinstance(values, dict):
        raise ValueError("scoring_weights must be a mapping")
    weights = {_canonical_competency(key): float(value) for key, value in values.items()}
    unsupported = sorted(set(weights) - set(COMPETENCY_TAXONOMY))
    if unsupported:
        raise ValueError(f"Unsupported competencies in scoring_weights: {unsupported}")
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        raise ValueError("scoring_weights must include positive values")
    return {key: max(0.0, value) / total for key, value in weights.items()}


def _validate_requirements(values: Any) -> tuple[EvidenceRequirement, ...]:
    if values is None:
        return ()
    items = values.values() if isinstance(values, dict) else values
    if not isinstance(items, (list, tuple, type(values.values()) if isinstance(values, dict) else list)):
        raise ValueError("evidence_requirements must be a list")
    result: list[EvidenceRequirement] = []
    for raw in list(items):
        if isinstance(raw, str):
            result.append(EvidenceRequirement(_canonical_competency(raw)))
            continue
        if not isinstance(raw, dict):
            raise ValueError("Each evidence requirement must be a mapping or string")
        name = _canonical_competency(raw.get("name"))
        if name not in COMPETENCY_TAXONOMY:
            raise ValueError(f"Unsupported evidence requirement competency: {name}")
        min_score = float(raw.get("min_score", 0.35))
        if not 0 <= min_score <= 1:
            raise ValueError("evidence requirement min_score must be 0..1")
        result.append(
            EvidenceRequirement(
                name=name,
                min_score=min_score,
                required_sources=tuple(str(item) for item in raw.get("required_sources", ())),
            )
        )
    return tuple(result)


def _validate_caps(values: Any) -> tuple[ScoreCap, ...]:
    if values is None:
        return ()
    items = values.values() if isinstance(values, dict) else values
    result: list[ScoreCap] = []
    for raw in list(items):
        if not isinstance(raw, dict):
            raise ValueError("Each score cap must be a mapping")
        rule = str(raw.get("rule", "")).strip()
        if rule not in SCORE_CAP_RULES:
            raise ValueError(f"Unsupported score cap rule: {rule}")
        max_score = float(raw.get("max_score", 1.0))
        if not 0 <= max_score <= 1:
            raise ValueError("score cap max_score must be 0..1")
        result.append(
            ScoreCap(
                rule=rule,
                max_score=max_score,
                applies_to=_validate_competencies(raw.get("applies_to", ()), "score_caps.applies_to"),
            )
        )
    return tuple(result)


def _validate_redrob_policy(policy: dict[str, Any]) -> None:
    for key in ("hireability_multiplier_min", "hireability_multiplier_max", "trust_multiplier_min", "trust_multiplier_max"):
        if key in policy:
            float(policy[key])
    if policy.get("hireability_multiplier_min", 0.85) < 0.5 or policy.get("hireability_multiplier_max", 1.15) > 1.5:
        raise ValueError("hireability multiplier bounds are outside safe range")
    if policy.get("trust_multiplier_min", 0.95) < 0.8 or policy.get("trust_multiplier_max", 1.05) > 1.2:
        raise ValueError("trust multiplier bounds are outside safe range")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            data[current_key] = _parse_scalar(value.strip()) if value.strip() else []
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            data.setdefault(current_key, []).append(_parse_scalar(stripped[2:].strip()))
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
            if not isinstance(data.get(current_key), dict):
                data[current_key] = {}
            data[current_key][key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def _dump_simple_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except ImportError:
        return "\n".join(_dump_lines(data)) + "\n"
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def _dump_lines(data: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dump_lines(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.extend(_dump_lines(item, indent + 4))
                else:
                    lines.append(f"{prefix}  - {item!r}")
        else:
            lines.append(f"{prefix}{key}: {value!r}")
    return lines
