"""Generate a validated RoleSpec scoring plan from a job description.

The LLM is used only as a config planner. It must return JSON data matching the
RoleSpec schema. It never ranks candidates and never generates executable code.
The saved RoleSpec is then used by the offline deterministic ranker.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competency_taxonomy import COMPETENCY_PATTERNS
from src.rolespec import (
    COMPETENCY_TAXONOMY,
    RoleSpec,
    EvidenceRequirement,
    ScoreCap,
    NEGATIVE_SIGNALS,
    REDROB_SIGNAL_NAMES,
    SCORE_CAP_RULES,
    rolespec_from_mapping,
    save_rolespec,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    _load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate a validated RoleSpec YAML/JSON config from a JD.")
    parser.add_argument("--jd", required=True, type=Path, help="Path to .docx or text job description.")
    parser.add_argument("--out", required=True, type=Path, help="Output RoleSpec config path.")
    parser.add_argument("--provider", choices=("ollama", "draft-json"), default="ollama")
    parser.add_argument("--draft-json", type=Path, help="Use a pre-generated LLM JSON draft instead of calling Ollama.")
    parser.add_argument("--ollama-base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    parser.add_argument(
        "--ollama-api-key",
        default=os.environ.get("OLLAMA_API_KEY", ""),
        help="Bearer token for hosted/Ollama-compatible cloud endpoints. Local Ollama usually leaves this empty.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("LOG_FILE", ""),
        help="Optional path to write a background log file in addition to console logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level, args.log_file)
    LOGGER.info("Starting RoleSpec generation")
    LOGGER.info("Reading job description from %s", args.jd)
    jd_text = _read_jd(args.jd)
    LOGGER.info("Loaded job description (%d characters)", len(jd_text))
    if args.provider == "draft-json":
        if args.draft_json is None:
            raise ValueError("--draft-json is required when --provider draft-json is used")
        LOGGER.info("Using draft JSON provider; no LLM call will be made")
        payload = json.loads(args.draft_json.read_text(encoding="utf-8"))
    else:
        LOGGER.info(
            "Calling LLM provider=ollama endpoint=%s model=%s auth=%s",
            _safe_endpoint(args.ollama_base_url),
            args.model,
            "enabled" if args.ollama_api_key else "not set",
        )
        payload = _generate_with_ollama(
            jd_text=jd_text,
            base_url=args.ollama_base_url,
            model=args.model,
            api_key=args.ollama_api_key,
        )
    LOGGER.info("Validating generated RoleSpec against schema")
    rolespec = rolespec_from_mapping(payload)
    rolespec = _harden_rolespec(rolespec)
    LOGGER.info(
        "Validated RoleSpec role_title=%s role_family=%s must_have=%s",
        rolespec.role_title,
        rolespec.role_family,
        ",".join(rolespec.must_have_competencies),
    )
    save_rolespec(args.out, rolespec)
    LOGGER.info("Wrote validated RoleSpec to %s", args.out)
    _emit_rolespec_diagnostics(rolespec)
    print(f"Wrote validated RoleSpec to {args.out}")
    return 0


def _configure_logging(level: str, log_file: str = "") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _safe_endpoint(base_url: str) -> str:
    return base_url.rstrip("/")


def _read_jd(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        texts = [node.text or "" for node in root.iter() if node.tag.endswith("}t")]
        return " ".join(texts)
    return path.read_text(encoding="utf-8")


def _generate_with_ollama(jd_text: str, base_url: str, model: str, api_key: str = "") -> dict:
    prompt = _build_prompt(jd_text)
    LOGGER.info("Built RoleSpec compiler prompt (%d characters)", len(prompt))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    request = urllib.request.Request(
        url=endpoint,
        data=json.dumps(
            {
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a scoring-plan compiler. Return only a JSON object. "
                            "Do not rank candidates. Do not write code. Do not invent unsupported competencies."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        started = time.perf_counter()
        LOGGER.info("Sending RoleSpec compiler request to Ollama")
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
        LOGGER.info("Received Ollama response in %.2fs", time.perf_counter() - started)
    except urllib.error.URLError as exc:
        LOGGER.error("Ollama request failed for endpoint %s: %s", _safe_endpoint(endpoint), exc)
        raise RuntimeError(
            "Could not reach the Ollama endpoint. For local Ollama, start the local server. "
            "For Ollama Cloud or another hosted Ollama-compatible endpoint, set "
            "OLLAMA_BASE_URL, OLLAMA_MODEL, and OLLAMA_API_KEY."
        ) from exc
    content = raw.get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama returned an empty RoleSpec draft")
    LOGGER.info("Extracting JSON RoleSpec draft from LLM response (%d characters)", len(content))
    return _extract_json_object(content)


def _build_prompt(jd_text: str) -> str:
    schema = {
        "role_title": "string",
        "role_family": "string",
        "target_seniority": "junior|mid|senior|lead_plus",
        "experience_range": [0, 50],
        "must_have_competencies": list(COMPETENCY_TAXONOMY),
        "preferred_competencies": list(COMPETENCY_TAXONOMY),
        "negative_signals": list(NEGATIVE_SIGNALS),
        "evidence_requirements": [
            {"name": "one competency", "min_score": 0.35, "required_sources": ["career_history", "profile"]}
        ],
        "score_caps": [
            {"rule": "skills_only_match", "max_score": 0.60, "applies_to": []},
            {"rule": "no_production_evidence", "max_score": 0.70, "applies_to": ["production_ml_systems"]},
        ],
        "redrob_signal_policy": {
            "hireability_multiplier_min": 0.85,
            "hireability_multiplier_max": 1.15,
            "trust_multiplier_min": 0.95,
            "trust_multiplier_max": 1.05,
            "market_validation_weight": 0.05,
        },
        "reasoning_priorities": list(COMPETENCY_TAXONOMY),
        "scoring_weights": {"competency_name": 0.1},
        "logistics_preferences": {},
    }
    taxonomy_summary = {
        name: {
            "title_patterns": patterns["title_patterns"],
            "career_history_evidence_patterns": patterns["career_history_evidence_patterns"],
            "negative_confusing_patterns": patterns["negative_confusing_patterns"],
        }
        for name, patterns in COMPETENCY_PATTERNS.items()
    }
    return (
        "Create a curated RoleSpec JSON scoring plan for the job description below.\n"
        "The RoleSpec is data only. It will be validated and then used by an offline deterministic ranker.\n\n"
        "Output contract:\n"
        "- Return one JSON object only, with no markdown, notes, comments, or self-check text.\n"
        "- Use only the allowed competencies, negative signals, and score cap rules listed below.\n"
        "- Do not include candidate IDs. Do not create executable code. Do not rank candidates.\n\n"
        "Analysis instructions:\n"
        "- Analyze what the JD means, not just exact keywords.\n"
        "- Identify must-have competencies, preferred competencies, and negative/confusing signals.\n"
        "- Distinguish hard differentiators from tie-breakers. Hard differentiators get most capability weight.\n"
        "- Put exact numeric scoring_weights on competencies only; weights should sum to exactly 1.0 before validation.\n"
        "- At least 65% of capability weight must go to must-have technical evidence, not title/company fit.\n"
        "- Require career_history evidence for critical competencies whenever the JD asks for production ownership.\n"
        "- Score caps must protect against skills-only, unrelated-title/history, no-production, and missing critical evidence profiles.\n"
        "- Redrob behavioral and hireability signals are bounded modifiers only; never include them in capability competencies.\n"
        "- Generic title fit, company fit, market validation, and availability are tie-breakers, not primary capability evidence.\n"
        "- reasoning_priorities must order the competencies whose exact snippets should be surfaced in candidate reasoning.\n\n"
        "Senior technical role calibration:\n"
        "- For senior AI/search/recommendation roles, prefer complete career evidence over broad generic AI wording.\n"
        "- Strong complete matches combine: relevant title family, 5-9 years experience, production ranking/search/recommendation systems, evaluation evidence, retrieval/vector infrastructure, and scale or measurable product impact.\n"
        "- Treat 4.5-9.5 years as a recall-safe senior band when evidence is exceptional; 4.0-4.5 years can pass only with unusually strong ranking/evaluation evidence.\n"
        "- Relevant titles include Senior AI Engineer, Lead AI Engineer, Recommendation Systems Engineer, Search Engineer, Applied ML Engineer, Machine Learning Engineer, NLP Engineer, and Senior Data Scientist only with strong retrieval/ranking evidence.\n"
        "- Senior Applied Scientist is relevant when the career history has production ranking/search/retrieval and evaluation evidence.\n"
        "- Do not let generic AI Engineer, Data Scientist, LLM demo, or skill-list-only profiles outrank search/recommendation/NLP profiles with ranking-evaluation evidence.\n\n"
        "Redrob Senior AI Engineer calibration, when the JD matches that role:\n"
        "- scoring_weights should approximate these ranges:\n"
        "  ranking_recommendation 0.20-0.24; ranking_evaluation 0.16-0.20; retrieval_search 0.14-0.18; vector_hybrid_search 0.10-0.14; production_ml_systems 0.10-0.14; python_engineering 0.06-0.09; ai_ml_engineering 0.04-0.07; product_management as product/company context 0.04-0.07; open_source_validation 0.02-0.05.\n"
        "- evidence_requirements should include:\n"
        "  ranking_recommendation min_score >= 0.45 with career_history; retrieval_search min_score >= 0.40 with career_history; ranking_evaluation min_score >= 0.35 with career_history; vector_hybrid_search min_score >= 0.35 with career_history or profile; production_ml_systems min_score >= 0.40 with career_history.\n"
        "- score_caps should include supported equivalents for: skills_only_match max 0.60; unrelated_title_and_history max 0.50; no_production_evidence max 0.70; no_ranking_search_recommendation_evidence max 0.78; generic_ml_without_retrieval_or_evaluation max 0.82; pure_research_without_deployment max 0.68; recent_llm_demo_only max 0.72; inactive_low_response max 0.75.\n"
        "- reasoning should prioritize exact evidence such as offline-online correlation, A/B testing, NDCG/MRR/MAP/recall@K, precision@K, relevance labeling, human relevance judgments, BM25 + dense retrieval, FAISS HNSW, semantic search at document/item scale, production recommendation/search/ranking systems, 30M+ candidate corpus, 50M+ queries/month, 8K QPS, sub-200ms p95 latency, time-to-shortlist improvement, recruiter engagement lift, CTR/revenue/conversion impact, and millions of users/items/queries.\n"
        "- Do not let notice period alone cap complete technical matches; notice period should only affect bounded hireability.\n\n"
        "Internal self-check before returning JSON:\n"
        "- Capability weights sum to 1.0.\n"
        "- Must-have technical competencies receive at least 65% total weight.\n"
        "- Redrob/hireability/trust/market signals are not in scoring_weights.\n"
        "- Generic title/company context has lower weight than ranking/evaluation/retrieval evidence.\n"
        "- score_caps include skills_only_match and a production-evidence cap.\n"
        "- evidence_requirements include career_history for critical competencies.\n"
        "- Final response is still JSON only.\n\n"
        f"Allowed schema example:\n{json.dumps(schema, indent=2)}\n\n"
        f"Allowed score_cap rules: {json.dumps(list(SCORE_CAP_RULES))}\n"
        f"Allowed Redrob signals: {json.dumps(list(REDROB_SIGNAL_NAMES))}\n"
        f"Competency taxonomy summary:\n{json.dumps(taxonomy_summary, indent=2)}\n\n"
        f"Job description:\n{jd_text[:12000]}\n"
    )


def _emit_rolespec_diagnostics(rolespec: RoleSpec) -> None:
    LOGGER.info("Generated scoring_weights descending:")
    for name, weight in sorted(rolespec.scoring_weights.items(), key=lambda item: (-item[1], item[0])):
        LOGGER.info("  %s: %.4f", name, weight)

    LOGGER.info("Generated evidence_requirements:")
    for item in rolespec.evidence_requirements:
        LOGGER.info("  %s min_score=%.2f required_sources=%s", item.name, item.min_score, list(item.required_sources))

    LOGGER.info("Generated score_caps:")
    for item in rolespec.score_caps:
        LOGGER.info("  %s max_score=%.2f applies_to=%s", item.rule, item.max_score, list(item.applies_to))

    policy = rolespec.redrob_signal_policy
    LOGGER.info(
        "Redrob multiplier bounds: hireability %.2f..%.2f trust %.2f..%.2f",
        float(policy.get("hireability_multiplier_min", 0.85)),
        float(policy.get("hireability_multiplier_max", 1.15)),
        float(policy.get("trust_multiplier_min", 0.95)),
        float(policy.get("trust_multiplier_max", 1.05)),
    )
    LOGGER.info("Generated reasoning_priorities: %s", list(rolespec.reasoning_priorities))

    weights = rolespec.scoring_weights
    if weights.get("ranking_evaluation", 0.0) < 0.14:
        LOGGER.warning("RoleSpec warning: ranking_evaluation weight is below 0.14")
    critical_sum = (
        weights.get("ranking_recommendation", 0.0)
        + weights.get("retrieval_search", 0.0)
        + weights.get("ranking_evaluation", 0.0)
    )
    if critical_sum < 0.50:
        LOGGER.warning("RoleSpec warning: ranking/retrieval/evaluation combined weight is %.3f below 0.50", critical_sum)
    if not any(cap.rule == "skills_only_match" for cap in rolespec.score_caps):
        LOGGER.warning("RoleSpec warning: no score cap protects skills-only profiles")
    if float(policy.get("hireability_multiplier_min", 0.85)) < 0.85 or float(policy.get("hireability_multiplier_max", 1.15)) > 1.15:
        LOGGER.warning("RoleSpec warning: Redrob hireability multiplier exceeds 0.85-1.15")

    critical_max = max(
        weights.get("ranking_recommendation", 0.0),
        weights.get("ranking_evaluation", 0.0),
        weights.get("retrieval_search", 0.0),
        weights.get("vector_hybrid_search", 0.0),
        weights.get("production_ml_systems", 0.0),
    )
    generic_max = max(weights.get("ai_ml_engineering", 0.0), weights.get("product_management", 0.0))
    if generic_max > critical_max:
        LOGGER.warning("RoleSpec warning: generic title/company weights exceed critical technical competency weights")


def _harden_rolespec(rolespec: RoleSpec) -> RoleSpec:
    role_text = f"{rolespec.role_title} {rolespec.role_family}".lower()
    if "senior" not in role_text or "ai" not in role_text:
        return rolespec

    requirements = {item.name: item for item in rolespec.evidence_requirements}
    required = {
        "ranking_recommendation": EvidenceRequirement("ranking_recommendation", 0.45, ("career_history",)),
        "retrieval_search": EvidenceRequirement("retrieval_search", 0.40, ("career_history",)),
        "ranking_evaluation": EvidenceRequirement("ranking_evaluation", 0.35, ("career_history",)),
        "vector_hybrid_search": EvidenceRequirement("vector_hybrid_search", 0.35, ("career_history", "profile")),
        "production_ml_systems": EvidenceRequirement("production_ml_systems", 0.40, ("career_history",)),
    }
    for name, item in required.items():
        current = requirements.get(name)
        if current is None or current.min_score < item.min_score or not set(item.required_sources).issubset(set(current.required_sources)):
            requirements[name] = item

    caps = {item.rule: item for item in rolespec.score_caps}
    required_caps = {
        "skills_only_match": ScoreCap("skills_only_match", 0.60),
        "unrelated_title_and_history": ScoreCap("unrelated_title_and_history", 0.50),
        "no_production_evidence": ScoreCap("no_production_evidence", 0.70, ("production_ml_systems",)),
        "no_ranking_search_recommendation_evidence": ScoreCap("no_ranking_search_recommendation_evidence", 0.78),
        "generic_ml_without_retrieval_or_evaluation": ScoreCap("generic_ml_without_retrieval_or_evaluation", 0.82),
        "pure_research_without_deployment": ScoreCap("pure_research_without_deployment", 0.68),
        "recent_llm_demo_only": ScoreCap("recent_llm_demo_only", 0.72),
        "inactive_low_response": ScoreCap("inactive_low_response", 0.75),
    }
    for rule, item in required_caps.items():
        current = caps.get(rule)
        if current is None or current.max_score > item.max_score:
            caps[rule] = item

    return replace(
        rolespec,
        evidence_requirements=tuple(requirements.values()),
        score_caps=tuple(caps.values()),
    )


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("RoleSpec draft must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
