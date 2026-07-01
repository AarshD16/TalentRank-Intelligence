"""Background job orchestration for ranking runs."""

from __future__ import annotations

import csv
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from rank import run_ranking
from src.rolespec import load_rolespec, rolespec_to_mapping

from backend.jobs.job_store import job_store
from backend.services.rolespec_service import resolve_rolespec


LOGGER = logging.getLogger(__name__)


def process_web_ranking_job(
    job_id: str,
    candidates_path: Path,
    export_dir: Path,
    shortlist_size: int,
    debug: bool,
    candidate_count: int,
    rolespec_mode: str,
    job_description_path: Path | None = None,
    uploaded_rolespec_path: Path | None = None,
) -> None:
    try:
        _update(
            job_id,
            status="running",
            progress=10,
            stage="Building RoleSpec",
            message="Checking for an existing RoleSpec policy",
        )
        role_path, role_payload = resolve_rolespec(
            mode=rolespec_mode,
            export_dir=export_dir,
            job_description_path=job_description_path,
            uploaded_rolespec_path=uploaded_rolespec_path,
            progress_callback=lambda progress, stage, message: _update(
                job_id,
                status="running",
                progress=progress,
                stage=stage,
                message=message,
            ),
        )
        _update(
            job_id,
            status="running",
            progress=15,
            stage="RoleSpec selected/generated",
            message=_rolespec_message(role_payload),
            rolespec=role_payload,
            metrics={"candidate_count": candidate_count, "shortlist_size": shortlist_size},
        )
        process_ranking_job(
            job_id=job_id,
            candidates_path=candidates_path,
            role_path=role_path,
            export_dir=export_dir,
            shortlist_size=shortlist_size,
            debug=debug,
            candidate_count=candidate_count,
            role_payload=role_payload,
        )
    except Exception as exc:  # noqa: BLE001 - user-facing job failure.
        LOGGER.exception("Ranking job %s failed while building RoleSpec", job_id)
        _update(
            job_id,
            status="failed",
            progress=100,
            stage="Failed",
            message=_friendly_error(exc),
        )


def process_ranking_job(
    job_id: str,
    candidates_path: Path,
    role_path: Path,
    export_dir: Path,
    shortlist_size: int,
    debug: bool,
    candidate_count: int,
    role_payload: dict[str, Any] | None = None,
) -> None:
    started = time.perf_counter()
    submission_path = export_dir / "submission.csv"
    debug_dir = export_dir / "debug"
    try:
        _update(job_id, status="running", progress=15, stage="Building RoleSpec", message="RoleSpec selected and validated")
        metadata = run_ranking(
            candidates_path=candidates_path,
            role_path=role_path,
            out_path=submission_path,
            debug=debug,
            shortlist_size=shortlist_size,
            debug_dir=debug_dir,
            external_validate=False,
            progress_callback=lambda progress, stage, message: _update(
                job_id,
                status="running",
                progress=progress,
                stage=stage,
                message=message,
            ),
        )
        _update(job_id, status="running", progress=90, stage="CSV/export built", message="Building dashboard payload")
        top_candidates = _read_submission(submission_path)
        debug_rows = _read_optional_csv(debug_dir / "top_200_debug.csv")
        component_rows = _read_optional_csv(debug_dir / "component_scores.csv")
        enriched = _enrich_candidates(top_candidates, debug_rows, component_rows)
        metrics = _metrics(enriched, candidate_count, time.perf_counter() - started, metadata)
        role = role_payload or rolespec_to_mapping(load_rolespec(role_path))
        diagnostics = _diagnostics(enriched)
        debug_path = debug_dir / "top_200_debug.csv"
        _update(
            job_id,
            status="completed",
            progress=100,
            stage="Completed",
            message="Ranking completed successfully",
            metrics=metrics,
            diagnostics=diagnostics,
            rolespec=role,
            top_candidates=enriched,
            submission_path=str(submission_path),
            debug_path=str(debug_path) if debug_path.exists() else None,
        )
    except Exception as exc:  # noqa: BLE001 - user-facing job failure.
        LOGGER.exception("Ranking job %s failed", job_id)
        _update(
            job_id,
            status="failed",
            progress=100,
            stage="Failed",
            message=_friendly_error(exc),
        )


def _update(job_id: str, **changes: Any) -> None:
    job_store.update(job_id, **changes)


def _rolespec_message(role_payload: dict[str, Any]) -> str:
    source = role_payload.get("_talentrank", {}) if isinstance(role_payload, dict) else {}
    source_name = source.get("source")
    if source_name == "llm_cache":
        return "Reused cached LLM RoleSpec for this job description"
    if source_name == "llm_generated":
        return "Compiled and cached a new LLM RoleSpec for this job description"
    if source_name == "uploaded_rolespec":
        return "Using uploaded RoleSpec"
    if source_name == "saved_redrob":
        return "Using saved Redrob RoleSpec"
    return "RoleSpec selected and validated"


def _read_submission(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_optional_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _enrich_candidates(
    submission_rows: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    debug_by_id = {row.get("candidate_id"): row for row in debug_rows}
    components_by_id = {row.get("candidate_id"): row for row in component_rows}
    result: list[dict[str, Any]] = []
    for row in submission_rows:
        candidate_id = row.get("candidate_id", "")
        debug = debug_by_id.get(candidate_id, {})
        components = components_by_id.get(candidate_id, {})
        result.append(
            {
                "candidate_id": candidate_id,
                "rank": _int(row.get("rank")),
                "score": _float(row.get("score")),
                "reasoning": row.get("reasoning", ""),
                "title": debug.get("current_title", ""),
                "years": debug.get("years_of_experience", ""),
                "recruiter_response_rate": _float(debug.get("recruiter_response_rate")),
                "notice_period_days": _float(debug.get("notice_period_days")),
                "score_flags": debug.get("score_flags", ""),
                "missing_must_have": debug.get("missing_must_have", ""),
                "top_competencies": debug.get("top_competencies", ""),
                "components": {
                    key: _float(value)
                    for key, value in components.items()
                    if key not in {"candidate_id", "debug_rank", "score"} and value not in {"", None}
                },
            }
        )
    return result


def _metrics(rows: list[dict[str, Any]], candidate_count: int, elapsed: float, metadata: dict[str, Any]) -> dict[str, Any]:
    scores = [_float(row.get("score")) for row in rows]
    response_rates = [_float(row.get("recruiter_response_rate")) for row in rows]
    response_rates = [rate for rate in response_rates if rate > 0]
    notices = [_float(row.get("notice_period_days")) for row in rows]
    notices = [notice for notice in notices if notice > 0]
    return {
        "candidates_processed": metadata.get("processed", candidate_count),
        "shortlist_size": len(rows),
        "top_score": max(scores) if scores else 0.0,
        "runtime_seconds": round(elapsed, 3),
        "complete_match_count": sum("complete_match_bundle" in str(row.get("score_flags", "")) for row in rows),
        "evaluation_evidence_count": sum("evaluation" in str(row.get("reasoning", "")).lower() or "a/b" in str(row.get("reasoning", "")).lower() for row in rows),
        "vector_evidence_count": sum(any(term in str(row.get("reasoning", "")).lower() for term in ("bm25", "faiss", "dense", "vector", "semantic search")) for row in rows),
        "average_response_rate": round(sum(response_rates) / len(response_rates), 3) if response_rates else 0.0,
        "median_notice_period": statistics.median(notices) if notices else 0.0,
    }


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "skills_only_count": sum("skills_only" in str(row.get("score_flags", "")) for row in rows),
        "unrelated_count": sum("unrelated" in str(row.get("score_flags", "")) for row in rows),
        "low_activity_count": sum("low_activity" in str(row.get("score_flags", "")) for row in rows),
        "reason_phrase_frequency": _reason_phrase_frequency(rows),
    }


def _reason_phrase_frequency(rows: list[dict[str, Any]]) -> dict[str, int]:
    phrases = ("A/B testing", "NDCG", "MRR", "BM25", "FAISS", "semantic search", "human relevance", "50M", "30M")
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reasoning", "")).lower()
        for phrase in phrases:
            if phrase.lower() in reason:
                counts[phrase] = counts.get(phrase, 0) + 1
    return counts


def _friendly_error(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return "The ranking job failed. Check backend logs for details."
    return text.splitlines()[0][:500]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
