"""FastAPI routes for TalentRank Intelligence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.schemas import JobCreateResponse, JobResultsResponse, JobStatusResponse
from backend.jobs.job_store import job_store
from backend.services.file_service import (
    ALLOWED_CANDIDATE_SUFFIXES,
    ALLOWED_JD_SUFFIXES,
    ALLOWED_ROLESPEC_SUFFIXES,
    ensure_job_dirs,
    prepare_candidates_file,
    save_upload,
)
from backend.services.ranking_service import process_web_ranking_job


router = APIRouter(prefix="/api")
executor = ThreadPoolExecutor(max_workers=2)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/jobs", response_model=JobCreateResponse)
def create_job(
    background_tasks: BackgroundTasks,
    candidates: UploadFile = File(...),
    job_description: UploadFile | None = File(None),
    rolespec: UploadFile | None = File(None),
    rolespec_mode: str = Form("auto_llm"),
    shortlist_size: int = Form(100),
    debug: bool = Form(True),
) -> JobCreateResponse:
    try:
        if shortlist_size <= 0 or shortlist_size > 100:
            raise ValueError("shortlist_size must be between 1 and 100")
        job = job_store.create()
        tmp_dir, export_dir = ensure_job_dirs(job.job_id)
        job_store.update(job.job_id, progress=5, stage="Uploaded files", message="Saving uploaded files")

        candidate_upload = save_upload(candidates, tmp_dir, ALLOWED_CANDIDATE_SUFFIXES, "Candidates file")
        candidate_path, candidate_count = prepare_candidates_file(candidate_upload, tmp_dir)

        jd_path: Path | None = None
        if job_description is not None and job_description.filename:
            jd_path = save_upload(job_description, tmp_dir, ALLOWED_JD_SUFFIXES, "Job description")

        rolespec_path: Path | None = None
        if rolespec is not None and rolespec.filename:
            rolespec_path = save_upload(rolespec, tmp_dir, ALLOWED_ROLESPEC_SUFFIXES, "RoleSpec")

        job_store.update(
            job.job_id,
            status="queued",
            progress=5,
            stage="Uploaded files",
            message="Job queued; RoleSpec will be selected or compiled in the background",
            metrics={"candidate_count": candidate_count, "shortlist_size": shortlist_size},
        )
        background_tasks.add_task(
            executor.submit,
            process_web_ranking_job,
            job.job_id,
            candidate_path,
            export_dir,
            shortlist_size,
            debug,
            candidate_count,
            rolespec_mode,
            jd_path,
            rolespec_path,
        )
        return JobCreateResponse(job_id=job.job_id, status="queued")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        message=job.message,
        metrics=job.metrics,
    )


@router.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
def get_results(job_id: str) -> JobResultsResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status}")
    return JobResultsResponse(
        top_candidates=job.top_candidates,
        metrics=job.metrics,
        rolespec=job.rolespec,
        diagnostics=job.diagnostics,
    )


@router.get("/jobs/{job_id}/download/submission")
def download_submission(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None or not job.submission_path:
        raise HTTPException(status_code=404, detail="Submission not found")
    return FileResponse(job.submission_path, media_type="text/csv", filename="submission.csv")


@router.get("/jobs/{job_id}/download/debug")
def download_debug(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None or not job.debug_path:
        raise HTTPException(status_code=404, detail="Debug export not found")
    return FileResponse(job.debug_path, media_type="text/csv", filename="talentrank_debug.csv")
