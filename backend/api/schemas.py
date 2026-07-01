"""Pydantic response schemas for the TalentRank web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "completed", "failed"]


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class JobResultsResponse(BaseModel):
    top_candidates: list[dict[str, Any]]
    metrics: dict[str, Any]
    rolespec: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
