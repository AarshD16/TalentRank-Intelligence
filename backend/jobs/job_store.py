"""Thread-safe in-memory job store for web ranking runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: str = "queued"
    progress: int = 0
    stage: str = "Uploading files"
    message: str = "Job queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rolespec: dict[str, Any] | None = None
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    submission_path: str | None = None
    debug_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobStore:
    """Small in-memory store suitable for hackathon/demo deployment."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self) -> JobRecord:
        job = JobRecord(job_id=uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()
            return job


job_store = JobStore()
