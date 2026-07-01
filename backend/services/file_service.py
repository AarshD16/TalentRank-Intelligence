"""Upload persistence and lightweight input validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TMP_ROOT = BACKEND_ROOT / "tmp"
EXPORT_ROOT = BACKEND_ROOT / "static_exports"

ALLOWED_JD_SUFFIXES = {".docx", ".txt", ".md"}
ALLOWED_CANDIDATE_SUFFIXES = {".jsonl", ".json"}
ALLOWED_ROLESPEC_SUFFIXES = {".json", ".yaml", ".yml"}


def ensure_job_dirs(job_id: str) -> tuple[Path, Path]:
    tmp_dir = TMP_ROOT / job_id
    export_dir = EXPORT_ROOT / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir, export_dir


def save_upload(upload: UploadFile, target_dir: Path, allowed_suffixes: Iterable[str], label: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in set(allowed_suffixes):
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ValueError(f"{label} must be one of: {allowed}")
    target = target_dir / _safe_filename(upload.filename or f"upload{suffix}")
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return target


def prepare_candidates_file(path: Path, target_dir: Path) -> tuple[Path, int]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        count = _validate_jsonl(path)
        return path, count
    if suffix == ".json":
        converted = target_dir / "candidates.converted.jsonl"
        count = _convert_json_array_to_jsonl(path, converted)
        return converted, count
    raise ValueError("Candidates must be .jsonl or .json")


def _validate_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid candidates JSONL at line {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Candidate at line {line_number} must be an object")
            count += 1
    if count == 0:
        raise ValueError("Candidates file contains no candidate objects")
    return count


def _convert_json_array_to_jsonl(source: Path, target: Path) -> int:
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Candidate JSON must be an array of candidate objects")
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Candidate at index {index} must be an object")
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("Candidates file contains no candidate objects")
    return count


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name)
