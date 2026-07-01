"""RoleSpec selection and compilation for web jobs."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Callable

from scripts.generate_rolespec import _generate_with_ollama, _harden_rolespec, _read_jd
from src.role_parser import parse_rolespec_from_docx
from src.rolespec import default_redrob_ai_engineer_rolespec, load_rolespec, rolespec_from_mapping, rolespec_to_mapping, save_rolespec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REDR0B_ROLE = PROJECT_ROOT / "configs" / "rolespec_redrob_senior_ai_engineer.yaml"
ROLESPEC_CACHE_DIR = PROJECT_ROOT / "outputs" / "rolespec_cache"


def resolve_rolespec(
    mode: str,
    export_dir: Path,
    job_description_path: Path | None = None,
    uploaded_rolespec_path: Path | None = None,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> tuple[Path, dict]:
    out_path = export_dir / "rolespec.json"
    source: dict[str, Any] = {"mode": mode, "source": "unknown"}
    if mode == "saved_redrob":
        _progress(progress_callback, 12, "Building RoleSpec", "Using saved Redrob RoleSpec")
        rolespec = load_rolespec(DEFAULT_REDR0B_ROLE) if DEFAULT_REDR0B_ROLE.exists() else default_redrob_ai_engineer_rolespec()
        source = {"mode": mode, "source": "saved_redrob"}
    elif mode == "uploaded_rolespec":
        if uploaded_rolespec_path is None:
            raise ValueError("Upload a RoleSpec file when using uploaded_rolespec mode")
        _progress(progress_callback, 12, "Building RoleSpec", "Validating uploaded RoleSpec")
        rolespec = load_rolespec(uploaded_rolespec_path)
        source = {"mode": mode, "source": "uploaded_rolespec", "filename": uploaded_rolespec_path.name}
    elif mode == "compile_local":
        if job_description_path is None:
            raise ValueError("Upload a job description when using compile_local mode")
        _progress(progress_callback, 12, "Building RoleSpec", "Compiling deterministic local RoleSpec")
        rolespec = _compile_local(job_description_path)
        source = {"mode": mode, "source": "compile_local"}
    elif mode == "compile_llm":
        if job_description_path is None:
            raise ValueError("Upload a job description when using compile_llm mode")
        rolespec, source = _compile_llm(job_description_path, progress_callback=progress_callback)
    elif mode == "auto_llm":
        rolespec, source = _resolve_auto(job_description_path, uploaded_rolespec_path, progress_callback)
    else:
        raise ValueError(f"Unsupported RoleSpec mode: {mode}")
    rolespec = _harden_rolespec(rolespec)
    save_rolespec(out_path, rolespec)
    payload = rolespec_to_mapping(rolespec)
    payload["_talentrank"] = source
    return out_path, payload


def _compile_local(path: Path):
    if path.suffix.lower() == ".docx":
        return parse_rolespec_from_docx(path)
    text = _read_jd(path).lower()
    if "senior ai" in text or "ranking" in text or "retrieval" in text:
        return default_redrob_ai_engineer_rolespec()
    return default_redrob_ai_engineer_rolespec()


def _resolve_auto(
    job_description_path: Path | None,
    uploaded_rolespec_path: Path | None,
    progress_callback: Callable[[int, str, str], None] | None = None,
):
    if uploaded_rolespec_path is not None:
        _progress(progress_callback, 12, "Building RoleSpec", "Validating uploaded RoleSpec")
        return load_rolespec(uploaded_rolespec_path), {
            "mode": "auto_llm",
            "source": "uploaded_rolespec",
            "filename": uploaded_rolespec_path.name,
        }
    if job_description_path is None:
        raise ValueError("Upload a job description so TalentRank can reuse or compile the RoleSpec automatically")
    return _compile_llm(job_description_path, progress_callback=progress_callback, use_cache=True)


def _compile_llm(
    path: Path,
    progress_callback: Callable[[int, str, str], None] | None = None,
    use_cache: bool = True,
):
    jd_text = _read_jd(path)
    cache_key = _rolespec_cache_key(jd_text)
    cache_path = ROLESPEC_CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_path.exists():
        _progress(progress_callback, 12, "Building RoleSpec", "Found cached LLM RoleSpec for this job description")
        return load_rolespec(cache_path), {
            "mode": "auto_llm",
            "source": "llm_cache",
            "cache_key": cache_key,
        }

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL")
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not model:
        raise ValueError("No cached RoleSpec exists for this JD, and LLM compilation requires OLLAMA_MODEL to be configured on the backend")
    _progress(progress_callback, 12, "Building RoleSpec", f"Calling LLM to compile RoleSpec with model {model}")
    payload = _generate_with_ollama(
        jd_text=jd_text,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    rolespec = rolespec_from_mapping(payload)
    rolespec = _harden_rolespec(rolespec)
    if use_cache:
        ROLESPEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_rolespec(cache_path, rolespec)
    _progress(progress_callback, 15, "RoleSpec selected/generated", "LLM RoleSpec compiled and cached")
    return rolespec, {
        "mode": "auto_llm",
        "source": "llm_generated",
        "cache_key": cache_key,
        "model": model,
    }


def _rolespec_cache_key(jd_text: str) -> str:
    normalized = " ".join(jd_text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _progress(
    callback: Callable[[int, str, str], None] | None,
    progress: int,
    stage: str,
    message: str,
) -> None:
    if callback is not None:
        callback(progress, stage, message)
