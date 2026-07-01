"""FastAPI application serving the TalentRank web UI and API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
STATIC_ASSETS = FRONTEND_DIST / "assets"

app = FastAPI(title="TalentRank Intelligence", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if STATIC_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_ASSETS), name="assets")


@app.get("/{full_path:path}", response_model=None)
def serve_frontend(full_path: str):
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "TalentRank API is running. Build the frontend to serve the web app."}
