"""
GET /health — Liveness check. No auth required.
"""

from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter()


@router.get("/health", tags=["System"])
async def health(request: Request):
    """
    Liveness / readiness check.

    - Always callable — no API key required.
    - `sources_loaded` tells you how many transcripts are currently in memory
      so you can verify state without ingesting anything.
    """
    settings = get_settings()
    rag = request.app.state.rag
    return {
        "status": "ok",
        "env": settings.ENV,
        "sources_loaded": len(rag._ingested_sources),
    }
