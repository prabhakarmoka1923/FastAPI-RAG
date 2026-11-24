"""
FastAPI dependency functions shared across routers.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from app.config import Settings, get_settings


def get_rag(request: Request):
    """Return the shared PodcastRAG instance from app state."""
    return request.app.state.rag


def get_executor(request: Request):
    """Return the shared ThreadPoolExecutor from app state."""
    return request.app.state.executor


def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for auth"),
    settings: Settings = Depends(get_settings),
):
    """
    Validate the X-API-Key header.
    Returns 401 (Unauthorized) for missing or wrong key — not 403.
    Using Optional so a missing header reaches this function (avoids 422).
    """
    if x_api_key is None or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
