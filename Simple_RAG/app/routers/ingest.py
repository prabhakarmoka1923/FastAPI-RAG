"""
POST /ingest — Upload a transcript file (.srt or .txt) to add it to the RAG index.
"""

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.dependencies import get_executor, get_rag, verify_api_key
from app.schemas.ingest import IngestResponse
from app.services import rag_service

router = APIRouter()

_ALLOWED_EXTENSIONS = {".srt", ".txt"}


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    file: UploadFile,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):
    """
    Ingest a podcast transcript into the RAG index.

    - Accepts `.srt` (SubRip subtitle) or `.txt` files.
    - If the transcript was already ingested in a previous session it is loaded
      from the on-disk cache (no re-embedding).
    - Returns chunk counts and whether the source was served from cache.
    """
    # Validate file extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .srt and .txt are accepted.",
        )

    # Write upload to a temp file so rag.ingest() can read it by path
    tmp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        stats = await rag_service.ingest_file(rag, executor, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return IngestResponse(
        source=stats["source"],
        chunks=stats["chunks"],
        cached=stats.get("cached", False),
        total_chunks=stats["total_chunks"],
        total_sources=stats["total_sources"],
    )
