"""
POST /ask — Query the RAG index and return ranked source chunks + formatted context.
"""

from fastapi import APIRouter, Depends

from app.dependencies import get_executor, get_rag, verify_api_key
from app.schemas.query import AskRequest, AskResponse, SourceChunk
from app.services import rag_service

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask(
    request: AskRequest,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):
    """
    Retrieve relevant transcript chunks for a natural-language query.

    - Runs hybrid retrieval (dense + BM25) followed by cross-encoder reranking.
    - Returns the top chunks and a pre-formatted context string ready to paste
      into an LLM prompt.
    - **This endpoint does retrieval only** — LLM generation is handled by the
      Streamlit front-end (`app.py`).
    """
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)

    if not hits:
        return AskResponse(
            query=request.query,
            sources=[],
            context="No relevant excerpts found.",
            total_hits=0,
        )

    # Trim to requested top_k
    hits = hits[: request.top_k]

    context = await rag_service.build_context(rag, executor, hits)

    sources = [
        SourceChunk(
            text=text,
            source=meta.get("source", "unknown"),
            start_time=meta.get("start_time", ""),
            end_time=meta.get("end_time", ""),
            score=round(score, 4),
        )
        for text, meta, score in hits
    ]

    return AskResponse(
        query=request.query,
        sources=sources,
        context=context,
        total_hits=len(sources),
    )
