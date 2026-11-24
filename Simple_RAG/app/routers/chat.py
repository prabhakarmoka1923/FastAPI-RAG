"""
POST /chat — Full RAG + LLM pipeline in one request.
Retrieves relevant chunks then calls OpenAI / Groq to generate an answer.
"""

import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import get_executor, get_rag, verify_api_key
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.query import SourceChunk
from app.services import rag_service

router = APIRouter()


def _call_llm_sync(messages, context: str, provider: str, api_key: str, active_sources: list) -> str:
    if not context or context.strip() == "No relevant excerpts found.":
        return "I couldn't find relevant information in the transcripts to answer this question."

    source_list = ", ".join(active_sources) if active_sources else "unknown"
    system_prompt = f"""You are a podcast transcript assistant. Answer questions using the transcript excerpts provided below, drawn from {len(active_sources)} transcript(s): {source_list}.

STRICT RULES:
1. ONLY use information explicitly present in the TRANSCRIPT EXCERPTS below.
2. If the excerpts do not contain a clear answer, say: "This topic is not covered in the available transcripts."
3. When answering, cite which source and approximate timestamp (e.g., "In [source_name] around 12:34, ...").
4. If multiple transcripts discuss the same topic, compare or synthesize their perspectives.
5. Do NOT guess, infer, or use outside knowledge under any circumstances.
6. Keep answers concise and grounded.

TRANSCRIPT EXCERPTS:
{context}"""

    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages += [{"role": m.role, "content": m.content} for m in messages]

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):
    """
    Full RAG + LLM pipeline.

    - Retrieves relevant transcript chunks for the query.
    - Passes them as context to the chosen LLM (OpenAI gpt-4o-mini or Groq llama-3.3-70b).
    - Returns the generated answer alongside the source chunks used.
    """
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)
    hits = hits[: request.top_k]

    if not hits:
        return ChatResponse(
            answer="I couldn't find relevant information in the transcripts to answer this question.",
            sources=[],
            total_hits=0,
        )

    context = await rag_service.build_context(rag, executor, hits)
    active_sources = list(rag._ingested_sources.keys()) if hasattr(rag, "_ingested_sources") else []

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        executor,
        _call_llm_sync,
        request.messages,
        context,
        request.provider,
        request.llm_api_key,
        active_sources,
    )

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

    return ChatResponse(answer=answer, sources=sources, total_hits=len(sources))
