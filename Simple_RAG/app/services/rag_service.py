"""
Async wrappers around the synchronous (CPU-bound) PodcastRAG methods.
All blocking calls must go through run_in_executor — never call rag.* directly
from a route handler.
"""

import asyncio


async def ingest_file(rag, executor, tmp_path: str) -> dict:
    """
    Run rag.ingest() in the thread pool.
    Returns the stats dict produced by PodcastRAG.ingest().
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.ingest, tmp_path)


async def retrieve_chunks(rag, executor, query: str) -> list:
    """
    Run rag.retrieve() in the thread pool.
    Returns List[Tuple[str, dict, float]].
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.retrieve, query)


async def build_context(rag, executor, hits: list) -> str:
    """
    Run rag.format_context() in the thread pool.
    Returns the formatted context string.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.format_context, hits)
