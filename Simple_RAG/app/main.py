"""
FastAPI application entry point.
Defines the lifespan (startup/shutdown), registers all routers.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make the project root importable so `rag_pipeline` can be found
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import at module level so tests can patch app.main.PodcastRAG.
# Models inside PodcastRAG are lazy-loaded (first use only) — importing the
# module does NOT trigger any model downloads or GPU allocation.
from rag_pipeline import PodcastRAG  # noqa: E402

from app.config import get_settings
from app.routers import health, ingest, query, chat

STATIC_DIR = Path(__file__).parent / "static"

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise PodcastRAG and restore state from disk cache.
    Shutdown: clean up the thread pool.
    """
    settings = get_settings()
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    rag = PodcastRAG(
        persist_dir=settings.RAG_PERSIST_DIR,
        chunk_size=settings.RAG_CHUNK_SIZE,
        overlap=settings.RAG_OVERLAP,
        top_k_retrieve=settings.RAG_TOP_K,
        top_n_rerank=settings.RAG_TOP_N,
    )

    # Restore previously ingested sources from disk on startup
    await loop.run_in_executor(executor, rag.load_state)

    app.state.rag = rag
    app.state.executor = executor

    yield  # server is live here

    executor.shutdown(wait=True)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Podcast RAG API",
    description=(
        "A FastAPI backend for the Podcast RAG system.\n\n"
        "Upload transcript files, query them with natural language, and get "
        "ranked source chunks ready to pass to an LLM.\n\n"
        "**Authentication:** click the 🔒 Authorize button and enter `dev-key-123`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(chat.router)
app.include_router(health.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")
