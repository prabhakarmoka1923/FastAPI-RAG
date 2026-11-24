"""
Shared fixtures for all tests.
Real ML models are never loaded — PodcastRAG is replaced with a lightweight mock.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ─── Mock PodcastRAG ──────────────────────────────────────────────────────────

class MockRAG:
    """Minimal stand-in for PodcastRAG that avoids loading any ML model."""

    _ingested_sources: dict = {}

    def ingest(self, file_path: str) -> dict:
        name = Path(file_path).stem
        self._ingested_sources[name] = True
        return {
            "source": name,
            "chunks": 42,
            "cached": False,
            "total_chunks": 42,
            "total_sources": len(self._ingested_sources),
        }

    def retrieve(self, query: str):
        return [
            (
                "This is a sample transcript excerpt about AI.",
                {
                    "source": "test_podcast",
                    "chunk_id": 0,
                    "start_time": "00:01:00.000",
                    "end_time": "00:02:00.000",
                    "token_count": 10,
                },
                0.9321,
            )
        ]

    def format_context(self, hits) -> str:
        if not hits:
            return "No relevant excerpts found."
        parts = []
        for i, (text, meta, score) in enumerate(hits, 1):
            source = meta.get("source", "unknown")
            ts = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
            parts.append(f"[Excerpt {i} | Source: {source} | Timestamp: {ts}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def load_state(self) -> bool:
        return False


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    TestClient with dependency overrides:
    - get_rag  → MockRAG (no ML models)
    - get_executor → None (service layer not called in mock path)
    - verify_api_key → always passes
    """
    # Patch PodcastRAG before importing main so the lifespan never loads real models
    import unittest.mock as mock

    mock_rag_instance = MockRAG()

    with mock.patch("app.main.PodcastRAG", return_value=mock_rag_instance):
        from app.main import app
        from app.dependencies import get_rag, get_executor
        from app.services import rag_service

        # Override only the heavy dependencies — leave verify_api_key alone so
        # the 401 tests can exercise real auth.  Happy-path tests send the
        # correct "dev-key-123" header explicitly.
        app.dependency_overrides[get_rag] = lambda: mock_rag_instance
        app.dependency_overrides[get_executor] = lambda: None

        # Patch service layer to call mock directly (no executor needed)
        async def _mock_ingest(rag, executor, tmp_path):
            return rag.ingest(tmp_path)

        async def _mock_retrieve(rag, executor, query):
            return rag.retrieve(query)

        async def _mock_context(rag, executor, hits):
            return rag.format_context(hits)

        with (
            mock.patch.object(rag_service, "ingest_file", _mock_ingest),
            mock.patch.object(rag_service, "retrieve_chunks", _mock_retrieve),
            mock.patch.object(rag_service, "build_context", _mock_context),
        ):
            with TestClient(app) as c:
                yield c

        app.dependency_overrides.clear()
