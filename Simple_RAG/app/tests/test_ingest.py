"""
Tests for POST /ingest
"""

import io


HEADERS = {"X-API-Key": "dev-key-123"}


def test_ingest_valid_txt(client):
    """Valid .txt upload should return 200 with IngestResponse shape."""
    file_content = b"Hello, this is a podcast transcript."
    response = client.post(
        "/ingest",
        files={"file": ("episode1.txt", io.BytesIO(file_content), "text/plain")},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], int)
    assert "cached" in data
    assert "total_chunks" in data
    assert "total_sources" in data


def test_ingest_valid_srt(client):
    """Valid .srt upload should also return 200."""
    srt_content = b"1\n00:00:01,000 --> 00:00:03,000\nHello world.\n\n"
    response = client.post(
        "/ingest",
        files={"file": ("episode1.srt", io.BytesIO(srt_content), "text/plain")},
        headers=HEADERS,
    )
    assert response.status_code == 200


def test_ingest_invalid_extension(client):
    """Non-SRT/TXT file should return 400."""
    response = client.post(
        "/ingest",
        files={"file": ("document.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ingest_missing_api_key(client):
    """Missing X-API-Key header should return 401."""
    file_content = b"Some transcript text."
    response = client.post(
        "/ingest",
        files={"file": ("episode.txt", io.BytesIO(file_content), "text/plain")},
        # No headers — auth dependency should reject
    )
    assert response.status_code == 401
