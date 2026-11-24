from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    chunks: int
    cached: bool
    total_chunks: int
    total_sources: int
