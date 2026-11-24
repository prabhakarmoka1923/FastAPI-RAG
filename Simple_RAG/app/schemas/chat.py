from pydantic import BaseModel
from app.schemas.query import SourceChunk


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    messages: list[ChatMessage] = []
    provider: str = "openai"
    llm_api_key: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    total_hits: int
