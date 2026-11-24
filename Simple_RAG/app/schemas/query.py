from pydantic import BaseModel, ConfigDict


class SourceChunk(BaseModel):
    text: str
    source: str
    start_time: str
    end_time: str
    score: float


class AskRequest(BaseModel):
    query: str
    top_k: int = 5

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What did the guest say about AI?",
                "top_k": 5,
            }
        }
    )


class AskResponse(BaseModel):
    query: str
    sources: list[SourceChunk]
    context: str        # formatted context string ready for an LLM
    total_hits: int
