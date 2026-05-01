"""Pydantic schemas used across the triage pipeline."""

from typing import Optional

from pydantic import BaseModel, Field


class TriageOutput(BaseModel):
    """Normalized output schema for a single ticket."""

    response: str = Field(default="")
    product_area: str = Field(default="")
    status: str = Field(default="escalated")
    request_type: str = Field(default="invalid")
    justification: str = Field(default="")


class Chunk(BaseModel):
    """Chunk metadata and text used for retrieval."""

    chunk_id: str
    domain: str
    product_area: str
    title: str
    source_path: str
    source_url: Optional[str] = None
    content: str


class RetrievedChunk(BaseModel):
    """Retrieved chunk and similarity score."""

    chunk: Chunk
    score: float


class AgentLLMOutput(BaseModel):
    """Structured response expected from the LLM."""

    status: str = Field(default="replied")
    product_area: str = Field(default="general")
    response: str = Field(default="")
    justification: str = Field(default="")
    request_type: str = Field(default="product_issue")
