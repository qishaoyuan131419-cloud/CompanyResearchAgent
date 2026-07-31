from datetime import datetime

from pydantic import AnyHttpUrl, Field

from app.core.enums import SourceType
from app.schemas.base import StrictModel


class SearchResult(StrictModel):
    title: str = Field(max_length=1_000)
    url: AnyHttpUrl
    text: str = ""
    published_at: datetime | None = None
    author: str | None = None
    score: float | None = None
    source_type: SourceType = SourceType.OTHER


class QueryExecution(StrictModel):
    query_id: str
    query: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)
    cache_hit: bool = False
    results: list[SearchResult] = Field(default_factory=list)
    error: str | None = None


class SearchStatistics(StrictModel):
    query_count: int = Field(ge=0)
    successful_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    total_results: int = Field(ge=0)
    deduplicated_results: int = Field(default=0, ge=0)
    total_duration_ms: float = Field(ge=0.0)
    retries: int = Field(default=0, ge=0)


class SearchBatch(StrictModel):
    executions: list[QueryExecution] = Field(default_factory=list)
    statistics: SearchStatistics


class SourceDocument(StrictModel):
    source_id: str
    title: str = Field(max_length=1_000)
    url: AnyHttpUrl
    publisher: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    source_type: SourceType
    summary: str | None = None
    # Raw provider text is retained internally for exact-quote checks, but is
    # never serialized through the public API or trace.
    content: str = Field(default="", exclude=True, repr=False)
    content_hash: str
    query_ids: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
