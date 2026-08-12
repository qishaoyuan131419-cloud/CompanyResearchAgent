from pydantic import Field, field_validator

from app.schemas.base import StrictModel


class ResearchTopic(StrictModel):
    topic: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=1_000)
    expected_output: str = Field(min_length=1, max_length=1_000)
    estimated_value: float = Field(ge=0.0, le=1.0)


class ResearchPlan(StrictModel):
    topics: list[ResearchTopic] = Field(min_length=1)


class SearchQuery(StrictModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=1_000)
    topic: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=1, le=5)
    expected_evidence: str = Field(min_length=1, max_length=1_000)
    language: str = Field(default="en", min_length=2, max_length=20)
    source_preference: list[str] = Field(default_factory=list)
    addresses_missing_items: list[str] = Field(default_factory=list)
    intent: str | None = Field(default=None, min_length=1, max_length=200)
    coverage_dimensions: list[str] = Field(default_factory=list)
    round: int = Field(default=0, ge=0)

    @field_validator("query")
    @classmethod
    def reject_urls(cls, value: str) -> str:
        lowered = value.casefold()
        if "http://" in lowered or "https://" in lowered:
            raise ValueError("search queries must not contain URLs")
        return value


class QueryPlan(StrictModel):
    queries: list[SearchQuery] = Field(default_factory=list)


class ResearchPlanAndQueries(StrictModel):
    topics: list[ResearchTopic] = Field(min_length=1)
    queries: list[SearchQuery] = Field(default_factory=list)
