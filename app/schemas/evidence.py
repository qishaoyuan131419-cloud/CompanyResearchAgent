from datetime import datetime
from typing import TypeAlias

from pydantic import AnyHttpUrl, Field, model_validator

from app.core.enums import EvidenceStatus, SourceType
from app.schemas.base import StrictModel
from app.schemas.search import SourceDocument

JsonScalar: TypeAlias = str | int | float | bool | None
EvidenceValue: TypeAlias = JsonScalar | list[JsonScalar]


class SupportingQuote(StrictModel):
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=5, max_length=300)


class ExtractedClaim(StrictModel):
    claim: str = Field(min_length=1)
    value: EvidenceValue
    source_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[SupportingQuote] = Field(default_factory=list)
    is_inference: bool = False
    derived_from_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Existing supported claim IDs used as premises; required when is_inference is true"
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Evidence(StrictModel):
    evidence_id: str
    claim_id: str
    claim: str
    value: EvidenceValue
    status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source_id: str | None = None
    source: str | None = None
    title: str | None = None
    url: AnyHttpUrl | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    source_type: SourceType | None = None
    supporting_quote: str | None = Field(default=None, min_length=5, max_length=300)
    derived_from_claim_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_source_lineage(self) -> "Evidence":
        source_metadata = (self.source, self.title, self.url, self.source_type)
        if self.source_id is not None:
            if not all(source_metadata) or not self.supporting_quote:
                raise ValueError("source-backed evidence must retain complete quoted lineage")
        elif any(item is not None for item in (*source_metadata, self.supporting_quote)):
            raise ValueError("source lineage fields require source_id")
        if self.status != EvidenceStatus.UNKNOWN and self.source_id is None:
            raise ValueError("non-unknown evidence must retain a complete source lineage")
        return self


class EvidenceConflict(StrictModel):
    claim: str
    claim_ids: list[str] = Field(min_length=2)
    values: list[EvidenceValue] = Field(min_length=2)
    description: str


class EvidenceBundle(StrictModel):
    sources: list[SourceDocument] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    rejected_claims: int = Field(default=0, ge=0)
    processing_errors: list[str] = Field(default_factory=list)
