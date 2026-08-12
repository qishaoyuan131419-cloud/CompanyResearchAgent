import re
from datetime import datetime
from typing import Any, TypeAlias

from pydantic import AnyHttpUrl, Field, JsonValue, field_validator, model_validator

from app.core.enums import EvidenceStatus, ExtractionMethod, ProcessingStage, SourceType
from app.schemas.base import StrictModel
from app.schemas.search import SourceDocument
from app.utils.hashing import stable_hash

EvidenceValue: TypeAlias = JsonValue


class SupportingQuote(StrictModel):
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=5, max_length=300)


class ExtractedClaim(StrictModel):
    claim: str = Field(min_length=1)
    value: EvidenceValue
    status: EvidenceStatus = EvidenceStatus.SINGLE_SOURCE
    source_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[SupportingQuote | str] = Field(default_factory=list)
    is_inference: bool = False
    derived_from_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Existing supported claim IDs used as premises; required when is_inference is true"
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("claim")
    @classmethod
    def reject_prohibited_claim_content(cls, value: str) -> str:
        lowered = value.casefold()
        prohibited = (
            "http://",
            "https://",
            "www.",
            "](",
            "source_id",
            "source id",
            "according to source",
            "```",
        )
        citation = re.search(r"\[(?:\d+|src_[^\]]+)\]", lowered)
        if any(token in lowered for token in prohibited) or citation:
            raise ValueError("claim contains a prohibited URL, citation, or source identifier")
        return value


class RejectedClaim(StrictModel):
    index: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=1_000)
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)


class ProcessingError(StrictModel):
    error_id: str
    stage: ProcessingStage
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1_000)
    recoverable: bool
    source_id: str | None = None
    query_id: str | None = None
    claim_index: int | None = Field(default=None, ge=0)


class Evidence(StrictModel):
    evidence_id: str
    claim_id: str
    claim: str
    value: EvidenceValue
    status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)
    derived_from_claim_ids: list[str] = Field(default_factory=list)
    extraction_method: ExtractionMethod = ExtractionMethod.STRUCTURED_LLM
    # Deprecated compatibility-only fields. They are excluded from API output
    # and never consulted by application logic; SourceDocument is authoritative.
    source_id: str | None = Field(default=None, exclude=True, description="Deprecated input alias")
    source: str | None = Field(default=None, exclude=True, description="Deprecated input only")
    title: str | None = Field(default=None, exclude=True, description="Deprecated input only")
    url: AnyHttpUrl | None = Field(default=None, exclude=True, description="Deprecated input only")
    published_at: datetime | None = Field(
        default=None, exclude=True, description="Deprecated input only"
    )
    retrieved_at: datetime | None = Field(
        default=None, exclude=True, description="Deprecated input only"
    )
    source_type: SourceType | None = Field(
        default=None, exclude=True, description="Deprecated input only"
    )
    supporting_quote: str | None = Field(
        default=None, exclude=True, description="Deprecated input alias"
    )

    @model_validator(mode="after")
    def enforce_source_lineage(self) -> "Evidence":
        if self.source_id and not self.source_ids:
            self.source_ids = [self.source_id]
        if self.supporting_quote and not self.supporting_quotes:
            self.supporting_quotes = [self.supporting_quote]
        if self.source_ids and self.source_id is None:
            self.source_id = self.source_ids[0]
        if self.supporting_quotes and self.supporting_quote is None:
            self.supporting_quote = self.supporting_quotes[0]
        self.source_ids = list(dict.fromkeys(self.source_ids))
        self.derived_from_claim_ids = list(dict.fromkeys(self.derived_from_claim_ids))
        if self.status in {EvidenceStatus.VERIFIED_FACT, EvidenceStatus.SINGLE_SOURCE}:
            if not self.source_ids:
                raise ValueError("source-backed evidence requires at least one source ID")
        if self.status == EvidenceStatus.INFERENCE and not self.derived_from_claim_ids:
            raise ValueError("inference evidence requires derived_from_claim_ids")
        if self.status == EvidenceStatus.UNKNOWN and not self.source_ids:
            raise ValueError("unsupported unknowns belong in research gaps, not evidence")
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
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    processing_errors: list[ProcessingError] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    verified_fact_count: int = Field(default=0, ge=0)
    single_source_count: int = Field(default=0, ge=0)
    inference_count: int = Field(default=0, ge=0)
    research_gap_count: int = Field(default=0, ge=0)
    rejected_claim_count: int = Field(default=0, ge=0)
    processing_error_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_source_links(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_evidence = value.get("evidence", [])
        raw_sources = value.get("sources", [])
        evidence_items = [
            item if isinstance(item, Evidence) else Evidence.model_validate(item)
            for item in raw_evidence
        ]
        merged_by_claim: dict[str, Evidence] = {}
        for item in evidence_items:
            existing = merged_by_claim.get(item.claim_id)
            if existing is None:
                merged_by_claim[item.claim_id] = item
                continue
            if (
                existing.claim != item.claim
                or existing.value != item.value
                or existing.status != item.status
            ):
                continue
            merged_by_claim[item.claim_id] = existing.model_copy(
                update={
                    "source_ids": list(dict.fromkeys([*existing.source_ids, *item.source_ids])),
                    "supporting_quotes": list(
                        dict.fromkeys([*existing.supporting_quotes, *item.supporting_quotes])
                    ),
                    "confidence": max(existing.confidence, item.confidence),
                }
            )
        evidence_items = list(merged_by_claim.values())
        source_items = [
            item if isinstance(item, SourceDocument) else SourceDocument.model_validate(item)
            for item in raw_sources
        ]
        by_source = {item.source_id: item for item in source_items}
        claims_by_source: dict[str, set[str]] = {
            item.source_id: set(item.supported_claim_ids) for item in source_items
        }
        for item in evidence_items:
            for source_id in item.source_ids:
                claims_by_source.setdefault(source_id, set()).add(item.claim_id)
                if source_id in by_source:
                    continue
                if not all((item.title, item.url, item.retrieved_at, item.source_type)):
                    continue
                by_source[source_id] = SourceDocument(
                    source_id=source_id,
                    title=item.title or "Legacy source",
                    url=item.url,
                    publisher=item.source,
                    published_at=item.published_at,
                    retrieved_at=item.retrieved_at,
                    source_type=item.source_type,
                    summary=None,
                    content_hash=stable_hash(
                        {"source_id": source_id, "url": str(item.url)}, prefix="cnt_"
                    ),
                )
        normalized_sources = [
            source.model_copy(
                update={"supported_claim_ids": sorted(claims_by_source[source.source_id])}
            )
            for source in by_source.values()
        ]
        return {**value, "evidence": evidence_items, "sources": normalized_sources}

    @model_validator(mode="after")
    def calculate_and_validate_metrics(self) -> "EvidenceBundle":
        source_ids = {source.source_id for source in self.sources}
        claim_ids = {item.claim_id for item in self.evidence}
        if len(claim_ids) != len(self.evidence):
            raise ValueError("evidence must contain at most one item per claim ID")
        for item in self.evidence:
            missing = set(item.source_ids) - source_ids
            if missing:
                raise ValueError(f"evidence references unknown source IDs: {sorted(missing)}")
        for source in self.sources:
            missing = set(source.supported_claim_ids) - claim_ids
            if missing:
                raise ValueError(f"source references unknown claim IDs: {sorted(missing)}")
        supported_by_source = {
            source.source_id: set(source.supported_claim_ids) for source in self.sources
        }
        for item in self.evidence:
            for source_id in item.source_ids:
                if item.claim_id not in supported_by_source[source_id]:
                    raise ValueError(
                        f"source {source_id} does not reciprocally reference claim {item.claim_id}"
                    )
        self.source_count = len(self.sources)
        self.supported_claim_count = len(self.evidence)
        self.verified_fact_count = sum(
            item.status == EvidenceStatus.VERIFIED_FACT for item in self.evidence
        )
        self.single_source_count = sum(
            item.status == EvidenceStatus.SINGLE_SOURCE for item in self.evidence
        )
        self.inference_count = sum(
            item.status == EvidenceStatus.INFERENCE for item in self.evidence
        )
        self.rejected_claim_count = len(self.rejected_claims)
        self.processing_error_count = len(self.processing_errors)
        return self


def processing_error_id(*, stage: ProcessingStage, code: str, discriminator: Any) -> str:
    return stable_hash(
        {"stage": stage.value, "code": code, "discriminator": discriminator},
        prefix="err_",
        length=24,
    )
