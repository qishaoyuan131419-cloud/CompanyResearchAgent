from pydantic import Field

from app.core.enums import AssessmentDecision
from app.schemas.base import StrictModel
from app.schemas.planning import SearchQuery


class KnownFact(StrictModel):
    statement: str
    claim_ids: list[str] = Field(min_length=1)


class ResearchSummary(StrictModel):
    known_facts: list[KnownFact] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    weak_evidence: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class DimensionAssessment(StrictModel):
    dimension: str
    coverage_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_items: list[str] = Field(default_factory=list)


class InformationAssessment(StrictModel):
    dimensions: list[DimensionAssessment]
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    weak_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_followup_queries: list[str] = Field(default_factory=list)
    decision: AssessmentDecision
    decision_reason: str


class FollowupPlan(StrictModel):
    queries: list[SearchQuery] = Field(default_factory=list)
    addressed_missing_items: list[str] = Field(default_factory=list)
