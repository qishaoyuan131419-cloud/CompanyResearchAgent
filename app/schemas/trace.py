from datetime import datetime
from typing import Any

from pydantic import Field

from app.core.enums import AgentState, StopReason, TransitionOutcome
from app.schemas.base import StrictModel
from app.schemas.evidence import ProcessingError
from app.schemas.planning import ResearchTopic, SearchQuery
from app.schemas.reflection import FollowupPlan, InformationAssessment, ResearchSummary
from app.schemas.search import SearchStatistics


class StateTransition(StrictModel):
    sequence: int = Field(ge=1)
    from_state: AgentState | None = None
    to_state: AgentState
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0.0)
    round: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    outcome: TransitionOutcome = TransitionOutcome.SUCCEEDED
    error: str | None = None


class RoundTrace(StrictModel):
    round: int = Field(ge=1)
    planner_output: list[ResearchTopic] = Field(default_factory=list)
    queries: list[SearchQuery] = Field(default_factory=list)
    search_statistics: SearchStatistics | None = None
    retrieved_source_ids: list[str] = Field(default_factory=list)
    research_summary: ResearchSummary | None = None
    assessment: InformationAssessment | None = None
    followup: FollowupPlan | None = None
    new_evidence_count: int = Field(default=0, ge=0)
    evidence_processing_errors: list[ProcessingError] = Field(default_factory=list)
    stop_reason: StopReason | None = None


class BudgetStatus(StrictModel):
    budget_type: str
    configured_limit: float | int
    used: float | int
    remaining: float | int
    reached_at_stage: str | None = None


class RunTrace(StrictModel):
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    rounds: list[RoundTrace] = Field(default_factory=list)
    transitions: list[StateTransition] = Field(default_factory=list)
    final_research_summary: ResearchSummary | None = None
    final_assessment: InformationAssessment | None = None
    stop_reason: StopReason | None = None
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    source_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    verified_fact_count: int = Field(default=0, ge=0)
    single_source_count: int = Field(default=0, ge=0)
    inference_count: int = Field(default=0, ge=0)
    research_gap_count: int = Field(default=0, ge=0)
    rejected_claim_count: int = Field(default=0, ge=0)
    processing_error_count: int = Field(default=0, ge=0)
    budget_status: list[BudgetStatus] = Field(default_factory=list)
