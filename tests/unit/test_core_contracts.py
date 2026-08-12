import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.core.enums import (
    AgentState,
    AssessmentDecision,
    EvidenceStatus,
    SourceType,
    StopReason,
    TransitionOutcome,
)
from app.core.exceptions import InvalidStateTransitionError
from app.core.state_machine import ObservableStateMachine
from app.core.stop_policy import StopContext, StopPolicy
from app.schemas.company import CompanySnapshot
from app.schemas.evidence import Evidence
from app.schemas.reflection import DimensionAssessment, InformationAssessment
from app.schemas.research import ResearchFinding, ResearchReport, ResearchSection
from app.schemas.trace import RunTrace


class IncrementingClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        self.value += timedelta(milliseconds=1)
        return self.value


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def assessment(decision: AssessmentDecision) -> InformationAssessment:
    return InformationAssessment(
        dimensions=[
            DimensionAssessment(
                dimension="Company Identity",
                coverage_score=0.8,
                confidence=0.8,
                missing_items=[],
            )
        ],
        evidence_coverage=0.8,
        decision=decision,
        decision_reason="test",
    )


def test_minimum_company_input_is_strict() -> None:
    assert CompanySnapshot(canonical_name=" Acme Pharma ").canonical_name == "Acme Pharma"
    with pytest.raises(ValidationError):
        CompanySnapshot(canonical_name=" ")


def test_supported_evidence_requires_complete_retrieved_lineage() -> None:
    with pytest.raises(ValidationError, match="at least one source ID"):
        Evidence(
            evidence_id="evd_1",
            claim_id="clm_1",
            claim="Acme has a product",
            value="Product A",
            status=EvidenceStatus.SINGLE_SOURCE,
            confidence=0.5,
            retrieved_at=datetime.now(UTC),
        )
    with pytest.raises(ValidationError, match="research gaps"):
        Evidence(
            evidence_id="evd_2",
            claim_id="clm_2",
            claim="Acme pipeline stage",
            value="Unknown",
            status=EvidenceStatus.UNKNOWN,
            confidence=0.0,
            retrieved_at=datetime.now(UTC),
        )


def test_research_references_claims_and_rejects_urls() -> None:
    with pytest.raises(ValidationError, match="claim IDs, not URLs"):
        ResearchReport(
            overview=ResearchSection(
                findings=[
                    ResearchFinding(
                        statement="According to https://fabricated.invalid, this is true",
                        claim_ids=["clm_1"],
                    )
                ]
            )
        )


@pytest.mark.parametrize(
    ("decision", "failed", "budget", "round_number", "no_new", "expected"),
    [
        (AssessmentDecision.STOP, 2, True, 3, 2, StopReason.SUFFICIENT_INFORMATION),
        (
            AssessmentDecision.CONTINUE_SEARCH,
            2,
            True,
            3,
            2,
            StopReason.CONTINUOUS_SEARCH_FAILURE,
        ),
        (
            AssessmentDecision.CONTINUE_SEARCH,
            0,
            True,
            3,
            2,
            StopReason.BUDGET_REACHED,
        ),
        (
            AssessmentDecision.CONTINUE_SEARCH,
            0,
            False,
            3,
            2,
            StopReason.MAX_ROUNDS_REACHED,
        ),
        (
            AssessmentDecision.CONTINUE_SEARCH,
            0,
            False,
            1,
            2,
            StopReason.NO_NEW_EVIDENCE,
        ),
    ],
)
def test_stop_policy_has_deterministic_precedence(
    decision: AssessmentDecision,
    failed: int,
    budget: bool,
    round_number: int,
    no_new: int,
    expected: StopReason,
) -> None:
    reason = StopPolicy().evaluate(
        StopContext(
            assessment=assessment(decision),
            current_round=round_number,
            max_rounds=3,
            budget_exhausted=budget,
            consecutive_no_new_evidence_rounds=no_new,
            no_new_evidence_limit=2,
            consecutive_failed_rounds=failed,
            failed_rounds_limit=2,
        )
    )
    assert reason == expected


@pytest.mark.asyncio
async def test_state_machine_records_legal_transitions() -> None:
    clock = IncrementingClock()
    trace = RunTrace(run_id="run_test", started_at=clock.now())
    logger = RecordingLogger()
    machine = ObservableStateMachine(trace=trace, clock=clock, logger=logger)

    result = await machine.execute(
        AgentState.RESOLVE_COMPANY,
        round_number=0,
        action=lambda: "resolved",
    )
    await machine.execute(AgentState.PLAN_RESEARCH, round_number=0, action=lambda: None)

    assert result == "resolved"
    assert [event.to_state for event in trace.transitions] == [
        AgentState.RESOLVE_COMPANY,
        AgentState.PLAN_RESEARCH,
    ]
    assert all(event.completed_at >= event.started_at for event in trace.transitions)
    assert len(logger.events) == 2


@pytest.mark.asyncio
async def test_state_machine_rejects_illegal_transition() -> None:
    clock = IncrementingClock()
    trace = RunTrace(run_id="run_test", started_at=clock.now())
    machine = ObservableStateMachine(trace=trace, clock=clock, logger=RecordingLogger())
    with pytest.raises(InvalidStateTransitionError):
        await machine.execute(AgentState.PARALLEL_SEARCH, round_number=1, action=lambda: None)
    assert trace.transitions == []


@pytest.mark.asyncio
async def test_state_machine_records_cancellation_as_failed_transition() -> None:
    clock = IncrementingClock()
    trace = RunTrace(run_id="run_test", started_at=clock.now())
    machine = ObservableStateMachine(
        trace=trace,
        clock=clock,
        logger=RecordingLogger(),
    )

    async def cancel() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await machine.execute(
            AgentState.RESOLVE_COMPANY,
            round_number=0,
            action=cancel,
        )

    assert trace.transitions[0].outcome == TransitionOutcome.FAILED
    assert trace.transitions[0].error == "CancelledError:"


def test_complete_evidence_can_be_constructed_from_retrieval_metadata() -> None:
    evidence = Evidence(
        evidence_id="evd_1",
        claim_id="clm_1",
        claim="Acme announced Product A",
        value="Product A",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_1",
        source="acme.example",
        title="Acme announcement",
        url="https://acme.example/news/a",
        retrieved_at=datetime.now(UTC),
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme announced Product A.",
    )
    assert str(evidence.url) == "https://acme.example/news/a"
