import json
from datetime import UTC, datetime
from typing import Any

import pytest

from app.config import Settings
from app.core.budget import BudgetLedger
from app.core.enums import (
    AgentState,
    AssessmentDecision,
    IdentityStatus,
    SourceType,
    StopReason,
)
from app.core.exceptions import StructuredOutputError
from app.core.protocols import LLMUsage, StructuredLLMResult
from app.evidence.processor import ClaimExtractionResponse, EvidenceProcessor
from app.evidence.registry import SourceRegistry
from app.planner.service import ResearchPlanner
from app.reflection.service import ASSESSMENT_DIMENSIONS, ReflectionService
from app.resolver.service import CompanyResolver
from app.schemas.api import ResearchRequest
from app.schemas.company import ResolvedCompany
from app.schemas.evidence import ExtractedClaim, SupportingQuote
from app.schemas.planning import QueryPlan, ResearchPlan, ResearchTopic, SearchQuery
from app.schemas.reflection import (
    DimensionAssessment,
    FollowupPlan,
    InformationAssessment,
    ResearchSummary,
)
from app.schemas.research import ResearchReport
from app.schemas.search import SearchResult
from app.search.errors import RetryableSearchError
from app.search.executor import SearchExecutor
from app.services.container import ApplicationContainer
from app.services.finalizer import ResearchFinalizer
from app.services.orchestrator import ResearchOrchestrator


class FakePrompts:
    def render(self, name: str, variables: Any) -> str:
        return f"stage={name}; keys={','.join(sorted(variables))}"


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class WorkflowLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        cache_namespace: str,
    ) -> StructuredLLMResult[Any]:
        del system_prompt, cache_namespace
        self.calls.append(response_model.__name__)
        if response_model is ResolvedCompany:
            payload = json.loads(user_prompt)
            evidence = payload.get("evidence_bundle", {}).get("evidence", [])
            claim_ids = list(
                dict.fromkeys(
                    item["claim_id"] for item in evidence if "identity" in item["claim"].casefold()
                )
            )
            value = ResolvedCompany(
                canonical_name="Acme Pharma",
                identity_status="confirmed",
                confidence=0.9,
                claim_ids=claim_ids,
            )
        elif response_model is ResearchPlan:
            value = ResearchPlan(
                topics=[
                    ResearchTopic(
                        topic="Company Identity",
                        priority=5,
                        reason="Resolve the entity.",
                        expected_output="Supported identity information.",
                        estimated_value=1.0,
                    )
                ]
            )
        elif response_model is QueryPlan:
            value = QueryPlan(
                queries=[
                    _query("Acme identity official"),
                    _query("Acme identity regulatory"),
                ]
            )
        elif response_model is ClaimExtractionResponse:
            sources = json.loads(user_prompt)["sources"]
            source_ids = [item["source_id"] for item in sources]
            value = ClaimExtractionResponse(
                claims=[
                    ExtractedClaim(
                        claim="Company identity",
                        value="Acme Pharma",
                        source_ids=source_ids,
                        supporting_quotes=[
                            SupportingQuote(
                                source_id=item["source_id"],
                                quote=f"{item['content'].split('.', 1)[0]}.",
                            )
                            for item in sources
                        ],
                        confidence=0.9,
                    )
                ]
            )
        elif response_model is ResearchSummary:
            value = ResearchSummary(missing_information=["Current pipeline stage"])
        elif response_model is InformationAssessment:
            value = InformationAssessment(
                dimensions=[
                    DimensionAssessment(
                        dimension=name,
                        coverage_score=1.0,
                        confidence=1.0,
                        missing_items=(
                            [] if name == "Company Identity" else ["Current pipeline stage"]
                        ),
                    )
                    for name in ASSESSMENT_DIMENSIONS
                ],
                evidence_coverage=1.0,
                missing_information=["Current pipeline stage"],
                recommended_followup_queries=["Acme pipeline clinical trial status"],
                decision="continue_search",
                decision_reason="Gaps remain.",
            )
        elif response_model is FollowupPlan:
            value = FollowupPlan(
                queries=[
                    _query(
                        "Acme identity official",
                        addresses=["Current pipeline stage"],
                    ),
                    _query(
                        "Acme pipeline clinical trial status",
                        topic="Pipeline",
                        addresses=["Current pipeline stage"],
                    ),
                ],
                addressed_missing_items=["Current pipeline stage"],
            )
        elif response_model is ResearchReport:
            raise StructuredOutputError("force deterministic evidence-backed finalizer")
        else:
            raise AssertionError(f"unexpected response model: {response_model}")
        return StructuredLLMResult(value=value, usage=LLMUsage(), model="fake")


class FakeSearchClient:
    def __init__(self, mode: str = "results") -> None:
        self.mode = mode
        self.queries: list[str] = []

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del limit
        self.queries.append(query.query)
        if self.mode == "failure":
            raise RetryableSearchError("scripted outage")
        if self.mode == "empty":
            return []
        suffix = query.query_id.removeprefix("qry_")
        descriptor = (
            "official source one"
            if "official" in query.query.casefold()
            else "regulator source two"
        )
        return [
            SearchResult(
                title=f"Official source for {query.topic}",
                url=f"https://source-{suffix}.com/evidence",
                text=(
                    "Acme Pharma company identity is Acme Pharma "
                    f"according to {descriptor}. "
                    f"Source record for {query.query_id}."
                ),
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
                source_type=SourceType.OFFICIAL,
            )
        ]


class AlwaysPartialResolver:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        snapshot: ResearchRequest,
        evidence_bundle: Any = None,
    ) -> ResolvedCompany:
        self.calls += 1
        evidence = evidence_bundle.evidence if evidence_bundle is not None else []
        claim_ids = [item.claim_id for item in evidence if "identity" in item.claim.casefold()]
        return ResolvedCompany(
            canonical_name=snapshot.canonical_name,
            identity_status=(
                IdentityStatus.PARTIALLY_CONFIRMED if claim_ids else IdentityStatus.UNCONFIRMED
            ),
            confidence=0.7 if claim_ids else 0.0,
            claim_ids=list(dict.fromkeys(claim_ids)),
        )


class OptimisticReflection:
    def __init__(self, delegate: ReflectionService) -> None:
        self._delegate = delegate

    async def summarize(self, company: ResolvedCompany, bundle: Any) -> ResearchSummary:
        return await self._delegate.summarize(company, bundle)

    async def assess(self, *args: Any, **kwargs: Any) -> InformationAssessment:
        del args, kwargs
        return InformationAssessment(
            dimensions=[
                DimensionAssessment(
                    dimension=dimension,
                    coverage_score=1.0,
                    confidence=1.0,
                    missing_items=[],
                )
                for dimension in ASSESSMENT_DIMENSIONS
            ],
            evidence_coverage=1.0,
            decision=AssessmentDecision.STOP,
            decision_reason="Untrusted optimistic assessment.",
        )

    async def generate_followup(self, *args: Any, **kwargs: Any) -> FollowupPlan:
        return await self._delegate.generate_followup(*args, **kwargs)

    def reconcile_after_identity_constraint(
        self,
        bundle: Any,
    ) -> tuple[ResearchSummary, InformationAssessment]:
        return self._delegate.reconcile_after_identity_constraint(bundle)


def _query(
    text: str,
    *,
    topic: str = "Company Identity",
    addresses: list[str] | None = None,
) -> SearchQuery:
    return SearchQuery(
        query_id="pending",
        query=text,
        topic=topic,
        priority=5,
        expected_evidence="Supported company information",
        addresses_missing_items=addresses or [],
    )


def build_orchestrator(
    *,
    search_mode: str,
    max_rounds: int,
    failure_rounds: int = 2,
    no_new_rounds: int = 1,
) -> tuple[ResearchOrchestrator, FakeSearchClient]:
    settings = Settings(
        environment="test",
        cache_enabled=False,
        max_search_rounds=max_rounds,
        max_queries_per_round=5,
        max_total_queries=10,
        max_consecutive_failed_rounds=failure_rounds,
        no_new_evidence_rounds=no_new_rounds,
        search_max_retries=0,
        search_timeout_seconds=1,
        search_max_concurrency=2,
        minimum_dimension_coverage=0.1,
        sufficient_coverage_threshold=0.5,
    )
    llm = WorkflowLLM()
    prompts = FakePrompts()
    logger = RecordingLogger()
    search_client = FakeSearchClient(search_mode)
    search = SearchExecutor.from_settings(search_client, settings, logger=logger)
    registry = SourceRegistry(clock=search._clock)  # Same deterministic clock boundary.
    evidence = EvidenceProcessor(
        llm_client=llm,
        prompt_repository=prompts,
        registry=registry,
        subject_identifiers=("Acme Pharma",),
        evidence_limit=50,
    )
    budget = BudgetLedger(token_limit=10_000, cost_limit_usd=10, query_limit=10)
    orchestrator = ResearchOrchestrator(
        settings=settings,
        resolver=CompanyResolver(llm=llm, prompts=prompts),
        planner=ResearchPlanner(llm=llm, prompts=prompts, max_queries_per_round=5),
        search_executor=search,
        evidence_processor=evidence,
        reflection=ReflectionService(llm=llm, prompts=prompts, settings=settings),
        finalizer=ResearchFinalizer(llm=llm, prompts=prompts),
        budget=budget,
        logger=logger,
    )
    return orchestrator, search_client


@pytest.mark.asyncio
async def test_full_followup_workflow_is_traced_and_truthful() -> None:
    orchestrator, search = build_orchestrator(
        search_mode="results",
        max_rounds=2,
        no_new_rounds=2,
    )
    response = await orchestrator.run(ResearchRequest(canonical_name="Acme Pharma"))

    assert response.run_trace.stop_reason == StopReason.MAX_ROUNDS_REACHED
    assert len(response.run_trace.rounds) == 2
    assert [transition.to_state for transition in response.run_trace.transitions] == [
        AgentState.RESOLVE_COMPANY,
        AgentState.PLAN_RESEARCH,
        AgentState.GENERATE_SEARCH_QUERIES,
        AgentState.PARALLEL_SEARCH,
        AgentState.PROCESS_EVIDENCE,
        AgentState.SUMMARIZE,
        AgentState.ASSESS_INFORMATION,
        AgentState.REFINE_COMPANY_IDENTITY,
        AgentState.GENERATE_FOLLOWUP_QUERIES,
        AgentState.PARALLEL_SEARCH,
        AgentState.PROCESS_EVIDENCE,
        AgentState.SUMMARIZE,
        AgentState.ASSESS_INFORMATION,
        AgentState.REFINE_COMPANY_IDENTITY,
        AgentState.FINALIZE,
        AgentState.RETURN_RESULT,
    ]
    assert search.queries.count("Acme identity official") == 1
    assert "Acme Pharma drug pipeline clinical trial phase stage" in search.queries
    provider_urls = {
        f"https://source-{query.query_id.removeprefix('qry_')}.com/evidence"
        for round_trace in response.run_trace.rounds
        for query in round_trace.queries
    }
    for item in response.evidence.evidence:
        if item.url is not None:
            assert str(item.url) in provider_urls
    serialized_research = response.research.model_dump_json().casefold()
    assert "http://" not in serialized_research
    assert "https://" not in serialized_research
    assert any(item.status.value == "verified_fact" for item in response.evidence.evidence)
    assert response.run_trace.completed_at is not None
    assert response.run_trace.completed_at >= response.run_trace.transitions[-1].completed_at
    assert response.run_trace.rounds[0].assessment != response.run_trace.rounds[1].assessment
    assert (
        response.run_trace.final_research_summary == response.run_trace.rounds[-1].research_summary
    )
    assert response.run_trace.final_assessment == response.run_trace.rounds[-1].assessment
    final_source_ids = {source.source_id for source in response.evidence.sources}
    assert all(
        source_id in final_source_ids
        for round_trace in response.run_trace.rounds
        for source_id in round_trace.retrieved_source_ids
    )


@pytest.mark.asyncio
async def test_identity_refinement_prevents_stale_sufficient_stop_and_is_not_repeated() -> None:
    orchestrator, _ = build_orchestrator(search_mode="results", max_rounds=1)
    resolver = AlwaysPartialResolver()
    orchestrator._resolver = resolver
    orchestrator._reflection = OptimisticReflection(orchestrator._reflection)

    response = await orchestrator.run(ResearchRequest(canonical_name="Acme Pharma"))

    assert response.run_trace.stop_reason == StopReason.MAX_ROUNDS_REACHED
    assert resolver.calls == 2
    assert (
        sum(
            transition.to_state == AgentState.REFINE_COMPANY_IDENTITY
            for transition in response.run_trace.transitions
        )
        == 1
    )
    assert response.run_trace.rounds[0].assessment is not None
    assert response.run_trace.rounds[0].assessment.decision == AssessmentDecision.CONTINUE_SEARCH
    assert response.run_trace.final_assessment is not None
    assert response.run_trace.final_assessment.decision == AssessmentDecision.CONTINUE_SEARCH


@pytest.mark.asyncio
async def test_successful_zero_result_round_stops_as_no_new_evidence() -> None:
    orchestrator, _ = build_orchestrator(search_mode="empty", max_rounds=3)
    response = await orchestrator.run(ResearchRequest(canonical_name="Acme Pharma"))
    assert response.run_trace.stop_reason == StopReason.NO_NEW_EVIDENCE
    assert len(response.run_trace.rounds) == 1
    assert response.run_trace.rounds[0].search_statistics is not None
    assert response.run_trace.rounds[0].search_statistics.successful_queries == 2


@pytest.mark.asyncio
async def test_consecutive_all_failed_rounds_use_failure_stop_reason() -> None:
    orchestrator, _ = build_orchestrator(
        search_mode="failure",
        max_rounds=3,
        failure_rounds=2,
        no_new_rounds=1,
    )
    response = await orchestrator.run(ResearchRequest(canonical_name="Acme Pharma"))
    assert response.run_trace.stop_reason == StopReason.CONTINUOUS_SEARCH_FAILURE
    assert len(response.run_trace.rounds) == 2
    second_round_states = [
        transition.to_state
        for transition in response.run_trace.transitions
        if transition.round == 2
    ]
    assert second_round_states == [
        AgentState.PARALLEL_SEARCH,
        AgentState.FINALIZE,
        AgentState.RETURN_RESULT,
    ]


@pytest.mark.asyncio
async def test_tiny_token_budget_returns_partial_unknown_without_network_calls() -> None:
    settings = Settings(
        environment="test",
        cache_enabled=False,
        llm_api_key="test-key",
        llm_base_url="http://localhost:9998",
        llm_model="test-model",
        exa_mcp_url="http://localhost:9999/mcp",
        token_budget=1,
        cost_budget_usd=0,
    )
    container = ApplicationContainer(settings)
    try:
        response = await container.research(ResearchRequest(canonical_name="Budget Limited Pharma"))
    finally:
        await container.close()

    assert response.run_trace.stop_reason == StopReason.BUDGET_REACHED
    assert response.company.identity_status.value == "unconfirmed"
    assert len(response.run_trace.rounds) == 1
    assert response.run_trace.rounds[0].planner_output
    assert response.run_trace.rounds[0].queries == []
    assert response.run_trace.rounds[0].search_statistics is None
    assert response.run_trace.rounds[0].stop_reason == StopReason.BUDGET_REACHED
    assert any("Products" in finding.statement for finding in response.research.unknowns.findings)
    assert [transition.to_state for transition in response.run_trace.transitions] == [
        AgentState.RESOLVE_COMPANY,
        AgentState.PLAN_RESEARCH,
        AgentState.REFINE_COMPANY_IDENTITY,
        AgentState.FINALIZE,
        AgentState.RETURN_RESULT,
    ]
