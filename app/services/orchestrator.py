from dataclasses import dataclass, field
from functools import partial
from uuid import uuid4

from app.config import Settings
from app.core.budget import BudgetLedger
from app.core.clock import SystemClock
from app.core.enums import AgentState, IdentityStatus, StopReason
from app.core.exceptions import BudgetExceededError
from app.core.protocols import Clock, EventLogger
from app.core.state_machine import ObservableStateMachine
from app.core.stop_policy import StopContext, StopPolicy
from app.evidence.processor import EvidenceProcessingResult, EvidenceProcessor
from app.planner.service import ResearchPlanner
from app.reflection.service import ReflectionService, empty_assessment
from app.resolver.service import CompanyResolver
from app.schemas.api import ResearchRequest, ResearchResponse
from app.schemas.company import ResolvedCompany
from app.schemas.evidence import EvidenceBundle
from app.schemas.planning import QueryPlan, ResearchPlan, SearchQuery
from app.schemas.reflection import (
    FollowupPlan,
    InformationAssessment,
    ResearchSummary,
)
from app.schemas.search import SearchBatch
from app.schemas.trace import RoundTrace, RunTrace
from app.search.executor import SearchExecutor
from app.services.finalizer import FinalizedResearch, ResearchFinalizer
from app.utils.text import normalized_fingerprint_text


@dataclass(slots=True)
class _RunContext:
    request: ResearchRequest
    company: ResolvedCompany | None = None
    plan: ResearchPlan | None = None
    evidence_bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    summary: ResearchSummary | None = None
    assessment: InformationAssessment | None = None
    prior_queries: list[str] = field(default_factory=list)
    query_fingerprints: set[str] = field(default_factory=set)
    round_number: int = 0
    consecutive_failed_rounds: int = 0
    consecutive_no_new_evidence_rounds: int = 0
    refined_evidence_ids: frozenset[str] | None = None


class ResearchOrchestrator:
    """Runs one explicit, observable research state machine."""

    def __init__(
        self,
        *,
        settings: Settings,
        resolver: CompanyResolver,
        planner: ResearchPlanner,
        search_executor: SearchExecutor,
        evidence_processor: EvidenceProcessor,
        reflection: ReflectionService,
        finalizer: ResearchFinalizer,
        budget: BudgetLedger,
        logger: EventLogger,
        clock: Clock | None = None,
        stop_policy: StopPolicy | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._planner = planner
        self._search = search_executor
        self._evidence = evidence_processor
        self._reflection = reflection
        self._finalizer = finalizer
        self._budget = budget
        self._logger = logger
        self._clock = clock or SystemClock()
        self._stop_policy = stop_policy or StopPolicy()

    async def run(self, request: ResearchRequest) -> ResearchResponse:
        trace = RunTrace(
            run_id=f"run_{uuid4().hex}",
            started_at=self._clock.now(),
        )
        machine = ObservableStateMachine(trace=trace, clock=self._clock, logger=self._logger)
        context = _RunContext(request=request)

        context.company = await machine.execute(
            AgentState.RESOLVE_COMPANY,
            round_number=0,
            action=lambda: self._resolver.resolve(request),
            details={"input_fields": sorted(request.model_fields_set)},
        )
        if await self._llm_budget_exhausted():
            return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)

        context.plan = await machine.execute(
            AgentState.PLAN_RESEARCH,
            round_number=0,
            action=lambda: self._planner.build_plan(
                self._require_company(context), context.request
            ),
        )
        if await self._llm_budget_exhausted():
            self._record_unsearched_plan(context, trace, StopReason.BUDGET_REACHED)
            return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)

        query_plan: QueryPlan = await machine.execute(
            AgentState.GENERATE_SEARCH_QUERIES,
            round_number=0,
            action=lambda: self._planner.generate_queries(
                self._require_company(context),
                self._require_plan(context),
                context.query_fingerprints,
                context.request,
            ),
        )
        if await self._llm_budget_exhausted():
            self._record_unsearched_plan(
                context,
                trace,
                StopReason.BUDGET_REACHED,
                queries=query_plan.queries,
            )
            return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)

        pending_queries = await self._reserve_and_record_queries(
            query_plan.queries,
            context,
            round_number=1,
        )
        if not pending_queries:
            reason = (
                StopReason.BUDGET_REACHED
                if (await self._budget.snapshot()).query_exhausted
                else StopReason.NO_NEW_EVIDENCE
            )
            self._record_unsearched_plan(
                context,
                trace,
                reason,
                queries=query_plan.queries,
            )
            return await self._finish(machine, trace, context, reason)

        while pending_queries:
            context.round_number += 1
            round_trace = RoundTrace(
                round=context.round_number,
                planner_output=(
                    self._require_plan(context).topics if context.round_number == 1 else []
                ),
                queries=pending_queries,
            )
            trace.rounds.append(round_trace)

            batch: SearchBatch = await machine.execute(
                AgentState.PARALLEL_SEARCH,
                round_number=context.round_number,
                action=partial(
                    self._search.execute,
                    pending_queries,
                    run_id=trace.run_id,
                ),
                details={"query_ids": [query.query_id for query in pending_queries]},
            )
            round_trace.search_statistics = batch.statistics
            all_failed = (
                batch.statistics.query_count > 0 and batch.statistics.successful_queries == 0
            )
            context.consecutive_failed_rounds = (
                context.consecutive_failed_rounds + 1 if all_failed else 0
            )
            if (
                all_failed
                and context.consecutive_failed_rounds
                >= self._settings.max_consecutive_failed_rounds
            ):
                round_trace.stop_reason = StopReason.CONTINUOUS_SEARCH_FAILURE
                context.assessment = empty_assessment(
                    "Research stopped after the configured number of consecutive "
                    "fully failed search rounds."
                )
                return await self._finish(
                    machine,
                    trace,
                    context,
                    StopReason.CONTINUOUS_SEARCH_FAILURE,
                )

            previous_evidence_ids = {item.evidence_id for item in context.evidence_bundle.evidence}
            processing: EvidenceProcessingResult = await machine.execute(
                AgentState.PROCESS_EVIDENCE,
                round_number=context.round_number,
                action=partial(
                    self._evidence.process,
                    batch,
                    company_context=self._require_company(context).model_dump(mode="json"),
                ),
                details={
                    "raw_results": batch.statistics.total_results,
                    "failed_queries": batch.statistics.failed_queries,
                },
            )
            context.evidence_bundle = processing.bundle
            new_evidence_ids = {
                item.evidence_id for item in processing.bundle.evidence
            } - previous_evidence_ids
            new_evidence_count = getattr(processing, "new_evidence_count", len(new_evidence_ids))
            round_trace.new_evidence_count = new_evidence_count
            round_trace.evidence_processing_errors = list(processing.extraction_errors)
            query_ids = {query.query_id for query in pending_queries}
            round_trace.retrieved_source_ids = [
                source.source_id
                for source in processing.bundle.sources
                if query_ids.intersection(source.query_ids)
            ]
            if processing.duplicate_source_count:
                round_trace.search_statistics = batch.statistics.model_copy(
                    update={
                        "deduplicated_results": (
                            batch.statistics.deduplicated_results
                            + processing.duplicate_source_count
                        )
                    }
                )

            if all_failed:
                context.consecutive_no_new_evidence_rounds = 0
            elif new_evidence_count == 0:
                context.consecutive_no_new_evidence_rounds += 1
            else:
                context.consecutive_no_new_evidence_rounds = 0

            if await self._llm_budget_exhausted():
                return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)

            context.summary = await machine.execute(
                AgentState.SUMMARIZE,
                round_number=context.round_number,
                action=lambda: self._reflection.summarize(
                    self._require_company(context), context.evidence_bundle
                ),
            )
            round_trace.research_summary = context.summary
            if await self._llm_budget_exhausted():
                return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)

            context.assessment = await machine.execute(
                AgentState.ASSESS_INFORMATION,
                round_number=context.round_number,
                action=lambda: self._reflection.assess(
                    self._require_company(context),
                    context.evidence_bundle,
                    self._require_summary(context),
                    context.prior_queries,
                ),
            )
            round_trace.assessment = context.assessment

            refined_company: ResolvedCompany = await machine.execute(
                AgentState.REFINE_COMPANY_IDENTITY,
                round_number=context.round_number,
                action=lambda: self._refine_company(context),
                details={"evidence_count": len(context.evidence_bundle.evidence)},
            )
            context.company = refined_company
            context.refined_evidence_ids = self._evidence_revision(context.evidence_bundle)
            if refined_company.identity_status in {
                IdentityStatus.AMBIGUOUS,
                IdentityStatus.PARTIALLY_CONFIRMED,
                IdentityStatus.UNCONFIRMED,
            }:
                decision_bundle = self._constrain_evidence_to_resolved_identity(
                    refined_company,
                    context.evidence_bundle,
                )
                context.summary, context.assessment = (
                    self._reflection.reconcile_after_identity_constraint(decision_bundle)
                )
                round_trace.research_summary = context.summary
                round_trace.assessment = context.assessment

            budget_snapshot = await self._budget.snapshot()
            stop_reason = self._stop_policy.evaluate(
                StopContext(
                    assessment=self._require_assessment(context),
                    current_round=context.round_number,
                    max_rounds=self._settings.max_search_rounds,
                    budget_exhausted=budget_snapshot.exhausted,
                    consecutive_no_new_evidence_rounds=(context.consecutive_no_new_evidence_rounds),
                    no_new_evidence_limit=self._settings.no_new_evidence_rounds,
                    consecutive_failed_rounds=context.consecutive_failed_rounds,
                    failed_rounds_limit=self._settings.max_consecutive_failed_rounds,
                )
            )
            if stop_reason is not None:
                round_trace.stop_reason = stop_reason
                return await self._finish(machine, trace, context, stop_reason)

            followup: FollowupPlan = await machine.execute(
                AgentState.GENERATE_FOLLOWUP_QUERIES,
                round_number=context.round_number,
                action=lambda: self._reflection.generate_followup(
                    self._require_company(context),
                    self._require_assessment(context),
                    context.prior_queries,
                    context.query_fingerprints,
                ),
            )
            round_trace.followup = followup
            if await self._llm_budget_exhausted():
                round_trace.stop_reason = StopReason.BUDGET_REACHED
                return await self._finish(machine, trace, context, StopReason.BUDGET_REACHED)
            pending_queries = await self._reserve_and_record_queries(
                followup.queries,
                context,
                round_number=context.round_number + 1,
            )
            if not pending_queries:
                budget_exhausted = (await self._budget.snapshot()).query_exhausted
                stop_reason = (
                    StopReason.BUDGET_REACHED if budget_exhausted else StopReason.NO_NEW_EVIDENCE
                )
                round_trace.stop_reason = stop_reason
                return await self._finish(machine, trace, context, stop_reason)

        raise AssertionError("research loop exited without a stop reason")

    async def _finish(
        self,
        machine: ObservableStateMachine,
        trace: RunTrace,
        context: _RunContext,
        stop_reason: StopReason,
    ) -> ResearchResponse:
        trace.stop_reason = stop_reason
        if trace.rounds and trace.rounds[-1].stop_reason is None:
            trace.rounds[-1].stop_reason = stop_reason
        current_evidence_ids = self._evidence_revision(context.evidence_bundle)
        if context.refined_evidence_ids == current_evidence_ids:
            company = self._require_company(context)
        else:
            company = await machine.execute(
                AgentState.REFINE_COMPANY_IDENTITY,
                round_number=context.round_number,
                action=lambda: self._refine_company(context),
                details={"evidence_count": len(context.evidence_bundle.evidence)},
            )
            context.refined_evidence_ids = current_evidence_ids
        context.company = company
        pre_constraint_bundle = context.evidence_bundle
        constrained_bundle = self._constrain_evidence_to_resolved_identity(
            company,
            pre_constraint_bundle,
        )
        if (
            context.summary is None
            or context.assessment is None
            or constrained_bundle != pre_constraint_bundle
        ):
            context.summary, context.assessment = (
                self._reflection.reconcile_after_identity_constraint(constrained_bundle)
            )
        trace.final_research_summary = context.summary
        trace.final_assessment = context.assessment
        context.evidence_bundle = constrained_bundle

        async def finalize_action() -> FinalizedResearch:
            return await self._finalizer.finalize(
                company,
                context.evidence_bundle,
                self._require_assessment(context),
            )

        finalized: FinalizedResearch = await machine.execute(
            AgentState.FINALIZE,
            round_number=context.round_number,
            action=finalize_action,
            details={"stop_reason": stop_reason.value},
        )
        context.evidence_bundle = finalized.evidence_bundle
        self._reconcile_trace_source_ids(trace, finalized.evidence_bundle)

        async def return_action() -> ResearchResponse:
            usage = await self._budget.snapshot()
            trace.total_tokens = usage.total_tokens
            trace.estimated_cost_usd = usage.estimated_cost_usd
            return ResearchResponse(
                company=company,
                research=finalized.report,
                evidence=finalized.evidence_bundle,
                run_trace=trace,
            )

        response: ResearchResponse = await machine.execute(
            AgentState.RETURN_RESULT,
            round_number=context.round_number,
            action=return_action,
            details={"evidence_count": len(finalized.evidence_bundle.evidence)},
        )
        trace.completed_at = self._clock.now()
        return response

    async def _refine_company(self, context: _RunContext) -> ResolvedCompany:
        company = self._require_company(context)
        try:
            return await self._resolver.resolve(context.request, context.evidence_bundle)
        except BudgetExceededError:
            return company

    @staticmethod
    def _evidence_revision(bundle: EvidenceBundle) -> frozenset[str]:
        return frozenset(item.evidence_id for item in bundle.evidence)

    def _record_unsearched_plan(
        self,
        context: _RunContext,
        trace: RunTrace,
        stop_reason: StopReason,
        *,
        queries: list[SearchQuery] | None = None,
    ) -> None:
        if trace.rounds:
            return
        trace.rounds.append(
            RoundTrace(
                round=1,
                planner_output=self._require_plan(context).topics,
                queries=list(queries or []),
                stop_reason=stop_reason,
            )
        )

    def _reconcile_trace_source_ids(
        self,
        trace: RunTrace,
        bundle: EvidenceBundle,
    ) -> None:
        valid_source_ids = {source.source_id for source in bundle.sources}
        for round_trace in trace.rounds:
            resolved = {
                self._evidence.resolve_source_id(source_id)
                for source_id in round_trace.retrieved_source_ids
            }
            round_trace.retrieved_source_ids = sorted(resolved & valid_source_ids)

    @staticmethod
    def _constrain_evidence_to_resolved_identity(
        company: ResolvedCompany,
        bundle: EvidenceBundle,
    ) -> EvidenceBundle:
        if company.identity_status not in {
            IdentityStatus.AMBIGUOUS,
            IdentityStatus.PARTIALLY_CONFIRMED,
            IdentityStatus.UNCONFIRMED,
        }:
            return bundle
        allowed_claim_ids = {
            *company.claim_ids,
            *(claim_id for candidate in company.candidates for claim_id in candidate.claim_ids),
            *(
                claim_id
                for relationship in company.relationships
                for claim_id in relationship.claim_ids
            ),
        }
        evidence = [item for item in bundle.evidence if item.claim_id in allowed_claim_ids]
        retained_source_ids = {item.source_id for item in evidence if item.source_id is not None}
        sources = [
            source.model_copy(
                update={
                    "supported_claim_ids": sorted(
                        set(source.supported_claim_ids) & allowed_claim_ids
                    )
                }
            )
            for source in bundle.sources
            if source.source_id in retained_source_ids
        ]
        conflicts = [
            conflict
            for conflict in bundle.conflicts
            if set(conflict.claim_ids).issubset(allowed_claim_ids)
        ]
        return bundle.model_copy(
            update={
                "sources": sources,
                "evidence": evidence,
                "conflicts": conflicts,
            }
        )

    async def _reserve_and_record_queries(
        self,
        queries: list[SearchQuery],
        context: _RunContext,
        *,
        round_number: int,
    ) -> list[SearchQuery]:
        allowed_count = await self._budget.allow_queries(len(queries))
        selected = [
            query.model_copy(update={"round": round_number}) for query in queries[:allowed_count]
        ]
        for query in selected:
            context.prior_queries.append(query.query)
            context.query_fingerprints.add(normalized_fingerprint_text(query.query))
        return selected

    async def _llm_budget_exhausted(self) -> bool:
        return (await self._budget.snapshot()).llm_exhausted

    @staticmethod
    def _require_company(context: _RunContext) -> ResolvedCompany:
        if context.company is None:
            raise RuntimeError("company resolution is unavailable")
        return context.company

    @staticmethod
    def _require_plan(context: _RunContext) -> ResearchPlan:
        if context.plan is None:
            raise RuntimeError("research plan is unavailable")
        return context.plan

    @staticmethod
    def _require_summary(context: _RunContext) -> ResearchSummary:
        if context.summary is None:
            raise RuntimeError("research summary is unavailable")
        return context.summary

    @staticmethod
    def _require_assessment(context: _RunContext) -> InformationAssessment:
        if context.assessment is None:
            raise RuntimeError("information assessment is unavailable")
        return context.assessment
