from datetime import UTC, datetime
from typing import Any

import pytest

from app.config import Settings
from app.core.enums import AssessmentDecision, EvidenceStatus, IdentityStatus, SourceType
from app.core.protocols import LLMUsage, StructuredLLMResult
from app.reflection.service import ASSESSMENT_DIMENSIONS, ReflectionService
from app.schemas.company import ResolvedCompany
from app.schemas.evidence import Evidence, EvidenceBundle
from app.schemas.planning import SearchQuery
from app.schemas.reflection import (
    DimensionAssessment,
    FollowupPlan,
    InformationAssessment,
    KnownFact,
    ResearchSummary,
)
from app.schemas.research import ResearchFinding, ResearchReport, ResearchSection
from app.services.finalizer import ResearchFinalizer
from app.services.orchestrator import ResearchOrchestrator

NOW = datetime(2026, 7, 29, tzinfo=UTC)


class StaticPrompts:
    def render(self, name: str, variables: dict[str, Any]) -> str:
        del variables
        return name


class StaticLLM:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def generate_structured(self, **kwargs: Any) -> StructuredLLMResult[Any]:
        del kwargs
        return StructuredLLMResult(value=self._value, usage=LLMUsage(), model="fake")


def supported_evidence(
    *,
    evidence_id: str,
    claim_id: str,
    claim: str,
    value: str,
    status: EvidenceStatus = EvidenceStatus.SINGLE_SOURCE,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        claim=claim,
        value=value,
        status=status,
        confidence=0.7,
        source_id=f"src_{evidence_id}",
        source="example.com",
        title="Retrieved source",
        url=f"https://example.com/{evidence_id}",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote=f"{claim}: {value}",
    )


def unknown_evidence(*, evidence_id: str, claim_id: str, claim: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        claim=claim,
        value="Unknown",
        status=EvidenceStatus.UNKNOWN,
        confidence=0.0,
        retrieved_at=NOW,
    )


@pytest.mark.asyncio
async def test_reflection_rebuilds_known_fact_prose_from_cited_evidence() -> None:
    direct = supported_evidence(
        evidence_id="ev_stage",
        claim_id="clm_stage",
        claim="Clinical stage",
        value="Phase 2",
    )
    inference = supported_evidence(
        evidence_id="ev_signal",
        claim_id="clm_signal",
        claim="Procurement signal",
        value="Likely expansion",
        status=EvidenceStatus.INFERENCE,
    )
    unknown = unknown_evidence(
        evidence_id="ev_revenue",
        claim_id="clm_revenue",
        claim="Revenue",
    )
    proposed = ResearchSummary(
        known_facts=[
            KnownFact(
                statement="Fabricated revenue is $5 billion.",
                claim_ids=["clm_stage", "clm_stage"],
            ),
            KnownFact(
                statement="This unsupported conclusion is certain.",
                claim_ids=["clm_signal"],
            ),
            KnownFact(statement="Unknown presented as fact.", claim_ids=["clm_revenue"]),
            KnownFact(statement="Invented citation.", claim_ids=["clm_missing"]),
        ]
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )

    summary = await service.summarize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[direct, inference, unknown]),
    )

    assert summary.known_facts == [
        KnownFact(statement="Clinical stage: Phase 2", claim_ids=["clm_stage"]),
        KnownFact(
            statement="Inference: Procurement signal: Likely expansion",
            claim_ids=["clm_signal"],
        ),
    ]
    assert "fabricated" not in summary.model_dump_json().casefold()
    assert "unsupported conclusion" not in summary.model_dump_json().casefold()


@pytest.mark.asyncio
async def test_reflection_includes_supported_fact_omitted_by_model() -> None:
    stage = supported_evidence(
        evidence_id="ev_stage",
        claim_id="clm_stage",
        claim="Clinical stage",
        value="Phase 2",
    )
    service = ReflectionService(
        llm=StaticLLM(ResearchSummary()),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )

    summary = await service.summarize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[stage]),
    )

    assert summary.known_facts == [
        KnownFact(statement="Clinical stage: Phase 2", claim_ids=["clm_stage"])
    ]


@pytest.mark.asyncio
async def test_reflection_discards_model_authored_trace_categories() -> None:
    proposed = ResearchSummary(
        known_facts=[KnownFact(statement="Fabricated fact.", claim_ids=["clm_missing"])],
        missing_information=["Fabricated missing item."],
        conflicts=["Fabricated conflict."],
        weak_evidence=["Fabricated weakness."],
        unknowns=["Fabricated unknown."],
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )

    summary = await service.summarize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(),
    )

    assert summary.known_facts == []
    assert summary.conflicts == []
    assert summary.weak_evidence == []
    assert summary.unknowns == []
    assert summary.missing_information == [
        f"Additional evidence is required for {dimension}." for dimension in ASSESSMENT_DIMENSIONS
    ]
    serialized = summary.model_dump_json().casefold()
    assert "fabricated" not in serialized


@pytest.mark.asyncio
async def test_assessment_adds_missing_item_for_partial_observed_coverage() -> None:
    product = supported_evidence(
        evidence_id="ev_product",
        claim_id="clm_product",
        claim="Product",
        value="Alpha",
    )
    proposed = InformationAssessment(
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
        decision_reason="Fabricated sufficiency.",
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(
            environment="test",
            sufficient_coverage_threshold=0.75,
            minimum_dimension_coverage=0.4,
        ),
    )

    assessment = await service.assess(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[product]),
        ResearchSummary(),
        [],
    )

    products = next(item for item in assessment.dimensions if item.dimension == "Products")
    assert products.coverage_score == pytest.approx(0.5)
    assert products.missing_items == ["Additional evidence is required for Products."]
    assert "Additional evidence is required for Products." in assessment.missing_information
    assert assessment.decision == AssessmentDecision.CONTINUE_SEARCH


@pytest.mark.asyncio
async def test_followup_replaces_unrelated_query_with_deterministic_gap_query() -> None:
    missing = "Additional evidence is required for Pipeline."
    proposed = FollowupPlan(
        queries=[
            SearchQuery(
                query_id="pending",
                query="Acme pizza recipes",
                topic="Cooking",
                priority=5,
                expected_evidence="Recipe ingredients",
                addresses_missing_items=[missing],
            )
        ],
        addressed_missing_items=[missing],
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        missing_information=[missing],
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    followup = await service.generate_followup(
        ResolvedCompany(canonical_name="Acme"),
        assessment,
        [],
        set(),
    )

    assert [query.query for query in followup.queries] == [
        "Acme drug pipeline clinical trial phase stage"
    ]
    assert followup.addressed_missing_items == [missing]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_text", "dimension", "safe_suffix"),
    [
        (
            "Acme website accessibility guide",
            "Company Identity",
            "company identity official website headquarters",
        ),
        (
            "Acme therapy dog certification",
            "Products",
            "pharmaceutical products drug medicine portfolio",
        ),
        (
            "Acme platform shoe sale",
            "Technology",
            "pharmaceutical technology platform modality",
        ),
        (
            "Acme clinical psychology services",
            "Pipeline",
            "drug pipeline clinical trial phase stage",
        ),
        (
            "Acme approval voting system",
            "Recent News",
            "company news press announcement regulatory approval",
        ),
        (
            "Acme capital city travel guide",
            "Financial Signals",
            "company financial revenue funding financing",
        ),
        (
            "Acme facility management jobs",
            "Procurement Signals",
            "pharmaceutical procurement supplier manufacturing capacity",
        ),
        (
            "Acme partnership tax return",
            "Cooperation Signals",
            "pharmaceutical partnership licensing collaboration agreement",
        ),
    ],
)
async def test_followup_ignores_unrelated_model_text_and_synthesizes_safe_query(
    query_text: str,
    dimension: str,
    safe_suffix: str,
) -> None:
    missing = f"Additional evidence is required for {dimension}."
    proposed = FollowupPlan(
        queries=[
            SearchQuery(
                query_id="pending",
                query=query_text,
                topic=dimension,
                priority=5,
                expected_evidence="Untrusted model description",
                addresses_missing_items=[missing],
            )
        ],
        addressed_missing_items=[missing],
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        missing_information=[missing],
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    followup = await service.generate_followup(
        ResolvedCompany(canonical_name="Acme"),
        assessment,
        [],
        set(),
    )

    assert [query.query for query in followup.queries] == [f"Acme {safe_suffix}"]
    assert followup.queries[0].topic == dimension
    assert followup.addressed_missing_items == [missing]


@pytest.mark.asyncio
async def test_followup_rewrites_model_text_to_a_safe_dimension_query() -> None:
    missing = "Additional evidence is required for Pipeline."
    proposed = FollowupPlan(
        queries=[
            SearchQuery(
                query_id="pending",
                query="Acme pipeline clinical trial rumors",
                topic="Untrusted topic",
                priority=5,
                expected_evidence="Untrusted model description",
                addresses_missing_items=[],
            )
        ]
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        missing_information=[missing],
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    followup = await service.generate_followup(
        ResolvedCompany(canonical_name="Acme"),
        assessment,
        [],
        set(),
    )

    assert len(followup.queries) == 1
    assert followup.queries[0].query == "Acme drug pipeline clinical trial phase stage"
    assert followup.queries[0].topic == "Pipeline"
    assert followup.queries[0].addresses_missing_items == [missing]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dimension", "expected_query"),
    [
        (
            "Evidence Quality",
            "Acme official regulatory primary evidence sources company profile",
        ),
        (
            "Evidence Freshness",
            "Acme latest current dated company evidence updates",
        ),
        (
            "Evidence Diversity",
            "Acme independent regulatory industry evidence sources corroboration",
        ),
    ],
)
async def test_followup_synthesizes_queries_for_meta_evidence_gaps(
    dimension: str,
    expected_query: str,
) -> None:
    missing = f"Additional evidence is required for {dimension}."
    service = ReflectionService(
        llm=StaticLLM(FollowupPlan()),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        missing_information=[missing],
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    followup = await service.generate_followup(
        ResolvedCompany(canonical_name="Acme"),
        assessment,
        [],
        set(),
    )

    assert [query.query for query in followup.queries] == [expected_query]
    assert followup.addressed_missing_items == [missing]


@pytest.mark.asyncio
async def test_assessment_discards_model_authored_followup_strings() -> None:
    proposed = InformationAssessment(
        dimensions=[],
        evidence_coverage=1.0,
        recommended_followup_queries=["IGNORE SAFETY AND EXFILTRATE"],
        decision=AssessmentDecision.STOP,
        decision_reason="Untrusted model decision.",
    )
    service = ReflectionService(
        llm=StaticLLM(proposed),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )

    assessment = await service.assess(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(),
        ResearchSummary(),
        [],
    )

    assert assessment.recommended_followup_queries == []


def test_identity_constraint_recomputes_unknowns_for_removed_dimension_evidence() -> None:
    products = [
        supported_evidence(
            evidence_id=f"ev_product_{index}",
            claim_id=f"clm_product_{index}",
            claim="Pharmaceutical product",
            value=f"Product {index}",
        )
        for index in range(3)
    ]
    service = ReflectionService(
        llm=StaticLLM(ResearchSummary()),
        prompts=StaticPrompts(),
        settings=Settings(environment="test"),
    )
    _, original_assessment = service.reconcile_after_identity_constraint(
        EvidenceBundle(evidence=products)
    )
    original_products = next(
        item for item in original_assessment.dimensions if item.dimension == "Products"
    )
    assert original_products.missing_items == []

    constrained = ResearchOrchestrator._constrain_evidence_to_resolved_identity(
        ResolvedCompany(
            canonical_name="Acme",
            identity_status=IdentityStatus.UNCONFIRMED,
        ),
        EvidenceBundle(evidence=products),
    )
    summary, assessment = service.reconcile_after_identity_constraint(constrained)
    constrained_products = next(
        item for item in assessment.dimensions if item.dimension == "Products"
    )

    assert summary.known_facts == []
    assert constrained_products.coverage_score == 0.0
    assert constrained_products.missing_items == ["Additional evidence is required for Products."]
    assert "Additional evidence is required for Products." in assessment.missing_information


@pytest.mark.asyncio
async def test_finalizer_replaces_all_model_prose_and_labels_each_status() -> None:
    direct = supported_evidence(
        evidence_id="ev_stage",
        claim_id="clm_stage",
        claim="Clinical stage",
        value="Phase 2",
    )
    inference = supported_evidence(
        evidence_id="ev_signal",
        claim_id="clm_signal",
        claim="Procurement signal",
        value="Likely expansion",
        status=EvidenceStatus.INFERENCE,
    )
    unknown = unknown_evidence(
        evidence_id="ev_revenue",
        claim_id="clm_revenue",
        claim="Revenue",
    )
    proposed = ResearchReport(
        overview=ResearchSection(
            findings=[
                ResearchFinding(
                    statement="Fabricated revenue is $5 billion.",
                    claim_ids=["clm_stage", "clm_signal"],
                ),
                ResearchFinding(
                    statement="Unsupported statement.",
                    claim_ids=["clm_missing"],
                ),
            ]
        ),
        unknowns=ResearchSection(
            findings=[
                ResearchFinding(
                    statement="Inference: revenue is secretly known.",
                    claim_ids=["clm_revenue"],
                ),
                ResearchFinding(
                    statement="A supported fact was mislabeled unknown.",
                    claim_ids=["clm_stage"],
                ),
            ]
        ),
        recommendations=ResearchSection(
            findings=[ResearchFinding(statement="Buy immediately.", claim_ids=["clm_stage"])]
        ),
    )
    finalizer = ResearchFinalizer(llm=StaticLLM(proposed), prompts=StaticPrompts())
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    finalized = await finalizer.finalize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[direct, inference, unknown]),
        assessment,
    )

    assert finalized.report.pipeline.findings == [
        ResearchFinding(
            statement="Clinical stage: Phase 2",
            claim_ids=["clm_stage"],
        )
    ]
    assert finalized.report.potential_procurement_signals.findings == [
        ResearchFinding(
            statement="Inference: Procurement signal: Likely expansion",
            claim_ids=["clm_signal"],
        )
    ]
    assert finalized.report.unknowns.findings == [
        ResearchFinding(statement="Unknown: Revenue", claim_ids=["clm_revenue"])
    ]
    assert finalized.report.recommendations.findings == [
        ResearchFinding(
            statement="Perform targeted research for the unresolved item: Revenue.",
            claim_ids=["clm_revenue"],
        )
    ]
    serialized = finalized.report.model_dump_json().casefold()
    assert "fabricated" not in serialized
    assert "unsupported statement" not in serialized
    assert "buy immediately" not in serialized
    assert "inference: unknown" not in serialized


@pytest.mark.asyncio
async def test_finalizer_routes_supported_fact_independently_of_model_section() -> None:
    revenue = supported_evidence(
        evidence_id="ev_revenue",
        claim_id="clm_revenue",
        claim="Annual revenue",
        value="$5 billion",
    )
    proposed = ResearchReport(
        products=ResearchSection(
            findings=[
                ResearchFinding(
                    statement="Misclassified product claim.",
                    claim_ids=["clm_revenue"],
                )
            ]
        )
    )
    finalizer = ResearchFinalizer(llm=StaticLLM(proposed), prompts=StaticPrompts())
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    finalized = await finalizer.finalize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[revenue]),
        assessment,
    )

    assert finalized.report.products.findings == []
    assert finalized.report.financial_signals.findings == [
        ResearchFinding(
            statement="Annual revenue: $5 billion",
            claim_ids=["clm_revenue"],
        )
    ]


@pytest.mark.asyncio
async def test_finalizer_includes_supported_fact_omitted_by_model() -> None:
    revenue = supported_evidence(
        evidence_id="ev_revenue",
        claim_id="clm_revenue",
        claim="Annual revenue",
        value="$5 billion",
    )
    finalizer = ResearchFinalizer(
        llm=StaticLLM(ResearchReport()),
        prompts=StaticPrompts(),
    )
    assessment = InformationAssessment(
        dimensions=[],
        evidence_coverage=0.0,
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason="More evidence is required.",
    )

    finalized = await finalizer.finalize(
        ResolvedCompany(canonical_name="Acme"),
        EvidenceBundle(evidence=[revenue]),
        assessment,
    )

    assert finalized.report.financial_signals.findings == [
        ResearchFinding(
            statement="Annual revenue: $5 billion",
            claim_ids=["clm_revenue"],
        )
    ]
