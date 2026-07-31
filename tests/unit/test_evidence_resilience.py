import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.enums import (
    DimensionStatus,
    EvidenceStatus,
    ExecutionStatus,
    ExtractionMethod,
    GapReason,
    SourceType,
    StopReason,
    TransitionOutcome,
)
from app.core.exceptions import StructuredOutputError
from app.core.protocols import LLMUsage, StructuredLLMResult
from app.evidence.processor import ClaimExtractionResponse, EvidenceProcessor
from app.evidence.registry import SourceRegistry
from app.llm.errors import LLMProviderResponseError
from app.llm.parsing import parse_structured_output
from app.schemas.evidence import Evidence, EvidenceBundle
from app.schemas.reflection import DimensionAssessment
from app.schemas.research import ResearchGap, ResearchGapSection, ResearchReport
from app.schemas.search import QueryExecution, SearchBatch, SearchResult, SearchStatistics

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class Prompts:
    def render(self, name: str, variables: Mapping[str, Any]) -> str:
        del variables
        return name


class RawLLM:
    def __init__(self, response: ClaimExtractionResponse | Exception) -> None:
        self.response = response
        self.calls = 0

    async def generate_structured(self, **_: Any) -> StructuredLLMResult[Any]:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return StructuredLLMResult(
            value=self.response,
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="fixture",
        )


def pfizer_batch() -> SearchBatch:
    results = [
        SearchResult(
            title="Pfizer clinical update",
            url="https://www.pfizer.com/news/clinical-update",
            text="Pfizer clinical stage is Phase 2.",
            source_type=SourceType.OFFICIAL,
        ),
        SearchResult(
            title="Pfizer regulatory record",
            url="https://www.fda.gov/drugs/pfizer-record",
            text="Pfizer revenue was $5 billion.",
            source_type=SourceType.REGULATORY,
        ),
    ]
    execution = QueryExecution(
        query_id="qry_pfizer_1",
        query="Pfizer company research",
        started_at=NOW,
        completed_at=NOW,
        duration_ms=1,
        attempts=1,
        results=results,
    )
    return SearchBatch(
        executions=[execution],
        statistics=SearchStatistics(
            query_count=1,
            successful_queries=1,
            failed_queries=0,
            cache_hits=0,
            total_results=2,
            total_duration_ms=1,
        ),
    )


def pfizer_sec_pipeline_batch() -> SearchBatch:
    sec_text = "\n".join(
        [
            "Company name: PFIZER INC.",
            "Delaware 13-5315170 (State or other jurisdiction of incorporation or organization)",
            "66 Hudson Boulevard East, New York, New York 10001-2192 "
            "(Address of principal executive offices) (zip code)",
            "Common Stock, $0.05 par value PFE New York Stock Exchange",
        ]
    )
    pipeline_text = "\n".join(
        [
            "Company name: Pfizer.",
            "The Pfizer pipeline includes 102 programs across Phase 1, Phase 2, and Phase 3.",
        ]
    )
    results = [
        SearchResult(
            title="PFIZER INC.",
            url="https://www.sec.gov/Archives/edgar/data/78003/pfe-2025.htm",
            text=sec_text,
            source_type=SourceType.REGULATORY,
        ),
        SearchResult(
            title="Pfizer Pipeline",
            url="https://cdn.pfizer.com/pipeline/pipeline-update.pdf",
            text=pipeline_text,
            source_type=SourceType.OTHER,
        ),
    ]
    execution = QueryExecution(
        query_id="qry_pfizer_sec_pipeline",
        query="Pfizer SEC identity and pipeline",
        started_at=NOW,
        completed_at=NOW,
        duration_ms=1,
        attempts=1,
        results=results,
    )
    return SearchBatch(
        executions=[execution],
        statistics=SearchStatistics(
            query_count=1,
            successful_queries=1,
            failed_queries=0,
            cache_hits=0,
            total_results=2,
            total_duration_ms=1,
        ),
    )


@pytest.mark.asyncio
async def test_pfizer_sec_table_and_pipeline_claims_survive_validation() -> None:
    batch = pfizer_sec_pipeline_batch()
    preview_registry = SourceRegistry(clock=FixedClock())
    preview_registry.register_batch(batch)
    source_ids_by_title = {source.title: source.source_id for source in preview_registry.sources}
    sec_id = source_ids_by_title["PFIZER INC."]
    pipeline_id = source_ids_by_title["Pfizer Pipeline"]
    response = ClaimExtractionResponse(
        claims=[
            {
                "claim": "Company name",
                "value": "PFIZER INC.",
                "source_ids": [sec_id],
                "supporting_quotes": ["Company name: PFIZER INC."],
                "confidence": 0.99,
            },
            {
                "claim": "Pfizer Inc. is incorporated in Delaware.",
                "value": "Delaware",
                "source_ids": [sec_id],
                "supporting_quotes": [
                    "Delaware 13-5315170 "
                    "(State or other jurisdiction of incorporation or organization)"
                ],
                "confidence": 0.99,
            },
            {
                "claim": "Pfizer headquarters are located at "
                "66 Hudson Boulevard East, New York, New York 10001-2192.",
                "value": "66 Hudson Boulevard East, New York, New York 10001-2192",
                "source_ids": [sec_id],
                "supporting_quotes": [
                    "66 Hudson Boulevard East, New York, New York 10001-2192 "
                    "(Address of principal executive offices) (zip code)"
                ],
                "confidence": 0.99,
            },
            {
                "claim": "Pfizer common stock trades under ticker PFE.",
                "value": "PFE",
                "source_ids": [sec_id],
                "supporting_quotes": ["Common Stock, $0.05 par value PFE New York Stock Exchange"],
                "confidence": 0.99,
            },
            {
                "claim": "Company name",
                "value": "Pfizer",
                "source_ids": [pipeline_id],
                "supporting_quotes": ["Company name: Pfizer."],
                "confidence": 0.99,
            },
            {
                "claim": "Pfizer has 102 programs in its pipeline.",
                "value": 102,
                "source_ids": [pipeline_id],
                "supporting_quotes": [
                    "The Pfizer pipeline includes 102 programs across Phase 1, "
                    "Phase 2, and Phase 3."
                ],
                "confidence": 0.95,
            },
        ]
    )
    processor = EvidenceProcessor(
        llm_client=RawLLM(response),
        prompt_repository=Prompts(),
        registry=SourceRegistry(clock=FixedClock()),
        subject_identifiers=("Pfizer", "Pfizer Inc."),
        evidence_limit=20,
    )

    result = await processor.process(
        batch,
        company_context={"canonical_name": "Pfizer Inc."},
    )

    assert result.bundle.rejected_claim_count == 0
    assert result.bundle.supported_claim_count == 6
    assert result.bundle.verified_fact_count == 6
    assert result.bundle.conflicts == []
    known_source_ids = {source.source_id for source in result.bundle.sources}
    assert all(set(item.source_ids).issubset(known_source_ids) for item in result.bundle.evidence)


@pytest.mark.asyncio
async def test_known_legacy_claim_fields_are_normalized_but_unknown_extra_is_specific() -> None:
    batch = pfizer_batch()
    preview_registry = SourceRegistry(clock=FixedClock())
    preview_registry.register_batch(batch)
    source_id = next(
        source.source_id
        for source in preview_registry.sources
        if source.title == "Pfizer clinical update"
    )
    response = ClaimExtractionResponse(
        claims=[
            {
                "claim": "Clinical stage",
                "value": "Phase 2",
                "source_id": source_id,
                "supporting_quote": "Pfizer clinical stage is Phase 2.",
                "evidence_id": "ignored_draft_id",
                "confidence": 0.9,
            },
            {
                "claim": "Revenue",
                "value": "$5 billion",
                "source_ids": [source_id],
                "supporting_quotes": ["Pfizer clinical stage is Phase 2."],
                "unexpected_field": "must not be silently ignored",
                "confidence": 0.9,
            },
        ]
    )
    processor = EvidenceProcessor(
        llm_client=RawLLM(response),
        prompt_repository=Prompts(),
        registry=SourceRegistry(clock=FixedClock()),
        subject_identifiers=("Pfizer", "Pfizer Inc."),
        evidence_limit=20,
    )

    result = await processor.process(batch, company_context={"canonical_name": "Pfizer Inc."})

    assert result.bundle.supported_claim_count == 1
    assert result.bundle.rejected_claim_count == 1
    assert "unexpected_field: extra_forbidden" in result.bundle.rejected_claims[0].reason
    assert any(
        error.code == "pydantic_validation_error" for error in result.bundle.processing_errors
    )


@pytest.mark.asyncio
async def test_pfizer_url_claim_is_rejected_without_collapsing_batch() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registered = registry.register_batch(pfizer_batch())
    source_one, source_two = registered.new_source_ids
    registry = SourceRegistry(clock=FixedClock())
    response = ClaimExtractionResponse(
        claims=[
            {
                "claim": "Clinical stage",
                "value": "Phase 2",
                "source_ids": [source_one],
                "supporting_quotes": [
                    {"source_id": source_one, "quote": "Pfizer clinical stage is Phase 2."}
                ],
                "confidence": 0.9,
            },
            {
                "claim": "https://www.pfizer.com/about",
                "value": "Pfizer company information",
                "source_ids": [source_one],
                "supporting_quotes": [],
                "confidence": 0.5,
            },
            {
                "claim": "Revenue",
                "value": "$5 billion",
                "source_ids": [source_two],
                "supporting_quotes": [
                    {"source_id": source_two, "quote": "Pfizer revenue was $5 billion."}
                ],
                "confidence": 0.9,
            },
        ]
    )
    processor = EvidenceProcessor(
        llm_client=RawLLM(response),
        prompt_repository=Prompts(),
        registry=registry,
        subject_identifiers=("Pfizer", "Pfizer Inc."),
        evidence_limit=20,
    )

    result = await processor.process(
        pfizer_batch(), company_context={"canonical_name": "Pfizer Inc."}
    )

    assert result.new_evidence_count == 2
    assert result.bundle.supported_claim_count == len(result.bundle.evidence) == 2
    assert result.bundle.rejected_claim_count == len(result.bundle.rejected_claims) == 1
    assert result.bundle.rejected_claims[0].index == 1
    assert result.bundle.rejected_claims[0].reason == "prohibited_url_in_claim"
    assert result.transition_outcome == TransitionOutcome.SUCCEEDED_WITH_WARNINGS
    assert len(result.bundle.sources) == 2
    assert all(item.source_ids for item in result.bundle.evidence)
    assert any(error.code == "prohibited_url_in_claim" for error in result.bundle.processing_errors)
    assert all(item.status != EvidenceStatus.UNKNOWN for item in result.bundle.evidence)
    assert all(
        item.claim_id in source.supported_claim_ids
        for item in result.bundle.evidence
        for source in result.bundle.sources
        if source.source_id in item.source_ids
    )


def test_extraction_envelope_defers_url_validation_to_individual_claims() -> None:
    parsed = parse_structured_output(
        json.dumps(
            {
                "claims": [
                    {
                        "claim": "https://www.pfizer.com/about",
                        "value": None,
                        "source_ids": [],
                        "supporting_quotes": [],
                        "is_inference": False,
                        "derived_from_claim_ids": [],
                        "confidence": 0.0,
                    }
                ]
            }
        ),
        ClaimExtractionResponse,
    )
    assert len(parsed.claims) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        StructuredOutputError("invalid fixture output"),
        LLMProviderResponseError("provider output limit exceeded"),
    ],
)
async def test_total_structured_failure_repairs_once_then_uses_marked_fallback(
    failure: Exception,
) -> None:
    llm = RawLLM(failure)
    processor = EvidenceProcessor(
        llm_client=llm,
        prompt_repository=Prompts(),
        registry=SourceRegistry(clock=FixedClock()),
        subject_identifiers=("Pfizer",),
        evidence_limit=20,
    )

    result = await processor.process(
        pfizer_batch(), company_context={"canonical_name": "Pfizer Inc."}
    )

    assert llm.calls == 2
    assert result.transition_outcome == TransitionOutcome.PARTIAL_FAILURE
    assert result.bundle.sources
    assert result.bundle.evidence
    assert all(
        item.extraction_method == ExtractionMethod.DETERMINISTIC_FALLBACK
        for item in result.bundle.evidence
    )
    assert any(error.code == "extraction_unavailable" for error in result.bundle.processing_errors)


def test_evidence_supports_multiple_sources_without_serialized_metadata() -> None:
    evidence = Evidence(
        evidence_id="ev_1",
        claim_id="clm_1",
        claim="Pfizer Inc. is incorporated in Delaware.",
        value={"jurisdiction": "Delaware"},
        status=EvidenceStatus.VERIFIED_FACT,
        confidence=0.98,
        source_ids=["src_1", "src_2"],
        supporting_quotes=["quote one", "quote two"],
    )
    payload = evidence.model_dump(mode="json")
    assert payload["source_ids"] == ["src_1", "src_2"]
    assert not {"url", "title", "publisher", "source_type", "published_at"}.intersection(payload)
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="ev_bad",
            claim_id="clm_bad",
            claim="Unsupported verified claim",
            value=None,
            status=EvidenceStatus.VERIFIED_FACT,
            confidence=0.8,
        )


def test_unknown_gap_has_no_claim_id_and_does_not_count_as_evidence() -> None:
    gap = ResearchGap(
        gap_id="gap_1",
        dimension="Products",
        description="No supported product information was retained.",
        reason=GapReason.INSUFFICIENT_EVIDENCE,
    )
    report = ResearchReport(unknowns=ResearchGapSection(gaps=[gap]))
    bundle = EvidenceBundle(research_gap_count=1)
    assert "claim_id" not in gap.model_dump()
    assert report.unknowns.gaps == [gap]
    assert bundle.supported_claim_count == 0


def test_assessment_statuses_and_precise_budget_reason_contract() -> None:
    not_searched = DimensionAssessment(
        dimension="Products",
        status=DimensionStatus.NOT_SEARCHED,
        coverage_score=0,
        confidence=0,
    )
    searched_empty = not_searched.model_copy(update={"status": DimensionStatus.SEARCHED_NO_RESULTS})
    assert not_searched.status != searched_empty.status
    with pytest.raises(ValueError):
        StopReason("budget_reached")
    assert StopReason.QUERY_BUDGET_REACHED.value == "query_budget_reached"
    assert ExecutionStatus.PARTIAL_FAILURE.value == "partial_failure"
