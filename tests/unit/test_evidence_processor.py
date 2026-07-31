import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.enums import EvidenceStatus, SourceType
from app.core.protocols import LLMUsage, StructuredLLMResult
from app.evidence.processor import ClaimExtractionResponse, EvidenceProcessor
from app.evidence.registry import SourceRegistry
from app.schemas.evidence import ExtractedClaim, SupportingQuote
from app.schemas.search import (
    QueryExecution,
    SearchBatch,
    SearchResult,
    SearchStatistics,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class FakePrompts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def render(self, name: str, variables: Mapping[str, Any]) -> str:
        self.calls.append((name, variables))
        return "Extract only claims supported by allowed source IDs."


ResponseFactory = Callable[[dict[str, Any], int], ClaimExtractionResponse]


class FakeLLM:
    def __init__(self, response_factory: ResponseFactory) -> None:
        self._response_factory = response_factory
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        cache_namespace: str,
    ) -> StructuredLLMResult[Any]:
        payload = json.loads(user_prompt)
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "payload": payload,
                "response_model": response_model,
                "cache_namespace": cache_namespace,
            }
        )
        response = self._response_factory(payload, len(self.calls))
        return StructuredLLMResult(
            value=response,
            usage=LLMUsage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.01),
            model="fake",
        )


def make_result(
    url: str,
    text: str,
    *,
    title: str = "Source title",
    source_type: SourceType = SourceType.NEWS,
    bind_subject: bool = True,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,
        text=(text if not bind_subject or "acme" in text.casefold() else f"Acme Pharma: {text}"),
        source_type=source_type,
    )


def make_batch(*query_results: tuple[str, list[SearchResult]]) -> SearchBatch:
    executions = [
        QueryExecution(
            query_id=query_id,
            query=query_id,
            started_at=NOW,
            completed_at=NOW,
            duration_ms=0,
            attempts=1,
            results=results,
        )
        for query_id, results in query_results
    ]
    return SearchBatch(
        executions=executions,
        statistics=SearchStatistics(
            query_count=len(executions),
            successful_queries=len(executions),
            failed_queries=0,
            cache_hits=0,
            total_results=sum(len(results) for _, results in query_results),
            total_duration_ms=0,
        ),
    )


def source_ids(payload: dict[str, Any]) -> list[str]:
    return [source["source_id"] for source in payload["sources"]]


def supporting_quotes(payload: dict[str, Any]) -> list[SupportingQuote]:
    return [
        SupportingQuote(source_id=source["source_id"], quote=source["content"])
        for source in payload["sources"]
    ]


def make_processor(
    response_factory: ResponseFactory,
    *,
    evidence_limit: int = 100,
) -> tuple[EvidenceProcessor, FakeLLM, SourceRegistry, FakePrompts]:
    llm = FakeLLM(response_factory)
    registry = SourceRegistry(clock=FixedClock())
    prompts = FakePrompts()
    processor = EvidenceProcessor(
        llm_client=llm,
        prompt_repository=prompts,
        registry=registry,
        subject_identifiers=("Acme Pharma",),
        evidence_limit=evidence_limit,
    )
    return processor, llm, registry, prompts


def supporting_claim(
    payload: dict[str, Any],
    _call_number: int,
    *,
    claim: str = "Clinical stage",
    value: Any = "Phase 2",
    is_inference: bool = False,
    derived_from_claim_ids: list[str] | None = None,
    confidence: float = 0.9,
) -> ClaimExtractionResponse:
    return ClaimExtractionResponse(
        claims=[
            ExtractedClaim(
                claim=claim,
                value=value,
                source_ids=source_ids(payload),
                supporting_quotes=supporting_quotes(payload),
                is_inference=is_inference,
                derived_from_claim_ids=derived_from_claim_ids or [],
                confidence=confidence,
            )
        ]
    )


@pytest.mark.asyncio
async def test_evidence_processor_attaches_lineage_only_from_registry() -> None:
    processor, llm, registry, prompts = make_processor(supporting_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://company.example/release?utm_source=email",
                        "The program entered Phase 2.",
                        title="Company release",
                        source_type=SourceType.OFFICIAL,
                    )
                ],
            )
        ),
        company_context={"canonical_name": "Acme"},
    )

    assert result.usage.total_tokens == 15
    assert result.new_source_count == 1
    assert result.new_evidence_count == 1
    assert len(llm.calls) == 1
    assert llm.calls[0]["response_model"] is ClaimExtractionResponse
    assert prompts.calls[0][0] == "extractor"
    assert set(prompts.calls[0][1]) == {
        "existing_claims",
        "resolved_company",
        "source_documents",
    }
    evidence = result.bundle.evidence[0]
    source = registry.get(evidence.source_id or "")
    assert evidence.status == EvidenceStatus.VERIFIED_FACT
    assert evidence.title == source.title == "Company release"
    assert evidence.url == source.url
    assert evidence.source_type == source.source_type == SourceType.OFFICIAL
    assert evidence.supporting_quote == "Acme Pharma: The program entered Phase 2."
    assert str(evidence.url) == "https://company.example/release"


@pytest.mark.asyncio
async def test_evidence_processor_rejects_non_allowlisted_source_id_without_echoing_it() -> None:
    def fabricated_source(_payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=["src_fabricated"],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(fabricated_source)
    result = await processor.process(
        make_batch(("q1", [make_result("https://real.example/page", "Unrelated text")]))
    )

    assert result.bundle.rejected_claim_count == 1
    assert result.new_evidence_count == 0
    assert result.bundle.evidence == []


def test_evidence_extraction_schema_forbids_llm_supplied_url_or_title() -> None:
    envelope = ClaimExtractionResponse.model_validate(
        {
            "claims": [
                {
                    "claim": "Revenue",
                    "value": "$5 billion",
                    "source_ids": ["src_1"],
                    "is_inference": False,
                    "confidence": 0.9,
                    "url": "https://fabricated.example",
                    "title": "Fabricated source",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        ExtractedClaim.model_validate(envelope.claims[0])


@pytest.mark.asyncio
async def test_url_valued_claim_is_rejected_even_when_present_in_page_text() -> None:
    def url_claim(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Website",
                    value="https://fabricated.example",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(url_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://provider.example/page",
                        "Acme Pharma website is https://fabricated.example",
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_evidence_verified_requires_two_distinct_reliable_sources() -> None:
    processor, _, _, _ = make_processor(supporting_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://regulator.example/trial",
                        "Acme Pharma trial is Phase 2 according to regulator.",
                        source_type=SourceType.REGULATORY,
                    ),
                    make_result(
                        "https://company.example/release",
                        "Acme Pharma trial is Phase 2 according to company.",
                        source_type=SourceType.OFFICIAL,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert {item.status for item in result.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}
    assert len(result.bundle.evidence[0].source_ids) == 2


@pytest.mark.asyncio
async def test_evidence_mirrored_content_cannot_inflate_verified_status() -> None:
    processor, llm, _, _ = make_processor(supporting_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://wire.example/story",
                        "Acme Pharma trial is Phase 2.",
                        source_type=SourceType.NEWS,
                    ),
                    make_result(
                        "https://mirror.example/story",
                        " acme   pharma trial is phase 2. ",
                        source_type=SourceType.NEWS,
                    ),
                ],
            )
        )
    )

    assert result.duplicate_source_count == 1
    assert len(llm.calls[0]["payload"]["sources"]) == 1
    assert len(result.bundle.evidence) == 1
    assert result.bundle.evidence[0].status == EvidenceStatus.SINGLE_SOURCE


@pytest.mark.asyncio
async def test_republished_quote_cannot_inflate_verified_status() -> None:
    quote = "Acme Pharma revenue was $5 billion."

    def republished(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=source_ids(payload),
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=quote)
                        for source in payload["sources"]
                    ],
                    confidence=0.9,
                )
            ]
        )

    processor, _, _, _ = make_processor(republished)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://official.example/release",
                        quote,
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://news.example/copy",
                        f"Republished press release: {quote}",
                        source_type=SourceType.NEWS,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert {item.status for item in result.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}


@pytest.mark.asyncio
async def test_republished_quote_with_long_suffix_cannot_inflate_verified_status() -> None:
    fact = "Acme Pharma revenue was $5 billion"
    republished = (
        f"{fact}, republished press release copy distributed by the Acme Pharma "
        "media relations desk for syndication"
    )

    def copied(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(
                            source_id=source["source_id"],
                            quote=source["content"],
                        )
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(copied)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://official.example/release",
                        fact,
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://news.example/copy",
                        republished,
                        source_type=SourceType.NEWS,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert {item.status for item in result.bundle.evidence} == {EvidenceStatus.SINGLE_SOURCE}


@pytest.mark.asyncio
async def test_evidence_late_bridge_does_not_rewrite_verified_lineage() -> None:
    processor, llm, registry, _ = make_processor(supporting_claim)
    first = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://a.example/page",
                        "Acme Pharma trial confirms Phase 2 in first source.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://b.example/page",
                        "Acme Pharma trial confirms Phase 2 in second source.",
                        source_type=SourceType.REGULATORY,
                    ),
                ],
            )
        )
    )
    assert {item.status for item in first.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}

    second = await processor.process(
        make_batch(
            (
                "q2",
                [
                    make_result(
                        "https://a.example/page",
                        "Acme Pharma trial confirms Phase 2 in second source.",
                    )
                ],
            )
        )
    )

    assert len(registry) == 2
    assert len(llm.calls) == 1
    assert second.new_evidence_count == 0
    assert len(second.bundle.evidence) == 1
    assert {item.status for item in second.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}
    evidence = second.bundle.evidence[0]
    assert all(
        quote.casefold() in registry.get(source_id).content.casefold()
        for source_id, quote in zip(evidence.source_ids, evidence.supporting_quotes, strict=True)
    )


@pytest.mark.asyncio
async def test_evidence_social_sources_do_not_upgrade_to_verified() -> None:
    processor, _, _, _ = make_processor(supporting_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://social-one.com/one",
                        "Acme Pharma trial says Phase 2 on social one.",
                        source_type=SourceType.SOCIAL,
                    ),
                    make_result(
                        "https://social-two.net/two",
                        "Acme Pharma trial says Phase 2 on social two.",
                        source_type=SourceType.SOCIAL,
                    ),
                ],
            )
        )
    )

    assert result.bundle.evidence == []


@pytest.mark.asyncio
async def test_single_untrusted_source_remains_unknown() -> None:
    processor, _, _, _ = make_processor(supporting_claim)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://social-one.com/one",
                        "The trial is Phase 2.",
                        source_type=SourceType.SOCIAL,
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []


@pytest.mark.asyncio
async def test_free_form_model_inference_is_rejected() -> None:
    premise_ids: list[str] = []

    def inferred(payload: dict[str, Any], call_number: int) -> ClaimExtractionResponse:
        if call_number == 1:
            return supporting_claim(
                payload,
                call_number,
                claim="Hiring activity",
                value="Increased",
            )
        return supporting_claim(
            payload,
            call_number,
            claim="Procurement signal",
            value="Likely capacity expansion",
            is_inference=True,
            derived_from_claim_ids=[premise_ids[0]],
            confidence=0.99,
        )

    processor, _, _, _ = make_processor(inferred)
    first = await processor.process(
        make_batch(
            (
                "q1",
                [make_result("https://company.example/jobs", "Hiring increased")],
            )
        )
    )
    premise_ids.append(first.bundle.evidence[0].claim_id)
    result = await processor.process(
        make_batch(
            (
                "q2",
                [
                    make_result(
                        "https://industry.example/analysis",
                        "Hiring may indicate capacity expansion",
                        source_type=SourceType.INDUSTRY,
                    )
                ],
            )
        )
    )

    assert first.bundle.evidence[0].claim == "Hiring activity"
    assert all(item.status != EvidenceStatus.INFERENCE for item in result.bundle.evidence)
    assert all(item.claim != "Procurement signal" for item in result.bundle.evidence)
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_evidence_inference_without_premises_is_unknown() -> None:
    def missing_premise(payload: dict[str, Any], call_number: int) -> ClaimExtractionResponse:
        return supporting_claim(
            payload,
            call_number,
            claim="Procurement signal",
            value="Likely expansion",
            is_inference=True,
            confidence=0.9,
        )

    processor, _, _, _ = make_processor(missing_premise)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [make_result("https://industry.example/analysis", "Expansion analysis")],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1
    assert result.new_evidence_count == 0


@pytest.mark.asyncio
async def test_evidence_inference_with_fabricated_premise_is_unknown() -> None:
    def fabricated_premise(payload: dict[str, Any], call_number: int) -> ClaimExtractionResponse:
        if call_number == 1:
            return supporting_claim(
                payload,
                call_number,
                claim="Hiring activity",
                value="Increased",
            )
        return supporting_claim(
            payload,
            call_number,
            claim="Procurement signal",
            value="Likely expansion",
            is_inference=True,
            derived_from_claim_ids=["clm_fabricated"],
            confidence=0.9,
        )

    processor, _, _, _ = make_processor(fabricated_premise)
    await processor.process(
        make_batch(("q1", [make_result("https://company.example/jobs", "Hiring increased")]))
    )
    result = await processor.process(
        make_batch(
            (
                "q2",
                [make_result("https://industry.example/analysis", "Expansion analysis")],
            )
        )
    )

    assert all(item.claim != "Procurement signal" for item in result.bundle.evidence)
    assert result.bundle.rejected_claim_count == 1
    assert result.new_evidence_count == 0


@pytest.mark.asyncio
async def test_evidence_conflicting_values_are_preserved() -> None:
    def conflicts(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        sources = payload["sources"]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=(
                        "Acme Pharma clinical stage"
                        if "Phase 2" in source["content"]
                        else "Trial phase"
                    ),
                    value=source["content"].split(" is ", 1)[1].rstrip("."),
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9 if "Phase 2" in source["content"] else 0.8,
                )
                for source in sources
            ]
        )

    processor, _, _, _ = make_processor(conflicts)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result("https://a.example/trial", "The trial is Phase 2."),
                    make_result("https://b.example/trial", "The trial is Phase 3."),
                ],
            )
        )
    )

    assert {item.value for item in result.bundle.evidence} == {"Phase 2", "Phase 3"}
    assert len(result.bundle.conflicts) == 1
    conflict = result.bundle.conflicts[0]
    assert set(conflict.values) == {"Phase 2", "Phase 3"}
    assert set(conflict.claim_ids) == {item.claim_id for item in result.bundle.evidence}


@pytest.mark.asyncio
async def test_paraphrased_claims_corroborate_on_controlled_predicate() -> None:
    def paraphrases(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=(
                        "Clinical stage" if "Clinical stage" in source["content"] else "Trial phase"
                    ),
                    value="Phase 2",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(paraphrases)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://company-one.com/trial",
                        "Clinical stage is Phase 2.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://regulator-two.gov/trial",
                        "Trial phase is Phase 2.",
                        source_type=SourceType.REGULATORY,
                    ),
                ],
            )
        )
    )

    assert len({item.claim_id for item in result.bundle.evidence}) == 1
    assert {item.status for item in result.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}


@pytest.mark.asyncio
async def test_set_valued_products_coexist_without_false_conflict() -> None:
    def products(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Product",
                    value="Alpha" if "Alpha" in source["content"] else "Beta",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(products)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result("https://one.com/a", "Product Alpha is listed."),
                    make_result("https://two.org/b", "Product Beta is listed."),
                ],
            )
        )
    )

    assert {item.value for item in result.bundle.evidence} == {"Alpha", "Beta"}
    assert result.bundle.conflicts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim", "value", "content"),
    [
        ("Mars ownership", "active", "The placebo arm remains active."),
        ("Revenue", 2, "Revenue grew in 2026."),
    ],
)
async def test_semantically_unrelated_or_partial_numeric_quotes_are_rejected(
    claim: str,
    value: Any,
    content: str,
) -> None:
    def fabricated(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=claim,
                    value=value,
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(fabricated)
    result = await processor.process(
        make_batch(("q1", [make_result("https://untrusted.example/a", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "content"),
    [
        ("Phase 2", "Acme Pharma revenue discussion occurred during Phase 2."),
        ("$5 billion", "Acme Pharma revenue was not $5 billion."),
    ],
)
async def test_cooccurrence_and_negation_cannot_launder_a_direct_relation(
    value: str,
    content: str,
) -> None:
    def fabricated(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value=value,
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(fabricated)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/revenue", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Acme Pharma did not report revenue of $5 billion.",
        "If Acme Pharma revenue was $5 billion, the scenario would change.",
    ],
)
async def test_negated_or_hypothetical_clause_is_not_a_fact(content: str) -> None:
    def fabricated(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(fabricated)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/revenue", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim", "value", "content"),
    [
        ("Revenue under IFRS", "$5 billion", "Acme Pharma revenue was $5 billion."),
        (
            "Manufacturing implies procurement demand",
            "expanding",
            "Acme Pharma manufacturing is expanding.",
        ),
    ],
)
async def test_absent_qualifier_or_embedded_inference_is_rejected(
    claim: str,
    value: str,
    content: str,
) -> None:
    def fabricated(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=claim,
                    value=value,
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(fabricated)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/claim", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_wrong_subject_quote_cannot_set_company_operating_status() -> None:
    def wrong_subject(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Operating status",
                    value="active",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(wrong_subject)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/trial",
                        "The placebo control status was active.",
                        bind_subject=False,
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_target_name_elsewhere_cannot_launder_another_company_metric() -> None:
    def wrong_subject(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(wrong_subject)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/comparison",
                        "Acme Pharma reported that Rival Corp revenue was $5 billion.",
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_competing_subject_before_target_cannot_launder_revenue() -> None:
    def wrong_subject(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(wrong_subject)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/acquisition",
                        ("Rival Corp acquired Acme Pharma and reported revenue of $5 billion."),
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim", "value", "content"),
    [
        (
            "Program B clinical stage",
            "Phase 2",
            "Acme Pharma program A is Phase 2, while program B is Phase 3.",
        ),
        (
            "Revenue under IFRS",
            "$5 billion",
            ("Acme Pharma revenue was $5 billion, while Rival Corp reported under IFRS."),
        ),
        (
            "Manufacturing drives procurement demand",
            "expanding",
            ("Acme Pharma manufacturing is expanding, procurement demand drives purchasing."),
        ),
    ],
)
async def test_clause_global_context_cannot_rebind_relation(
    claim: str,
    value: str,
    content: str,
) -> None:
    def misbound(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=claim,
                    value=value,
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(misbound)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/context", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_claim_context_must_bind_the_selected_predicate_occurrence() -> None:
    def misbound(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Program 1 clinical stage",
                    value="Phase 3",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(misbound)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/programs",
                        "Acme Pharma program 1 Phase 2 and program 2 Phase 3.",
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Rumors say Acme Pharma revenue was $5 billion.",
        "Analysts estimate Acme Pharma revenue was $5 billion.",
        "Acme Pharma allegedly reported revenue of $5 billion.",
    ],
)
async def test_attributed_or_speculative_revenue_is_not_a_fact(content: str) -> None:
    def speculative(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(speculative)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/rumor", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "submitted_quote"),
    [
        (
            "Rumors say Acme Pharma revenue was $5 billion.",
            "Acme Pharma revenue was $5 billion",
        ),
        (
            "Acme Pharma revenue was $5 billion or $4 billion.",
            "Acme Pharma revenue was $5 billion",
        ),
        (
            "Acme Pharma revenue was $5 billion, which is incorrect.",
            "Acme Pharma revenue was $5 billion",
        ),
    ],
)
async def test_model_cannot_strip_qualifying_source_context(
    content: str,
    submitted_quote: str,
) -> None:
    def stripped(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(
                            source_id=source["source_id"],
                            quote=submitted_quote,
                        )
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(stripped)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/context", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Acme Pharma revenue was $5 billion or $4 billion.",
        "Acme Pharma revenue was $5 billion unless the filing is wrong.",
        "Acme Pharma revenue was $5 billion, supposedly.",
        "Acme Pharma revenue was $5 billion, a disputed figure.",
        "Acme Pharma revenue was $5 billion, maybe.",
        "Acme Pharma revenue was $5 billion, which is incorrect.",
    ],
)
async def test_alternative_or_post_value_qualified_scalar_is_rejected(content: str) -> None:
    def qualified(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(qualified)
    result = await processor.process(
        make_batch(("q1", [make_result("https://example.com/qualified", content)]))
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_nearby_asset_identifier_cannot_bind_to_later_program_predicate() -> None:
    def misbound(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Program 1 clinical stage",
                    value="Phase 3",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(
                            source_id=source["source_id"],
                            quote=source["content"],
                        )
                    ],
                    confidence=0.9,
                )
            ]
        )

    processor, _, _, _ = make_processor(misbound)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/programs",
                        "Acme Pharma program 1, program 2 is Phase 3.",
                    )
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1


@pytest.mark.asyncio
async def test_explicitly_republished_fact_does_not_inflate_verification() -> None:
    def revenue_claims(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(
                            source_id=source["source_id"],
                            quote=source["content"],
                        )
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(revenue_claims)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://acme.example/release",
                        "Acme Pharma revenue was $5 billion.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://news.example/story",
                        (
                            "Acme Pharma reported revenue was $5 billion, republished "
                            "from the company press release."
                        ),
                        source_type=SourceType.NEWS,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert {item.status for item in result.bundle.evidence} == {EvidenceStatus.SINGLE_SOURCE}


@pytest.mark.asyncio
async def test_quote_period_qualifier_prevents_model_label_conflict_suppression() -> None:
    def annual_revenues(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim=("Annual revenue" if "$5 billion" in source["content"] else "Revenue"),
                    value=("$5 billion" if "$5 billion" in source["content"] else "$4 billion"),
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(
                            source_id=source["source_id"],
                            quote=source["content"],
                        )
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(annual_revenues)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://one.example/revenue",
                        "Acme Pharma annual revenue was $5 billion.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://two.example/revenue",
                        "Acme Pharma annual revenue was $4 billion.",
                        source_type=SourceType.INDUSTRY,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.conflicts) == 1
    assert set(result.bundle.conflicts[0].values) == {"$5 billion", "$4 billion"}


@pytest.mark.asyncio
async def test_scalar_manufacturing_capacity_values_preserve_conflict() -> None:
    def capacities(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Manufacturing capacity",
                    value=("10,000 units" if "10,000" in source["content"] else "20,000 units"),
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(capacities)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://manufacturer-one.com/capacity",
                        "Acme Pharma manufacturing capacity is 10,000 units.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://manufacturer-two.org/capacity",
                        "Acme Pharma manufacturing capacity is 20,000 units.",
                        source_type=SourceType.INDUSTRY,
                    ),
                ],
            )
        )
    )

    assert {item.value for item in result.bundle.evidence} == {
        "10,000 units",
        "20,000 units",
    }
    assert len(result.bundle.conflicts) == 1


@pytest.mark.asyncio
async def test_scalar_facility_size_values_preserve_conflict() -> None:
    def sizes(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Facility size",
                    value=("100,000 sq ft" if "100,000" in source["content"] else "200,000 sq ft"),
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9,
                )
                for source in payload["sources"]
            ]
        )

    processor, _, _, _ = make_processor(sizes)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://one.example/facility",
                        "Acme Pharma facility size is 100,000 sq ft.",
                        source_type=SourceType.OFFICIAL,
                    ),
                    make_result(
                        "https://two.example/facility",
                        "Acme Pharma facility size is 200,000 sq ft.",
                        source_type=SourceType.INDUSTRY,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.conflicts) == 1


@pytest.mark.asyncio
async def test_decimal_value_and_company_abbreviation_remain_valid() -> None:
    def revenue(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        source = payload["sources"][0]
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5.2 billion",
                    source_ids=[source["source_id"]],
                    supporting_quotes=[
                        SupportingQuote(source_id=source["source_id"], quote=source["content"])
                    ],
                    confidence=0.9,
                )
            ]
        )

    processor, _, _, _ = make_processor(revenue)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://example.com/revenue",
                        "Acme Pharma Inc. revenue was $5.2 billion.",
                        source_type=SourceType.OFFICIAL,
                    )
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert result.bundle.evidence[0].value == "$5.2 billion"


def test_raw_source_content_is_excluded_from_public_serialization() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registry.register_batch(
        make_batch(("q1", [make_result("https://example.com/a", "sensitive page body")]))
    )

    serialized = registry.sources[0].model_dump(mode="json")
    assert "content" not in serialized
    assert registry.sources[0].content == "Acme Pharma: sensitive page body"


@pytest.mark.asyncio
async def test_allowlisted_sources_with_unrelated_quotes_cannot_support_fabricated_value() -> None:
    def unrelated(payload: dict[str, Any], _call_number: int) -> ClaimExtractionResponse:
        ids = source_ids(payload)
        return ClaimExtractionResponse(
            claims=[
                ExtractedClaim(
                    claim="Revenue",
                    value="$5 billion",
                    source_ids=ids,
                    supporting_quotes=[
                        SupportingQuote(source_id=source_id, quote="The trial entered Phase 2.")
                        for source_id in ids
                    ],
                    confidence=1.0,
                )
            ]
        )

    processor, _, _, _ = make_processor(unrelated)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result("https://a.example/trial", "The trial entered Phase 2."),
                    make_result("https://b.example/trial", "The trial entered Phase 2."),
                ],
            )
        )
    )

    assert result.bundle.evidence == []
    assert result.bundle.rejected_claim_count == 1
    assert result.new_evidence_count == 0


@pytest.mark.asyncio
async def test_evidence_registry_and_claims_persist_across_rounds() -> None:
    processor, llm, _, _ = make_processor(supporting_claim)
    first = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://company.example/release",
                        "Acme Pharma trial is Phase 2 in company report.",
                        source_type=SourceType.OFFICIAL,
                    )
                ],
            )
        )
    )
    second = await processor.process(
        make_batch(
            (
                "q2",
                [
                    make_result(
                        "https://regulator.example/record",
                        "Acme Pharma trial lists Phase 2 in regulator record.",
                        source_type=SourceType.REGULATORY,
                    )
                ],
            )
        )
    )

    assert first.bundle.evidence[0].status == EvidenceStatus.VERIFIED_FACT
    assert len(second.bundle.evidence) == 1
    assert {item.status for item in second.bundle.evidence} == {EvidenceStatus.VERIFIED_FACT}
    assert len({item.claim_id for item in second.bundle.evidence}) == 1
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_evidence_duplicate_round_skips_llm_extraction() -> None:
    processor, llm, _, _ = make_processor(supporting_claim)
    await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://company.example/page",
                        "Acme Pharma clinical stage is Phase 2.",
                    )
                ],
            )
        )
    )
    second = await processor.process(
        make_batch(
            (
                "q2",
                [
                    make_result(
                        "https://company.example/page?utm_campaign=repeat",
                        "Acme Pharma clinical stage is Phase 2.",
                    )
                ],
            )
        )
    )

    assert len(llm.calls) == 1
    assert second.new_source_count == 0
    assert second.new_evidence_count == 0
    assert second.usage == LLMUsage()
    assert len(second.bundle.evidence) == 1
    assert second.bundle.sources[0].query_ids == ["q1", "q2"]


@pytest.mark.asyncio
async def test_evidence_limit_reclassifies_truncated_verified_claim() -> None:
    processor, _, _, _ = make_processor(supporting_claim, evidence_limit=1)
    result = await processor.process(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://regulator.example/trial",
                        "Acme Pharma trial lists Phase 2 in regulator record.",
                        source_type=SourceType.REGULATORY,
                    ),
                    make_result(
                        "https://company.example/release",
                        "Acme Pharma trial lists Phase 2 in company release.",
                        source_type=SourceType.OFFICIAL,
                    ),
                ],
            )
        )
    )

    assert len(result.bundle.evidence) == 1
    assert result.bundle.evidence[0].status == EvidenceStatus.VERIFIED_FACT


@pytest.mark.asyncio
async def test_evidence_ids_are_stable_across_input_order() -> None:
    results = [
        make_result(
            "https://regulator.example/trial",
            "Acme Pharma trial lists Phase 2 in regulator record.",
            source_type=SourceType.REGULATORY,
        ),
        make_result(
            "https://company.example/release",
            "Acme Pharma trial lists Phase 2 in company release.",
            source_type=SourceType.OFFICIAL,
        ),
    ]
    first, _, _, _ = make_processor(supporting_claim)
    second, _, _, _ = make_processor(supporting_claim)

    first_result = await first.process(make_batch(("q1", results)))
    second_result = await second.process(make_batch(("q1", list(reversed(results)))))

    assert [item.model_dump(mode="json") for item in first_result.bundle.evidence] == [
        item.model_dump(mode="json") for item in second_result.bundle.evidence
    ]
