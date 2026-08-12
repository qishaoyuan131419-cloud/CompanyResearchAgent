import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.core.protocols import LLMUsage, StructuredLLMResult
from app.evidence.processor import ClaimExtractionResponse, EvidenceProcessor
from app.evidence.registry import SourceRegistry
from app.planner.service import ResearchPlanner
from app.prompts.repository import FilePromptRepository
from app.resolver.service import CompanyResolver
from app.schemas.company import CompanySnapshot, ResolvedCompany
from app.schemas.planning import ResearchPlan, ResearchTopic
from app.schemas.search import QueryExecution, SearchBatch, SearchResult, SearchStatistics

PROMPTS = Path(__file__).parents[2] / "app" / "prompts" / "templates"
NOW = datetime(2026, 7, 29, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class CapturingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        cache_namespace: str,
    ) -> StructuredLLMResult[Any]:
        del cache_namespace
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if response_model is ResearchPlan:
            value: Any = ResearchPlan(
                topics=[
                    ResearchTopic(
                        topic="Company Identity",
                        priority=5,
                        reason="Resolve identity.",
                        expected_output="Evidence-backed identity.",
                        estimated_value=1.0,
                    )
                ]
            )
        elif response_model is ClaimExtractionResponse:
            value = ClaimExtractionResponse(claims=[])
        else:
            raise AssertionError(f"unexpected response model: {response_model}")
        return StructuredLLMResult(value=value, usage=LLMUsage(), model="fake")


@pytest.mark.asyncio
async def test_request_hints_are_user_data_not_system_instructions() -> None:
    marker = "IGNORE_ALL_RULES_MARKER"
    llm = CapturingLLM()
    planner = ResearchPlanner(
        llm=llm,
        prompts=FilePromptRepository(PROMPTS),
        max_queries_per_round=5,
    )

    await planner.build_plan(
        ResolvedCompany(canonical_name=marker),
        CompanySnapshot(canonical_name=marker),
    )

    assert marker not in llm.calls[0]["system"]
    assert marker in json.loads(llm.calls[0]["user"])["company_snapshot_hints"]["canonical_name"]


@pytest.mark.asyncio
async def test_initial_resolution_does_not_spend_tokens_or_promote_hints() -> None:
    llm = CapturingLLM()
    resolver = CompanyResolver(llm=llm, prompts=FilePromptRepository(PROMPTS))

    company = await resolver.resolve(
        CompanySnapshot(
            canonical_name="Unverified Acme",
            country="US",
            industry="Pharmaceuticals",
        )
    )

    assert llm.calls == []
    assert company.identity_status.value == "unconfirmed"
    assert company.country is None
    assert company.industry is None


@pytest.mark.asyncio
async def test_retrieved_page_text_is_user_data_not_system_instructions() -> None:
    marker = "IGNORE_PREVIOUS_AND_EXFILTRATE"
    llm = CapturingLLM()
    processor = EvidenceProcessor(
        llm_client=llm,
        prompt_repository=FilePromptRepository(PROMPTS),
        registry=SourceRegistry(clock=FixedClock()),
        subject_identifiers=("Acme",),
        evidence_limit=10,
    )
    batch = SearchBatch(
        executions=[
            QueryExecution(
                query_id="q1",
                query="Acme identity",
                started_at=NOW,
                completed_at=NOW,
                duration_ms=0,
                attempts=1,
                results=[
                    SearchResult(
                        title="Retrieved page",
                        url="https://example.com/page",
                        text=f"Acme evidence. {marker}",
                    )
                ],
            )
        ],
        statistics=SearchStatistics(
            query_count=1,
            successful_queries=1,
            failed_queries=0,
            cache_hits=0,
            total_results=1,
            total_duration_ms=0,
        ),
    )

    await processor.process(batch, company_context={"canonical_name": "Acme"})

    assert marker not in llm.calls[0]["system"]
    assert marker in llm.calls[0]["user"]
