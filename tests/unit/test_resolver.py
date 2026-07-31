from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.enums import EvidenceStatus, IdentityStatus, SourceType
from app.core.protocols import LLMUsage, StructuredLLMResult
from app.resolver.service import CompanyResolver
from app.schemas.company import (
    CompanyCandidate,
    CompanyRelationship,
    CompanySnapshot,
    ResolvedCompany,
)
from app.schemas.evidence import Evidence, EvidenceBundle, EvidenceConflict
from app.schemas.search import SourceDocument
from app.services.orchestrator import ResearchOrchestrator

NOW = datetime(2026, 7, 29, tzinfo=UTC)


class StaticPrompts:
    def render(self, name: str, variables: dict[str, Any]) -> str:
        del variables
        return name


class ResolverLLM:
    async def generate_structured(self, **kwargs: Any) -> StructuredLLMResult[Any]:
        del kwargs
        return StructuredLLMResult(
            value=ResolvedCompany(
                canonical_name="Acme Pharma",
                identity_status=IdentityStatus.PARTIALLY_CONFIRMED,
                confidence=0.8,
                claim_ids=["clm_identity", "clm_domain"],
            ),
            usage=LLMUsage(),
            model="fake",
        )


class ProposedResolverLLM:
    def __init__(self, proposed: ResolvedCompany) -> None:
        self._proposed = proposed

    async def generate_structured(self, **kwargs: Any) -> StructuredLLMResult[Any]:
        del kwargs
        return StructuredLLMResult(value=self._proposed, usage=LLMUsage(), model="fake")


@pytest.mark.asyncio
async def test_pfizer_legal_name_and_short_name_resolve_as_one_confirmed_identity() -> None:
    sources = [
        SourceDocument(
            source_id="src_sec",
            title="PFIZER INC.",
            url="https://www.sec.gov/Archives/edgar/data/78003/pfe.htm",
            retrieved_at=NOW,
            source_type=SourceType.REGULATORY,
            content_hash="cnt_sec",
            supported_claim_ids=["clm_legal_name"],
        ),
        SourceDocument(
            source_id="src_pfizer",
            title="Pfizer Company Profile",
            url="https://www.pfizer.com/about",
            retrieved_at=NOW,
            source_type=SourceType.OFFICIAL,
            content_hash="cnt_pfizer",
            supported_claim_ids=["clm_short_name"],
        ),
    ]
    evidence = [
        Evidence(
            evidence_id="ev_legal_name",
            claim_id="clm_legal_name",
            claim="Company name",
            value="PFIZER INC.",
            status=EvidenceStatus.VERIFIED_FACT,
            confidence=0.99,
            source_ids=["src_sec"],
            supporting_quotes=["Company name: PFIZER INC."],
        ),
        Evidence(
            evidence_id="ev_short_name",
            claim_id="clm_short_name",
            claim="Company name",
            value="Pfizer",
            status=EvidenceStatus.VERIFIED_FACT,
            confidence=0.99,
            source_ids=["src_pfizer"],
            supporting_quotes=["Company name: Pfizer."],
        ),
    ]
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Pfizer",
                identity_status=IdentityStatus.AMBIGUOUS,
                confidence=0.9,
            )
        ),
        prompts=StaticPrompts(),
    )
    false_conflict = EvidenceConflict(
        claim="Company name",
        claim_ids=["clm_legal_name", "clm_short_name"],
        values=["PFIZER INC.", "Pfizer"],
        description="Legacy false-positive name conflict.",
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Pfizer Inc."),
        EvidenceBundle(
            sources=sources,
            evidence=evidence,
            conflicts=[false_conflict],
        ),
    )

    assert resolved.canonical_name == "Pfizer Inc."
    assert resolved.identity_status == IdentityStatus.CONFIRMED
    assert resolved.aliases == ["Pfizer"]
    assert set(resolved.claim_ids) == {"clm_legal_name", "clm_short_name"}


@pytest.mark.asyncio
async def test_resolver_joins_website_only_from_provider_owned_source_lineage() -> None:
    source = SourceDocument(
        source_id="src_official",
        title="About Acme",
        url="https://acme.com/news/company-profile",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        content="Acme Pharma company identity profile.",
        content_hash="cnt_identity",
    )
    evidence = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id=source.source_id,
        source="acme.com",
        title=source.title,
        url=source.url,
        retrieved_at=NOW,
        source_type=source.source_type,
        supporting_quote="Acme Pharma company identity profile.",
    )
    domain_evidence = evidence.model_copy(
        update={
            "evidence_id": "ev_domain",
            "claim_id": "clm_domain",
            "claim": "Official domain",
            "value": "acme.com",
            "supporting_quote": "Acme Pharma official domain is acme.com.",
        }
    )
    resolver = CompanyResolver(llm=ResolverLLM(), prompts=StaticPrompts())

    resolved = await resolver.resolve(
        CompanySnapshot(
            canonical_name="Acme Pharma",
            website="https://acme.com/about",
        ),
        EvidenceBundle(sources=[source], evidence=[evidence, domain_evidence]),
    )

    assert str(resolved.website) == "https://acme.com/news/company-profile"
    assert resolved.identity_status == IdentityStatus.PARTIALLY_CONFIRMED
    assert resolved.claim_ids == ["clm_identity", "clm_domain"]


@pytest.mark.asyncio
async def test_resolver_does_not_validate_hint_without_explicit_domain_claim() -> None:
    source = SourceDocument(
        source_id="src_acquirer",
        title="MegaCorp acquisition article",
        url="https://megacorp.com/news/acme",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        content="Acme Pharma company identity profile.",
        content_hash="cnt_acquirer",
    )
    evidence = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id=source.source_id,
        source="megacorp.com",
        title=source.title,
        url=source.url,
        retrieved_at=NOW,
        source_type=source.source_type,
        supporting_quote="Acme Pharma company identity profile.",
    )
    resolver = CompanyResolver(llm=ResolverLLM(), prompts=StaticPrompts())

    resolved = await resolver.resolve(
        CompanySnapshot(
            canonical_name="Acme Pharma",
            website="https://megacorp.com/acme",
        ),
        EvidenceBundle(sources=[source], evidence=[evidence]),
    )

    assert resolved.website is None


@pytest.mark.asyncio
async def test_resolver_rejects_substring_laundered_identity_fields() -> None:
    evidence = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Company profile",
        url="https://example.com/profile",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Pharma",
                aliases=["A"],
                country="MA",
                industry="Pharma",
                identity_status=IdentityStatus.CONFIRMED,
                confidence=1.0,
                claim_ids=["clm_identity"],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[evidence]),
    )

    assert resolved.canonical_name == "Acme Pharma"
    assert resolved.aliases == []
    assert resolved.country is None
    assert resolved.industry is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim", "value", "proposed_update"),
    [
        ("Drug alias", "DrugX", {"canonical_name": "DrugX"}),
        ("Compound name", "DrugX", {"canonical_name": "DrugX"}),
        ("Molecule name", "MoleculeX", {"canonical_name": "MoleculeX"}),
        ("Products aliases", "DrugX", {"aliases": ["DrugX"]}),
        ("Assets country", "Canada", {"country": "Canada"}),
        ("Clinical trial country", "Canada", {"country": "Canada"}),
        ("Subsidiaries country", "Canada", {"country": "Canada"}),
        ("Drug candidate active status", "active", {"is_closed": False}),
        ("Funding round status", "closed", {"is_closed": True}),
    ],
)
async def test_resolver_rejects_asset_facts_as_company_identity(
    claim: str,
    value: str,
    proposed_update: dict[str, Any],
) -> None:
    evidence = Evidence(
        evidence_id="ev_asset",
        claim_id="clm_asset",
        claim=claim,
        value=value,
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_asset",
        source="example.com",
        title="Asset profile",
        url="https://example.com/asset",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote=f"Acme Pharma {claim}: {value}",
    )
    proposed = ResolvedCompany(
        canonical_name="Acme Pharma",
        identity_status=IdentityStatus.PARTIALLY_CONFIRMED,
        confidence=0.7,
        claim_ids=["clm_asset"],
    ).model_copy(update=proposed_update)
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(proposed),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[evidence]),
    )

    assert resolved.canonical_name == "Acme Pharma"
    assert resolved.aliases == []
    assert resolved.country is None
    assert resolved.is_closed is None
    assert resolved.claim_ids == []
    assert resolved.identity_status == IdentityStatus.UNCONFIRMED


@pytest.mark.asyncio
async def test_resolver_rejects_relationship_type_supported_only_by_generic_partner_word() -> None:
    evidence = Evidence(
        evidence_id="ev_partner",
        claim_id="clm_partner",
        claim="Licensing partner",
        value="MegaCorp",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_partner",
        source="example.com",
        title="Licensing announcement",
        url="https://example.com/partner",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma licensing partner is MegaCorp.",
    )
    proposed = ResolvedCompany(
        canonical_name="Acme Pharma",
        relationships=[
            CompanyRelationship(
                relationship_type="manufacturing partner",
                related_company="MegaCorp",
                claim_ids=["clm_partner"],
            )
        ],
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(proposed),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[evidence]),
    )

    assert resolved.relationships == []


@pytest.mark.asyncio
async def test_resolver_matches_relationship_type_as_exact_tokens() -> None:
    identity = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Identity profile",
        url="https://example.com/identity",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    apparent = identity.model_copy(
        update={
            "evidence_id": "ev_apparent",
            "claim_id": "clm_apparent",
            "claim": "Apparent owner",
            "value": "Beta Holdings",
            "supporting_quote": "Acme Pharma apparent owner is Beta Holdings.",
        }
    )
    proposed = ResolvedCompany(
        canonical_name="Acme Pharma",
        confidence=0.7,
        claim_ids=["clm_identity"],
        relationships=[
            CompanyRelationship(
                relationship_type="parent company",
                related_company="Beta Holdings",
                claim_ids=["clm_apparent"],
            )
        ],
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(proposed),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[identity, apparent]),
    )

    assert resolved.identity_status == IdentityStatus.PARTIALLY_CONFIRMED
    assert resolved.relationships == []


@pytest.mark.asyncio
async def test_resolver_does_not_take_relationship_status_from_an_unrelated_claim() -> None:
    identity = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Identity profile",
        url="https://example.com/identity",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    partner = identity.model_copy(
        update={
            "evidence_id": "ev_partner",
            "claim_id": "clm_partner",
            "claim": "Licensing partner",
            "value": "MegaCorp",
            "supporting_quote": "Acme Pharma licensing partner is MegaCorp.",
        }
    )
    trial_status = identity.model_copy(
        update={
            "evidence_id": "ev_trial",
            "claim_id": "clm_trial",
            "claim": "Clinical trial status",
            "value": "active",
            "supporting_quote": "Acme Pharma clinical trial status is active.",
        }
    )
    proposed = ResolvedCompany(
        canonical_name="Acme Pharma",
        confidence=0.7,
        claim_ids=["clm_identity"],
        relationships=[
            CompanyRelationship(
                relationship_type="licensing partner",
                related_company="MegaCorp",
                status="active",
                claim_ids=["clm_partner", "clm_trial"],
            )
        ],
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(proposed),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[identity, partner, trial_status]),
    )

    assert len(resolved.relationships) == 1
    assert resolved.claim_ids == ["clm_identity", "clm_partner"]
    assert resolved.relationships[0].claim_ids == ["clm_partner"]
    assert resolved.relationships[0].status is None

    constrained = ResearchOrchestrator._constrain_evidence_to_resolved_identity(
        resolved,
        EvidenceBundle(evidence=[identity, partner, trial_status]),
    )
    assert [item.claim_id for item in constrained.evidence] == [
        "clm_identity",
        "clm_partner",
    ]


@pytest.mark.asyncio
async def test_resolver_rejects_asset_scoped_relationship_status_laundering() -> None:
    identity = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Identity profile",
        url="https://example.com/identity",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    partner = identity.model_copy(
        update={
            "evidence_id": "ev_partner",
            "claim_id": "clm_partner",
            "claim": "Licensing partner",
            "value": "MegaCorp",
            "supporting_quote": "Acme Pharma licensing partner is MegaCorp.",
        }
    )
    trial_status = identity.model_copy(
        update={
            "evidence_id": "ev_trial",
            "claim_id": "clm_trial",
            "claim": "MegaCorp licensing partner clinical trial status",
            "value": "active",
            "supporting_quote": "MegaCorp licensing partner clinical trial status is active.",
        }
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme Pharma",
                confidence=0.7,
                claim_ids=["clm_identity"],
                relationships=[
                    CompanyRelationship(
                        relationship_type="licensing partner",
                        related_company="MegaCorp",
                        status="active",
                        claim_ids=["clm_partner", "clm_trial"],
                    )
                ],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=[identity, partner, trial_status]),
    )

    assert len(resolved.relationships) == 1
    assert resolved.relationships[0].status is None
    assert resolved.relationships[0].claim_ids == ["clm_partner"]


@pytest.mark.asyncio
async def test_resolver_does_not_validate_website_hint_with_news_source() -> None:
    source = SourceDocument(
        source_id="src_news",
        title="Reuters profile",
        url="https://reuters.com/article/acme",
        retrieved_at=NOW,
        source_type=SourceType.NEWS,
        content="Acme Pharma company identity profile.",
        content_hash="cnt_news",
    )
    evidence = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id=source.source_id,
        source="reuters.com",
        title=source.title,
        url=source.url,
        retrieved_at=NOW,
        source_type=source.source_type,
        supporting_quote="Acme Pharma company identity profile.",
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme Pharma",
                confidence=0.7,
                claim_ids=["clm_identity"],
                candidates=[
                    CompanyCandidate(
                        name="Acme Pharma",
                        website="https://reuters.com/article/acme",
                        claim_ids=["clm_identity"],
                    )
                ],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(
            canonical_name="Acme Pharma",
            website="https://reuters.com/acme",
        ),
        EvidenceBundle(sources=[source], evidence=[evidence]),
    )

    assert resolved.website is None
    assert resolved.candidates[0].website is None


@pytest.mark.asyncio
async def test_resolver_marks_two_evidence_linked_candidates_ambiguous() -> None:
    evidence = [
        Evidence(
            evidence_id=f"ev_candidate_{index}",
            claim_id=f"clm_candidate_{index}",
            claim="Company name",
            value=name,
            status=EvidenceStatus.SINGLE_SOURCE,
            confidence=0.7,
            source_id=f"src_candidate_{index}",
            source=f"source-{index}.example",
            title="Candidate profile",
            url=f"https://source-{index}.example/profile",
            retrieved_at=NOW,
            source_type=SourceType.OFFICIAL,
            supporting_quote=f"The company name is {name}.",
        )
        for index, name in enumerate(("Acme Pharma", "Acme Biologics"), start=1)
    ]
    candidates = [
        CompanyCandidate(name=str(item.value), claim_ids=[item.claim_id]) for item in evidence
    ]
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme",
                confidence=0.9,
                candidates=candidates,
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme"),
        EvidenceBundle(evidence=evidence),
    )

    assert resolved.identity_status == IdentityStatus.AMBIGUOUS
    assert [candidate.name for candidate in resolved.candidates] == [
        "Acme Biologics",
        "Acme Pharma",
    ]
    assert resolved.claim_ids == ["clm_candidate_1", "clm_candidate_2"]


@pytest.mark.asyncio
async def test_resolver_keeps_same_name_candidates_with_distinct_countries() -> None:
    evidence: list[Evidence] = []
    candidates: list[CompanyCandidate] = []
    for index, country in enumerate(("Canada", "United States"), start=1):
        name_claim_id = f"clm_name_{index}"
        country_claim_id = f"clm_country_{index}"
        evidence.extend(
            [
                Evidence(
                    evidence_id=f"ev_name_{index}",
                    claim_id=name_claim_id,
                    claim="Company name",
                    value="Acme Pharma",
                    status=EvidenceStatus.VERIFIED_FACT,
                    confidence=0.9,
                    source_id=f"src_name_{index}",
                    source=f"source-{index}.example",
                    title="Company profile",
                    url=f"https://source-{index}.example/profile",
                    retrieved_at=NOW,
                    source_type=SourceType.OFFICIAL,
                    supporting_quote="The company name is Acme Pharma.",
                ),
                Evidence(
                    evidence_id=f"ev_country_{index}",
                    claim_id=country_claim_id,
                    claim="Headquarters country",
                    value=country,
                    status=EvidenceStatus.SINGLE_SOURCE,
                    confidence=0.7,
                    source_id=f"src_country_{index}",
                    source=f"source-{index}.example",
                    title="Company location",
                    url=f"https://source-{index}.example/location",
                    retrieved_at=NOW,
                    source_type=SourceType.OFFICIAL,
                    supporting_quote=f"Acme Pharma headquarters country is {country}.",
                ),
            ]
        )
        candidates.append(
            CompanyCandidate(
                name="Acme Pharma",
                country=country,
                claim_ids=[name_claim_id, country_claim_id],
            )
        )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme Pharma",
                confidence=0.95,
                claim_ids=["clm_name_1", "clm_name_2"],
                candidates=candidates,
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=evidence),
    )

    assert resolved.identity_status == IdentityStatus.AMBIGUOUS
    assert [candidate.country for candidate in resolved.candidates] == [
        "Canada",
        "United States",
    ]


@pytest.mark.asyncio
async def test_resolver_cannot_confirm_identity_while_country_evidence_conflicts() -> None:
    name_evidence = [
        Evidence(
            evidence_id=f"ev_name_{index}",
            claim_id="clm_name",
            claim="Company name",
            value="Acme Pharma",
            status=EvidenceStatus.VERIFIED_FACT,
            confidence=0.9,
            source_id=f"src_name_{index}",
            source=f"source-{index}.example",
            title="Company profile",
            url=f"https://source-{index}.example/profile",
            retrieved_at=NOW,
            source_type=SourceType.OFFICIAL,
            supporting_quote="The company name is Acme Pharma.",
        )
        for index in (1, 2)
    ]
    country_evidence = [
        Evidence(
            evidence_id=f"ev_country_{index}",
            claim_id=f"clm_country_{index}",
            claim="Headquarters country",
            value=country,
            status=EvidenceStatus.SINGLE_SOURCE,
            confidence=0.7,
            source_id=f"src_country_{index}",
            source=f"country-{index}.example",
            title="Headquarters profile",
            url=f"https://country-{index}.example/profile",
            retrieved_at=NOW,
            source_type=SourceType.OFFICIAL,
            supporting_quote=f"Acme Pharma headquarters country is {country}.",
        )
        for index, country in enumerate(("Canada", "United States"), start=1)
    ]
    conflict = EvidenceConflict(
        claim="Headquarters country",
        claim_ids=["clm_country_1", "clm_country_2"],
        values=["Canada", "United States"],
        description="Conflicting headquarters countries.",
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme Pharma",
                confidence=0.95,
                claim_ids=["clm_name"],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(
            evidence=[*name_evidence, *country_evidence],
            conflicts=[conflict],
        ),
    )

    assert resolved.identity_status == IdentityStatus.AMBIGUOUS
    assert resolved.country is None
    assert {"clm_country_1", "clm_country_2"}.issubset(resolved.claim_ids)


@pytest.mark.asyncio
async def test_resolver_requires_verified_name_evidence_for_confirmed_identity() -> None:
    evidence = [
        Evidence(
            evidence_id=f"ev_industry_{index}",
            claim_id="clm_industry",
            claim="Industry",
            value="Pharmaceuticals",
            status=EvidenceStatus.VERIFIED_FACT,
            confidence=0.9,
            source_id=f"src_industry_{index}",
            source=f"source-{index}.example",
            title="Industry profile",
            url=f"https://source-{index}.example/industry",
            retrieved_at=NOW,
            source_type=SourceType.INDUSTRY,
            supporting_quote="Acme Pharma industry is Pharmaceuticals.",
        )
        for index in (1, 2)
    ]
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme Pharma",
                industry="Pharmaceuticals",
                confidence=0.95,
                claim_ids=["clm_industry"],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme Pharma"),
        EvidenceBundle(evidence=evidence),
    )

    assert resolved.identity_status == IdentityStatus.PARTIALLY_CONFIRMED
    assert resolved.industry == "Pharmaceuticals"


@pytest.mark.asyncio
async def test_resolver_rejects_nonidentity_candidate_details_and_claim_ids() -> None:
    name = Evidence(
        evidence_id="ev_name",
        claim_id="clm_name",
        claim="Company name",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_name",
        source="example.com",
        title="Company profile",
        url="https://example.com/profile",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="The company name is Acme Pharma.",
    )
    revenue = name.model_copy(
        update={
            "evidence_id": "ev_revenue",
            "claim_id": "clm_revenue",
            "claim": "Revenue",
            "value": "$5 billion",
            "supporting_quote": "Acme Pharma revenue was $5 billion.",
        }
    )
    resolver = CompanyResolver(
        llm=ProposedResolverLLM(
            ResolvedCompany(
                canonical_name="Acme",
                candidates=[
                    CompanyCandidate(
                        name="Acme Pharma",
                        distinguishing_details=["$5 billion"],
                        claim_ids=["clm_name", "clm_revenue"],
                    )
                ],
            )
        ),
        prompts=StaticPrompts(),
    )

    resolved = await resolver.resolve(
        CompanySnapshot(canonical_name="Acme"),
        EvidenceBundle(evidence=[name, revenue]),
    )

    assert resolved.candidates[0].claim_ids == ["clm_name"]
    assert resolved.candidates[0].distinguishing_details == []
    assert resolved.claim_ids == ["clm_name"]


@pytest.mark.parametrize(
    "identity_status",
    [
        IdentityStatus.AMBIGUOUS,
        IdentityStatus.PARTIALLY_CONFIRMED,
        IdentityStatus.UNCONFIRMED,
    ],
)
def test_unresolved_identity_excludes_nonidentity_evidence_from_final_bundle(
    identity_status: IdentityStatus,
) -> None:
    identity = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Identity profile",
        url="https://example.com/identity",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    revenue = identity.model_copy(
        update={
            "evidence_id": "ev_revenue",
            "claim_id": "clm_revenue",
            "claim": "Revenue",
            "value": "$5 billion",
            "supporting_quote": "Acme Pharma revenue was $5 billion.",
        }
    )
    company = ResolvedCompany(
        canonical_name="Acme Pharma",
        identity_status=identity_status,
        claim_ids=["clm_identity"],
    )

    filtered = ResearchOrchestrator._constrain_evidence_to_resolved_identity(
        company,
        EvidenceBundle(evidence=[identity, revenue]),
    )

    assert filtered.evidence == [identity]


def test_identity_constraint_retains_nested_company_claim_lineage() -> None:
    identity = Evidence(
        evidence_id="ev_identity",
        claim_id="clm_identity",
        claim="Company identity",
        value="Acme Pharma",
        status=EvidenceStatus.SINGLE_SOURCE,
        confidence=0.7,
        source_id="src_identity",
        source="example.com",
        title="Identity profile",
        url="https://example.com/identity",
        retrieved_at=NOW,
        source_type=SourceType.OFFICIAL,
        supporting_quote="Acme Pharma company identity.",
    )
    candidate = identity.model_copy(
        update={
            "evidence_id": "ev_candidate",
            "claim_id": "clm_candidate",
            "claim": "Company name",
            "value": "Acme Biologics",
            "supporting_quote": "The company name is Acme Biologics.",
        }
    )
    partner = identity.model_copy(
        update={
            "evidence_id": "ev_partner",
            "claim_id": "clm_partner",
            "claim": "Licensing partner",
            "value": "MegaCorp",
            "supporting_quote": "Acme Pharma licensing partner is MegaCorp.",
        }
    )
    company = ResolvedCompany(
        canonical_name="Acme Pharma",
        identity_status=IdentityStatus.PARTIALLY_CONFIRMED,
        claim_ids=["clm_identity"],
        candidates=[CompanyCandidate(name="Acme Biologics", claim_ids=["clm_candidate"])],
        relationships=[
            CompanyRelationship(
                relationship_type="licensing partner",
                related_company="MegaCorp",
                claim_ids=["clm_partner"],
            )
        ],
    )

    filtered = ResearchOrchestrator._constrain_evidence_to_resolved_identity(
        company,
        EvidenceBundle(evidence=[identity, candidate, partner]),
    )

    assert {item.claim_id for item in filtered.evidence} == {
        "clm_identity",
        "clm_candidate",
        "clm_partner",
    }
