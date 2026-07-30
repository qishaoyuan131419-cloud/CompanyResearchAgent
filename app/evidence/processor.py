"""Strict LLM claim extraction and evidence materialization."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from typing import Any

from pydantic import Field

from app.core.enums import EvidenceStatus
from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import (
    LLMClient,
    LLMUsage,
    PromptRepository,
    StructuredLLMResult,
)
from app.evidence.policy import EvidencePolicy
from app.evidence.registry import SourceRegistry
from app.schemas.base import StrictModel
from app.schemas.evidence import Evidence, EvidenceBundle, EvidenceConflict, ExtractedClaim
from app.schemas.search import SearchBatch, SourceDocument
from app.utils.hashing import stable_hash
from app.utils.text import normalize_text, normalized_fingerprint_text
from app.utils.urls import source_domain


class ClaimExtractionResponse(StrictModel):
    """The only shape accepted back from the extraction model.

    Source metadata is intentionally absent. The LLM may refer to registry IDs,
    but only the processor can attach titles, URLs, dates, and source types.
    """

    claims: list[ExtractedClaim] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvidenceProcessingResult:
    bundle: EvidenceBundle
    usage: LLMUsage
    new_source_count: int
    new_evidence_count: int
    duplicate_source_count: int
    extraction_errors: tuple[str, ...] = ()


@dataclass(slots=True)
class _ClaimRecord:
    claim_id: str
    claim: str
    value: Any
    value_fingerprint: Any
    predicate_key: str
    is_set_valued: bool
    first_seen_at: datetime
    direct_source_ids: set[str] = field(default_factory=set)
    inference_source_ids: set[str] = field(default_factory=set)
    inference_derived_from_claim_ids: set[str] = field(default_factory=set)
    supporting_quotes_by_source: dict[str, str] = field(default_factory=dict)
    direct_confidence: float = 0.0
    inference_confidence: float = 0.0


class EvidenceProcessor:
    """Maintains evidence state across search rounds for one research run."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
        registry: SourceRegistry,
        subject_identifiers: Sequence[str],
        evidence_limit: int,
        extraction_batch_size: int = 8,
        extraction_max_prompt_bytes: int = 80_000,
        policy: EvidencePolicy | None = None,
        cache_namespace: str = "evidence.extract.v1",
    ) -> None:
        if evidence_limit < 1:
            raise ValueError("evidence_limit must be at least one")
        if extraction_batch_size < 1:
            raise ValueError("extraction_batch_size must be at least one")
        if extraction_max_prompt_bytes < 1:
            raise ValueError("extraction_max_prompt_bytes must be at least one")
        identifiers = tuple(subject_identifiers)
        subject_token_sequences = tuple(
            tokens for identifier in identifiers if (tokens := self._identifier_tokens(identifier))
        )
        if not subject_token_sequences:
            raise ValueError("at least one company subject identifier is required")
        self._llm_client = llm_client
        self._prompt_repository = prompt_repository
        self._registry = registry
        self._subject_token_sequences = subject_token_sequences
        self._subject_tokens = frozenset().union(
            *(frozenset(self._tokens(identifier)) for identifier in identifiers)
        )
        self._evidence_limit = evidence_limit
        self._extraction_batch_size = extraction_batch_size
        self._extraction_max_prompt_bytes = extraction_max_prompt_bytes
        self._policy = policy or EvidencePolicy()
        self._cache_namespace = cache_namespace
        self._claims: dict[str, _ClaimRecord] = {}
        self._rejected_claims = 0
        self._processing_errors: list[str] = []

    async def process(
        self,
        batch: SearchBatch,
        *,
        company_context: Mapping[str, Any] | None = None,
    ) -> EvidenceProcessingResult:
        visible_evidence_ids_before = {
            item.evidence_id
            for item in self._build_bundle().evidence
            if item.status != EvidenceStatus.UNKNOWN
        }
        registration = self._registry.register_batch(batch)
        extraction_source_ids = tuple(
            source_id
            for source_id in registration.new_source_ids
            if self._is_citable(self._registry.get(source_id))
        )
        usage = LLMUsage()
        extraction_errors: list[str] = []

        if extraction_source_ids:
            documents = [self._registry.get(source_id) for source_id in extraction_source_ids]
            for document_chunk in self._document_chunks(documents):
                existing_claims: list[dict[str, Any]] = []
                try:
                    llm_result = await self._extract_claims(
                        documents=document_chunk,
                        company_context=company_context or {},
                        existing_claims=existing_claims,
                    )
                except BudgetExceededError:
                    error = "BudgetExceededError: evidence extraction budget reached"
                    extraction_errors.append(error)
                    break
                except StructuredOutputError as exc:
                    error = f"{type(exc).__name__}: evidence extraction unavailable"
                    extraction_errors.append(error)
                    break
                extraction = ClaimExtractionResponse.model_validate(llm_result.value)
                self._merge_extracted_claims(
                    extraction.claims,
                    allowed_source_ids=frozenset(document.source_id for document in document_chunk),
                )
                usage = LLMUsage(
                    input_tokens=usage.input_tokens + llm_result.usage.input_tokens,
                    output_tokens=usage.output_tokens + llm_result.usage.output_tokens,
                    estimated_cost_usd=(
                        usage.estimated_cost_usd + llm_result.usage.estimated_cost_usd
                    ),
                )

        self._processing_errors.extend(extraction_errors)

        bundle = self._build_bundle()
        visible_evidence_ids_after = {
            item.evidence_id for item in bundle.evidence if item.status != EvidenceStatus.UNKNOWN
        }

        return EvidenceProcessingResult(
            bundle=bundle,
            usage=usage,
            new_source_count=len(registration.new_source_ids),
            new_evidence_count=len(visible_evidence_ids_after - visible_evidence_ids_before),
            duplicate_source_count=registration.duplicate_count,
            extraction_errors=tuple(extraction_errors),
        )

    async def _extract_claims(
        self,
        *,
        documents: list[SourceDocument],
        company_context: Mapping[str, Any],
        existing_claims: list[dict[str, Any]],
    ) -> StructuredLLMResult[ClaimExtractionResponse]:
        system_prompt = self._prompt_repository.render(
            "extractor",
            {
                "resolved_company": "Supplied in the structured user JSON payload.",
                "source_documents": "Supplied in the structured user JSON payload.",
                "existing_claims": "Supplied in the structured user JSON payload.",
            },
        )
        payload = {
            "company_context": company_context,
            "existing_claims": existing_claims,
            "sources": [self._source_payload(document) for document in documents],
        }
        return await self._llm_client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, sort_keys=True, default=str),
            response_model=ClaimExtractionResponse,
            cache_namespace=self._cache_namespace,
        )

    def _merge_extracted_claims(
        self,
        claims: list[ExtractedClaim],
        *,
        allowed_source_ids: frozenset[str],
    ) -> None:
        for extracted in claims:
            # Free-form model reasoning cannot establish entailment. Inference
            # status is reserved for future deterministic typed rules.
            if (
                extracted.is_inference
                or extracted.derived_from_claim_ids
                or self._value_contains_url(extracted.value)
            ):
                self._rejected_claims += 1
                continue
            resolved_source_ids: set[str] = set()
            claim_has_rejection = False
            quotes_by_source: dict[str, list[str]] = defaultdict(list)
            for support in extracted.supporting_quotes:
                quotes_by_source[support.source_id].append(support.quote)
            for source_id in extracted.source_ids:
                resolved = self._registry.resolve_id(source_id)
                if source_id not in allowed_source_ids or not self._registry.contains(resolved):
                    claim_has_rejection = True
                    continue
                source = self._registry.get(resolved)
                if not self._is_citable(source):
                    claim_has_rejection = True
                    continue
                quotes = quotes_by_source.get(source_id, [])
                if not quotes or not all(self._quote_is_present(quote, source) for quote in quotes):
                    claim_has_rejection = True
                    continue
                if not all(
                    self._quote_supports_direct_claim(
                        quote,
                        extracted.claim,
                        extracted.value,
                    )
                    for quote in quotes
                ):
                    claim_has_rejection = True
                    continue
                resolved_source_ids.add(source.source_id)
            if set(quotes_by_source) != set(extracted.source_ids):
                claim_has_rejection = True

            # Fail closed before a claim record exists. Rejected model text must
            # never be echoed back as an Unknown fact.
            if claim_has_rejection or not resolved_source_ids:
                self._rejected_claims += 1
                continue

            normalized_claim = normalize_text(extracted.claim)
            value_fingerprint = self._value_fingerprint(extracted.value)
            materialized_semantics = self._claim_semantics_for_quotes(
                normalized_claim,
                extracted.value,
                [
                    quote
                    for source_id in extracted.source_ids
                    for quote in quotes_by_source[source_id]
                ],
            )
            if materialized_semantics is None:
                self._rejected_claims += 1
                continue
            predicate_key, is_set_valued = materialized_semantics
            claim_id = stable_hash(
                {
                    "predicate": predicate_key,
                    "value": value_fingerprint,
                },
                prefix="clm_",
                length=24,
            )
            record = self._claims.get(claim_id)
            if record is None:
                record = _ClaimRecord(
                    claim_id=claim_id,
                    claim=normalized_claim,
                    value=extracted.value,
                    value_fingerprint=value_fingerprint,
                    predicate_key=predicate_key,
                    is_set_valued=is_set_valued,
                    first_seen_at=self._first_seen_at(resolved_source_ids),
                )
                self._claims[claim_id] = record
            else:
                record.claim = min(record.claim, normalized_claim)
                record.value = self._preferred_value(record.value, extracted.value)

            for source_id in extracted.source_ids:
                resolved = self._registry.resolve_id(source_id)
                preferred_quote = min(
                    (normalize_text(quote) for quote in quotes_by_source[source_id]),
                    key=lambda quote: (len(quote), quote.casefold()),
                )
                current_quote = record.supporting_quotes_by_source.get(resolved)
                if current_quote is None or (len(preferred_quote), preferred_quote.casefold()) < (
                    len(current_quote),
                    current_quote.casefold(),
                ):
                    record.supporting_quotes_by_source[resolved] = preferred_quote

            record.direct_source_ids.update(resolved_source_ids)
            record.direct_confidence = max(record.direct_confidence, extracted.confidence)

    def _build_bundle(self) -> EvidenceBundle:
        groups: list[tuple[_ClaimRecord, list[Evidence]]] = []
        for record in self._claims.values():
            evidence = self._materialize_claim(record)
            groups.append((record, evidence))

        groups.sort(key=self._group_sort_key)
        selected: list[Evidence] = []
        selected_records: dict[str, _ClaimRecord] = {}
        for record, group_evidence in groups:
            remaining = self._evidence_limit - len(selected)
            if remaining <= 0:
                break
            taken = group_evidence[:remaining]
            if taken:
                selected.extend(taken)
                selected_records[record.claim_id] = record

        selected = self._reclassify_after_limit(selected, selected_records)
        conflicts = self._build_conflicts(selected, selected_records)
        claims_by_source: dict[str, set[str]] = defaultdict(set)
        for item in selected:
            if item.source_id is not None and item.status != EvidenceStatus.UNKNOWN:
                claims_by_source[item.source_id].add(item.claim_id)
        sources = [
            source.model_copy(
                update={
                    "supported_claim_ids": sorted(claims_by_source.get(source.source_id, set()))
                }
            )
            for source in self._registry.sources
        ]
        return EvidenceBundle(
            sources=sources,
            evidence=selected,
            conflicts=conflicts,
            rejected_claims=self._rejected_claims,
            processing_errors=list(self._processing_errors),
        )

    def _all_evidence_ids(self) -> set[str]:
        return {
            item.evidence_id
            for record in self._claims.values()
            for item in self._materialize_claim(record)
            if item.status != EvidenceStatus.UNKNOWN
        }

    def _supported_premise_payload(self) -> list[dict[str, Any]]:
        premises: list[dict[str, Any]] = []
        for record in sorted(self._claims.values(), key=lambda item: item.claim_id):
            sources = self._resolved_sources(record.direct_source_ids)
            if not sources:
                continue
            status = self._policy.status_for(sources=sources, is_inference=False)
            premises.append(
                {
                    "claim_id": record.claim_id,
                    "claim": record.claim,
                    "value": record.value,
                    "status": status.value,
                }
            )
        return premises

    def _materialize_claim(self, record: _ClaimRecord) -> list[Evidence]:
        direct_sources = [
            source
            for source in self._resolved_sources(record.direct_source_ids)
            if self._source_still_supports(record, source)
        ]
        inference_sources = self._resolved_sources(record.inference_source_ids)
        if direct_sources:
            sources = direct_sources
            is_inference = False
            extracted_confidence = record.direct_confidence
        elif inference_sources:
            sources = inference_sources
            is_inference = True
            extracted_confidence = record.inference_confidence
        else:
            return [self._unknown_evidence(record)]

        status = self._policy.status_for(
            sources=self._independent_sources(record, sources),
            is_inference=is_inference,
        )
        confidence = self._policy.confidence_for(status=status, extracted=extracted_confidence)
        return [
            Evidence(
                evidence_id=self._evidence_id(record.claim_id, source.source_id),
                claim_id=record.claim_id,
                claim=record.claim,
                value=record.value,
                status=status,
                confidence=confidence,
                source_id=source.source_id,
                source=source_domain(str(source.url)),
                title=source.title,
                url=source.url,
                published_at=source.published_at,
                retrieved_at=source.retrieved_at,
                source_type=source.source_type,
                supporting_quote=self._supporting_quote(record, source.source_id),
                derived_from_claim_ids=(
                    sorted(record.inference_derived_from_claim_ids) if is_inference else []
                ),
            )
            for source in sources
        ]

    def _unknown_evidence(self, record: _ClaimRecord) -> Evidence:
        return Evidence(
            evidence_id=self._evidence_id(record.claim_id, None),
            claim_id=record.claim_id,
            claim=record.claim,
            value=record.value,
            status=EvidenceStatus.UNKNOWN,
            confidence=0.0,
            retrieved_at=record.first_seen_at,
        )

    def _reclassify_after_limit(
        self,
        evidence: list[Evidence],
        records: Mapping[str, _ClaimRecord],
    ) -> list[Evidence]:
        by_claim: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            by_claim[item.claim_id].append(item)

        selected_direct_claim_ids = {
            claim_id
            for claim_id, items in by_claim.items()
            if records[claim_id].direct_source_ids
            and any(item.source_id is not None for item in items)
        }
        reclassified: list[Evidence] = []
        for claim_id in sorted(by_claim):
            items = by_claim[claim_id]
            record = records[claim_id]
            sources = [
                self._registry.get(item.source_id) for item in items if item.source_id is not None
            ]
            is_inference = not record.direct_source_ids and bool(record.inference_source_ids)
            if is_inference and not record.inference_derived_from_claim_ids.issubset(
                selected_direct_claim_ids
            ):
                reclassified.append(self._unknown_evidence(record))
                continue
            status = self._policy.status_for(
                sources=self._independent_sources(record, sources),
                is_inference=is_inference,
            )
            extracted_confidence = (
                record.inference_confidence if is_inference else record.direct_confidence
            )
            confidence = self._policy.confidence_for(status=status, extracted=extracted_confidence)
            reclassified.extend(
                item.model_copy(update={"status": status, "confidence": confidence})
                for item in sorted(items, key=lambda value: value.evidence_id)
            )
        return reclassified

    def _build_conflicts(
        self,
        evidence: list[Evidence],
        records: Mapping[str, _ClaimRecord],
    ) -> list[EvidenceConflict]:
        supported_claim_ids = {
            item.claim_id for item in evidence if item.status != EvidenceStatus.UNKNOWN
        }
        by_claim_name: dict[str, list[_ClaimRecord]] = defaultdict(list)
        for claim_id in supported_claim_ids:
            record = records[claim_id]
            if not record.is_set_valued:
                by_claim_name[record.predicate_key].append(record)

        conflicts: list[EvidenceConflict] = []
        for normalized_claim in sorted(by_claim_name):
            records_for_claim = by_claim_name[normalized_claim]
            distinct_values = {
                json.dumps(record.value_fingerprint, sort_keys=True, default=str)
                for record in records_for_claim
            }
            if len(distinct_values) < 2:
                continue
            ordered = sorted(records_for_claim, key=lambda record: record.claim_id)
            display_claim = min(record.claim for record in ordered)
            conflicts.append(
                EvidenceConflict(
                    claim=display_claim,
                    claim_ids=[record.claim_id for record in ordered],
                    values=[record.value for record in ordered],
                    description=(
                        f"Conflicting supported values were retained for '{display_claim}'."
                    ),
                )
            )
        return conflicts

    def _resolved_sources(self, source_ids: set[str]) -> list[SourceDocument]:
        resolved_ids = {
            self._registry.resolve_id(source_id)
            for source_id in source_ids
            if self._registry.contains(source_id)
        }
        return [
            self._registry.get(source_id)
            for source_id in sorted(resolved_ids)
            if self._is_citable(self._registry.get(source_id))
        ]

    def _independent_sources(
        self,
        record: _ClaimRecord,
        sources: list[SourceDocument],
    ) -> list[SourceDocument]:
        independent: list[SourceDocument] = []
        ordered = sorted(
            sources,
            key=lambda source: (
                not self._policy.is_reliable(source),
                source.source_id,
            ),
        )
        for source in ordered:
            if any(
                self._sources_appear_syndicated(record, source, existing)
                for existing in independent
            ):
                continue
            independent.append(source)
        return independent

    def _sources_appear_syndicated(
        self,
        record: _ClaimRecord,
        first: SourceDocument,
        second: SourceDocument,
    ) -> bool:
        try:
            first_quote_tokens = self._tokens(self._supporting_quote(record, first.source_id))
            second_quote_tokens = self._tokens(self._supporting_quote(record, second.source_id))
        except RuntimeError:
            return True
        if any(
            self._has_redistribution_marker(tokens)
            for tokens in (
                first_quote_tokens,
                second_quote_tokens,
                self._tokens(first.content),
                self._tokens(second.content),
            )
        ):
            return True

        first_quote = " ".join(first_quote_tokens)
        second_quote = " ".join(second_quote_tokens)
        shorter_quote, longer_quote = sorted((first_quote, second_quote), key=len)
        if len(shorter_quote) >= 20 and shorter_quote in longer_quote:
            return True

        first_content = " ".join(self._tokens(first.content))
        second_content = " ".join(self._tokens(second.content))
        if not first_content or not second_content:
            return False
        shorter, longer = sorted((first_content, second_content), key=len)
        return len(shorter) >= 20 and shorter in longer

    def _first_seen_at(self, source_ids: set[str]) -> datetime:
        if source_ids:
            return min(self._registry.get(source_id).retrieved_at for source_id in source_ids)
        sources = self._registry.sources
        if sources:
            return max(source.retrieved_at for source in sources)
        raise RuntimeError("claims cannot be extracted without a registered source")

    def _group_sort_key(self, group: tuple[_ClaimRecord, list[Evidence]]) -> tuple[int, float, str]:
        record, evidence = group
        status_rank = {
            EvidenceStatus.VERIFIED_FACT: 0,
            EvidenceStatus.SINGLE_SOURCE: 1,
            EvidenceStatus.INFERENCE: 2,
            EvidenceStatus.UNKNOWN: 3,
        }
        status = evidence[0].status
        confidence = max(item.confidence for item in evidence)
        return (status_rank[status], -confidence, record.claim_id)

    @staticmethod
    def _is_citable(source: SourceDocument) -> bool:
        return bool(normalize_text(source.title))

    def _quote_is_present(self, quote: str, source: SourceDocument) -> bool:
        """Require an extracted quote to cover a complete source statement.

        A model-selected substring is not enough: it could trim a rumor prefix,
        an alternative value, or a contradicting suffix. The only permitted
        omitted statement context is an explicit redistribution notice, which is
        handled separately when source independence is calculated.
        """

        quote_tokens = self._tokens(quote)
        if not quote_tokens or not self._is_single_statement(quote):
            return False
        for statement_tokens in self._statement_token_sequences(source.content):
            if statement_tokens == quote_tokens:
                return True
            for start, end in self._sequence_ranges(statement_tokens, quote_tokens):
                omitted = [*statement_tokens[:start], *statement_tokens[end:]]
                if self._redistribution_context_is_safe(omitted):
                    return True
        return False

    @classmethod
    def _statement_token_sequences(cls, value: str) -> list[list[str]]:
        protected = cls._mask_nonterminal_periods(normalize_text(value))
        return [
            tokens
            for statement in re.split(r"[.!?;]+", protected)
            if (tokens := cls._tokens(statement))
        ]

    @staticmethod
    def _mask_nonterminal_periods(value: str) -> str:
        marker = "\u2024"
        candidate = re.sub(r"(?<=\d)\.(?=\d)", marker, value)
        candidate = re.sub(
            r"\b(?:ag|bv|co|corp|dr|inc|ltd|llc|nv|plc|sa|se|st)\.",
            lambda match: f"{match.group(0)[:-1]}{marker}",
            candidate,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"\b(?:[a-z]\.){2,}",
            lambda match: match.group(0).replace(".", marker),
            candidate,
            flags=re.IGNORECASE,
        )

    @classmethod
    def _has_redistribution_marker(cls, tokens: Sequence[str]) -> bool:
        return any(
            token.startswith(stem) for token in tokens for stem in cls._REDISTRIBUTION_STEMS
        ) or any(
            phrase in " ".join(tokens)
            for phrase in ("press release copy", "wire copy", "distributed by", "copied from")
        )

    def _redistribution_context_is_safe(self, tokens: Sequence[str]) -> bool:
        if not tokens or not self._has_redistribution_marker(tokens):
            return False
        allowed = (
            self._REDISTRIBUTION_CONTEXT_WORDS
            | self._SOURCE_ATTRIBUTION_WORDS
            | self._RELATION_GLUE
            | self._CLAIM_STOPWORDS
            | self._GENERIC_COMPANY_WORDS
            | self._subject_tokens
        )
        return all(token in allowed for token in tokens)

    @classmethod
    def _quote_supports_value(cls, quote: str, value: Any) -> bool:
        normalized_quote = normalized_fingerprint_text(quote)
        if isinstance(value, list):
            return bool(value) and all(cls._quote_supports_value(quote, item) for item in value)
        if value is None:
            return False
        normalized_value = normalized_fingerprint_text(str(value))
        if not normalized_value:
            return False
        pattern = r"(?<!\w)" + re.escape(normalized_value).replace(r"\ ", r"\s+") + r"(?!\w)"
        return re.search(pattern, normalized_quote) is not None

    _CLAIM_TAXONOMY: tuple[tuple[str, frozenset[str], bool], ...] = (
        (
            "clinical_stage",
            frozenset(
                {
                    "clinical",
                    "development",
                    "phase",
                    "pipeline",
                    "program",
                    "stage",
                    "trial",
                }
            ),
            False,
        ),
        ("revenue", frozenset({"revenue", "sales", "income", "turnover"}), False),
        (
            "company_identity",
            frozenset({"identity", "legal", "name", "alias", "renamed", "company"}),
            False,
        ),
        (
            "operating_status",
            frozenset({"closed", "active", "dissolved", "operating", "status"}),
            False,
        ),
        (
            "country",
            frozenset({"country", "headquarter", "headquarters", "headquartered", "location"}),
            False,
        ),
        ("industry", frozenset({"industry", "sector", "business", "focus"}), False),
        (
            "product",
            frozenset({"product", "products", "drug", "therapy", "asset", "portfolio"}),
            True,
        ),
        ("technology", frozenset({"technology", "platform", "capability", "capabilities"}), True),
        (
            "manufacturing",
            frozenset({"manufacturing", "facility", "plant", "production", "capacity"}),
            True,
        ),
        (
            "partnership",
            frozenset({"partnership", "partner", "collaboration", "licensing", "license"}),
            True,
        ),
        (
            "supply_chain",
            frozenset({"supply", "supplier", "chain", "dependency", "dependencies"}),
            True,
        ),
        ("procurement", frozenset({"procurement", "purchasing", "purchase", "vendor"}), True),
        ("funding", frozenset({"funding", "financing", "raised", "investment"}), True),
        ("hiring", frozenset({"hiring", "jobs", "recruitment", "vacancies"}), True),
        ("risk", frozenset({"risk", "risks", "competition", "competitor"}), True),
    )
    _CLAIM_STOPWORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "of",
            "for",
            "and",
            "or",
            "current",
            "recent",
            "reported",
            "information",
            "details",
            "overview",
            "signal",
            "activity",
            "fact",
        }
    )
    _GENERIC_COMPANY_WORDS = frozenset(
        {
            "ag",
            "biopharma",
            "biotech",
            "company",
            "corp",
            "corporation",
            "inc",
            "llc",
            "ltd",
            "pharma",
            "pharmaceutical",
            "pharmaceuticals",
            "plc",
            "sa",
            "therapeutics",
        }
    )
    _SCALAR_QUALIFIERS = frozenset(
        {
            "amount",
            "area",
            "capacity",
            "count",
            "date",
            "footprint",
            "number",
            "phase",
            "scale",
            "size",
            "sqft",
            "stage",
            "status",
            "throughput",
            "total",
            "value",
            "volume",
            "year",
        }
    )
    _PERIOD_LINKAGE_WORDS = frozenset(
        {
            "annual",
            "calendar",
            "fiscal",
            "full",
            "fy",
            "half",
            "monthly",
            "quarter",
            "quarterly",
            "s",
            "week",
            "weekly",
            "year",
            "yearly",
        }
    )
    _PERIOD_PREDICATE_QUALIFIERS = _PERIOD_LINKAGE_WORDS - {"s"}
    _SOURCE_ATTRIBUTION_WORDS = frozenset(
        {
            "agency",
            "authority",
            "company",
            "database",
            "document",
            "filing",
            "first",
            "journal",
            "news",
            "official",
            "one",
            "provider",
            "publication",
            "record",
            "registry",
            "regulator",
            "release",
            "report",
            "second",
            "social",
            "source",
            "study",
            "two",
        }
    )
    _REDISTRIBUTION_STEMS = (
        "copied",
        "redistribut",
        "reprint",
        "republish",
        "syndicat",
    )
    _REDISTRIBUTION_CONTEXT_WORDS = frozenset(
        {
            "by",
            "copy",
            "copied",
            "desk",
            "distributed",
            "distribution",
            "for",
            "from",
            "media",
            "press",
            "redistributed",
            "reprint",
            "reprinted",
            "relations",
            "release",
            "republished",
            "syndicated",
            "syndication",
            "the",
            "wire",
        }
    )
    _RELATION_GLUE = frozenset(
        {
            "about",
            "approximately",
            "are",
            "as",
            "at",
            "been",
            "being",
            "confirms",
            "decreased",
            "entered",
            "from",
            "grew",
            "had",
            "has",
            "have",
            "increased",
            "is",
            "listed",
            "lists",
            "of",
            "record",
            "remains",
            "reported",
            "reports",
            "says",
            "shows",
            "stands",
            "to",
            "totaled",
            "was",
            "were",
        }
    )
    _NEGATION_WORDS = frozenset({"denied", "denies", "false", "never", "no", "not", "without"})
    _NONFACTUAL_WORDS = frozenset(
        {
            "about",
            "appears",
            "alleged",
            "allegedly",
            "analyst",
            "analysts",
            "approximate",
            "approximately",
            "around",
            "could",
            "estimate",
            "estimated",
            "estimates",
            "expected",
            "forecast",
            "hypothetical",
            "if",
            "implies",
            "likely",
            "may",
            "might",
            "nearly",
            "perhaps",
            "possibly",
            "potential",
            "projected",
            "purported",
            "purportedly",
            "roughly",
            "rumor",
            "rumors",
            "rumoured",
            "seems",
            "supposed",
            "supposedly",
            "suggests",
            "dispute",
            "disputed",
            "disputes",
            "unverified",
            "would",
        }
    )
    _COMPLEX_RELATION_WORDS = frozenset(
        {"although", "but", "however", "or", "unless", "whereas", "while"}
    )

    def _quote_supports_direct_claim(self, quote: str, claim: str, value: Any) -> bool:
        if not self._is_single_statement(quote):
            return False
        quote_tokens = set(self._tokens(quote))
        if quote_tokens & self._COMPLEX_RELATION_WORDS:
            return False
        return (
            self._quote_supports_claim(quote, claim)
            and self._quote_supports_relation(quote, claim, value)
            and self._quote_supports_subject_relation(quote, claim, value)
            and self._quote_has_safe_trailing_context(quote, claim, value)
        )

    def _quote_has_safe_trailing_context(
        self,
        quote: str,
        claim: str,
        value: Any,
    ) -> bool:
        quote_tokens = self._tokens(quote)
        values = value if isinstance(value, list) else [value]
        value_ranges = [
            token_range
            for item in values
            if item is not None
            for token_range in self._sequence_ranges(quote_tokens, self._tokens(str(item)))
        ]
        if not value_ranges:
            return False
        tail = quote_tokens[max(end for _, end in value_ranges) :]
        if not tail:
            return True

        predicate_key, _ = self._claim_semantics(claim)
        category = predicate_key.split(":", 1)[0]
        taxonomy = next(
            (vocabulary for key, vocabulary, _ in self._CLAIM_TAXONOMY if key == category),
            frozenset(),
        )
        context_tokens = self._claim_context_tokens(claim, taxonomy)
        core_tokens = (
            self._RELATION_GLUE
            | taxonomy
            | self._CLAIM_STOPWORDS
            | self._GENERIC_COMPANY_WORDS
            | self._subject_tokens
            | context_tokens
            | self._PERIOD_LINKAGE_WORDS
        )
        if all(token in core_tokens or token.isdigit() for token in tail):
            return True
        if self._redistribution_context_is_safe(tail):
            return True
        return self._tail_has_safe_attribution(tail, core_tokens)

    def _tail_has_safe_attribution(
        self,
        tail: list[str],
        core_tokens: frozenset[str],
    ) -> bool:
        for index in range(len(tail)):
            if tail[index : index + 2] == ["according", "to"]:
                descriptor_start = index + 2
            elif tail[index] in {"from", "in", "on", "per"}:
                descriptor_start = index + 1
            else:
                continue
            prefix = tail[:index]
            descriptor = tail[descriptor_start:]
            if not descriptor or len(descriptor) > 4:
                continue
            if not all(token in core_tokens or token.isdigit() for token in prefix):
                continue
            allowed_descriptor = (
                self._SOURCE_ATTRIBUTION_WORDS
                | self._GENERIC_COMPANY_WORDS
                | self._subject_tokens
                | self._PERIOD_LINKAGE_WORDS
            )
            if all(
                token in allowed_descriptor or self._is_calendar_year(token) for token in descriptor
            ):
                return True
        return False

    @classmethod
    def _is_single_statement(cls, quote: str) -> bool:
        candidate = cls._mask_nonterminal_periods(quote)
        candidate = candidate.strip().rstrip(".!?")
        return not re.search(r"[.!?;\n]", candidate)

    def _claim_semantics(self, claim: str) -> tuple[str, bool]:
        tokens = frozenset(self._tokens(claim))
        matches = [
            (len(tokens & vocabulary), index, key, vocabulary, is_set)
            for index, (key, vocabulary, is_set) in enumerate(self._CLAIM_TAXONOMY)
            if tokens & vocabulary
        ]
        if not matches:
            context = sorted(tokens - self._subject_tokens - self._CLAIM_STOPWORDS)
            return " ".join(context), False
        _, _, key, vocabulary, is_set = max(matches, key=lambda item: (item[0], -item[1]))
        taxonomy_words = frozenset().union(*(entry[1] for entry in self._CLAIM_TAXONOMY))
        context = sorted(
            tokens - vocabulary - taxonomy_words - self._CLAIM_STOPWORDS - self._subject_tokens
        )
        is_set_valued = is_set and not bool(tokens & self._SCALAR_QUALIFIERS)
        return f"{key}:{' '.join(context)}" if context else key, is_set_valued

    def _claim_semantics_for_quotes(
        self,
        claim: str,
        value: Any,
        quotes: Sequence[str],
    ) -> tuple[str, bool] | None:
        """Canonicalize model labels with qualifiers stated by every quote.

        The model may shorten ``Annual revenue`` to ``Revenue``. Period
        qualifiers are therefore recovered from the quoted statement before
        claim IDs and conflict groups are built. If one model claim combines
        differently qualified statements, reject it rather than merging them.
        """

        predicate_key, is_set_valued = self._claim_semantics(claim)
        category, separator, context_text = predicate_key.partition(":")
        claim_context = set(context_text.split()) if separator else set()
        canonical_keys: set[str] = set()
        for quote in quotes:
            context = claim_context | self._quote_period_qualifiers(quote, value)
            canonical_keys.add(f"{category}:{' '.join(sorted(context))}" if context else category)
        if len(canonical_keys) != 1:
            return None
        return canonical_keys.pop(), is_set_valued

    def _quote_period_qualifiers(self, quote: str, value: Any) -> set[str]:
        values = value if isinstance(value, list) else [value]
        value_tokens = {
            token for item in values if item is not None for token in self._tokens(str(item))
        }
        return {
            token
            for token in self._tokens(quote)
            if token not in value_tokens
            and token not in self._subject_tokens
            and (token in self._PERIOD_PREDICATE_QUALIFIERS or self._is_calendar_year(token))
        }

    @staticmethod
    def _is_calendar_year(token: str) -> bool:
        return len(token) == 4 and token.isdigit() and 1900 <= int(token) <= 2100

    def _quote_supports_claim(self, quote: str, claim: str) -> bool:
        quote_tokens = frozenset(self._tokens(quote))
        claim_tokens = frozenset(self._tokens(claim))
        predicate_key, _ = self._claim_semantics(claim)
        category = predicate_key.split(":", 1)[0]
        taxonomy = next(
            (vocabulary for key, vocabulary, _ in self._CLAIM_TAXONOMY if key == category),
            frozenset(),
        )
        all_taxonomy_words = frozenset().union(*(entry[1] for entry in self._CLAIM_TAXONOMY))
        context = (
            claim_tokens
            - taxonomy
            - all_taxonomy_words
            - self._CLAIM_STOPWORDS
            - self._subject_tokens
        )
        predicate_matches = (
            bool(quote_tokens & taxonomy)
            if taxonomy
            else bool(quote_tokens & (claim_tokens - self._CLAIM_STOPWORDS))
        )
        return predicate_matches and context.issubset(quote_tokens)

    def _claim_context_tokens(
        self,
        claim: str,
        taxonomy: frozenset[str],
    ) -> frozenset[str]:
        claim_tokens = frozenset(self._tokens(claim))
        all_taxonomy_words = frozenset().union(*(entry[1] for entry in self._CLAIM_TAXONOMY))
        return (
            claim_tokens
            - taxonomy
            - all_taxonomy_words
            - self._CLAIM_STOPWORDS
            - self._subject_tokens
        )

    def _claim_context_sequence(
        self,
        claim: str,
        taxonomy: frozenset[str],
    ) -> tuple[str, ...]:
        context = self._claim_context_tokens(claim, taxonomy)
        return tuple(token for token in self._tokens(claim) if token in context)

    def _quote_supports_relation(self, quote: str, claim: str, value: Any) -> bool:
        if isinstance(value, list):
            return bool(value) and all(
                self._quote_supports_relation(quote, claim, item) for item in value
            )
        if value is None or not self._quote_supports_value(quote, value):
            return False

        quote_tokens = self._tokens(quote)
        if set(quote_tokens) & (self._NEGATION_WORDS | self._NONFACTUAL_WORDS):
            return False
        value_tokens = self._tokens(str(value))
        if not value_tokens:
            return False
        value_ranges = self._sequence_ranges(quote_tokens, value_tokens)
        if not value_ranges:
            return False

        predicate_key, _ = self._claim_semantics(claim)
        category = predicate_key.split(":", 1)[0]
        taxonomy = next(
            (vocabulary for key, vocabulary, _ in self._CLAIM_TAXONOMY if key == category),
            frozenset(),
        )
        claim_tokens = frozenset(self._tokens(claim))
        anchors = (taxonomy or claim_tokens) - frozenset(value_tokens)
        anchor_positions = [index for index, token in enumerate(quote_tokens) if token in anchors]
        context_tokens = self._claim_context_tokens(claim, taxonomy)
        context_sequence = self._claim_context_sequence(claim, taxonomy)
        allowed_connectors = (
            self._RELATION_GLUE
            | taxonomy
            | self._CLAIM_STOPWORDS
            | self._GENERIC_COMPANY_WORDS
            | self._subject_tokens
            | context_tokens
        )
        for value_start, value_end in value_ranges:
            for anchor in anchor_positions:
                if not self._context_is_bound_to_relation(
                    quote_tokens,
                    context_sequence,
                    anchor,
                    value_start,
                ):
                    continue
                if anchor < value_start:
                    between = quote_tokens[anchor + 1 : value_start]
                elif anchor >= value_end:
                    between = quote_tokens[value_end:anchor]
                else:
                    continue
                if (
                    len(between) <= 4
                    and not (set(between) & self._NEGATION_WORDS)
                    and all(token in allowed_connectors or token.isdigit() for token in between)
                ):
                    return True
        return False

    def _quote_supports_subject_relation(
        self,
        quote: str,
        claim: str,
        value: Any,
    ) -> bool:
        """Require the target company to participate in the asserted relation."""

        if isinstance(value, list):
            return bool(value) and all(
                self._quote_supports_subject_relation(quote, claim, item) for item in value
            )
        if value is None:
            return False

        quote_tokens = self._tokens(quote)
        value_tokens = self._tokens(str(value))
        value_ranges = self._sequence_ranges(quote_tokens, value_tokens)
        if not value_ranges:
            return False

        predicate_key, _ = self._claim_semantics(claim)
        category = predicate_key.split(":", 1)[0]
        taxonomy = next(
            (vocabulary for key, vocabulary, _ in self._CLAIM_TAXONOMY if key == category),
            frozenset(),
        )
        claim_tokens = frozenset(self._tokens(claim))
        anchors = (taxonomy or claim_tokens) - frozenset(value_tokens)
        anchor_ranges = [
            (index, index + 1) for index, token in enumerate(quote_tokens) if token in anchors
        ]
        subject_ranges = [
            token_range
            for subject in self._subject_token_sequences
            for token_range in self._sequence_ranges(quote_tokens, list(subject))
            if token_range[0] == 0 or (token_range[0] == 1 and quote_tokens[0] == "the")
        ]
        if not subject_ranges or not anchor_ranges:
            return False

        context_tokens = self._claim_context_tokens(claim, taxonomy)
        context_sequence = self._claim_context_sequence(claim, taxonomy)
        allowed_linkage = (
            self._RELATION_GLUE
            | taxonomy
            | self._CLAIM_STOPWORDS
            | self._GENERIC_COMPANY_WORDS
            | self._subject_tokens
            | context_tokens
            | self._PERIOD_LINKAGE_WORDS
        )
        for subject_range in subject_ranges:
            for anchor_range in anchor_ranges:
                for value_range in value_ranges:
                    if not self._context_is_bound_to_relation(
                        quote_tokens,
                        context_sequence,
                        anchor_range[0],
                        value_range[0],
                    ):
                        continue
                    components = self._merge_ranges([subject_range, anchor_range, value_range])
                    if components[-1][1] - components[0][0] > 16:
                        continue
                    gaps = [
                        quote_tokens[left[1] : right[0]] for left, right in pairwise(components)
                    ]
                    if all(
                        len(gap) <= 6
                        and not (set(gap) & self._NEGATION_WORDS)
                        and all(token in allowed_linkage or token.isdigit() for token in gap)
                        for gap in gaps
                    ):
                        return True
        return False

    @staticmethod
    def _context_is_bound_to_relation(
        quote_tokens: list[str],
        context_tokens: Sequence[str],
        anchor: int,
        value_start: int,
    ) -> bool:
        """Bind ordered claim qualifiers to one unshadowed predicate occurrence."""

        if not context_tokens:
            return True
        if anchor >= value_start:
            return False
        width = len(context_tokens)
        context_start = anchor - width
        bound_before = (
            context_start >= 0
            and quote_tokens[context_start:anchor] == list(context_tokens)
            and (context_start == 0 or quote_tokens[context_start - 1] != quote_tokens[anchor])
        )
        bound_after = anchor + 1 + width <= value_start and quote_tokens[
            anchor + 1 : anchor + 1 + width
        ] == list(context_tokens)
        if not (bound_before or bound_after):
            return False
        return quote_tokens[anchor] not in quote_tokens[anchor + 1 : value_start]

    def _source_still_supports(
        self,
        record: _ClaimRecord,
        source: SourceDocument,
    ) -> bool:
        try:
            quote = self._supporting_quote(record, source.source_id)
        except RuntimeError:
            return False
        return self._quote_is_present(quote, source) and self._quote_supports_direct_claim(
            quote,
            record.claim,
            record.value,
        )

    @classmethod
    def _identifier_tokens(cls, identifier: str) -> tuple[str, ...]:
        tokens = tuple(cls._tokens(identifier))
        distinctive = tuple(token for token in tokens if token not in cls._GENERIC_COMPANY_WORDS)
        return distinctive or tokens

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", normalized_fingerprint_text(value))

    @classmethod
    def _value_contains_url(cls, value: Any) -> bool:
        if isinstance(value, str):
            return re.search(r"(?i)(?:https?://|\bwww\.)", value) is not None
        if isinstance(value, list):
            return any(cls._value_contains_url(item) for item in value)
        return False

    @staticmethod
    def _sequence_ranges(tokens: list[str], needle: list[str]) -> list[tuple[int, int]]:
        width = len(needle)
        return [
            (index, index + width)
            for index in range(len(tokens) - width + 1)
            if tokens[index : index + width] == needle
        ]

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _supporting_quote(self, record: _ClaimRecord, source_id: str) -> str:
        candidates = [
            quote
            for recorded_source_id, quote in record.supporting_quotes_by_source.items()
            if self._registry.resolve_id(recorded_source_id) == self._registry.resolve_id(source_id)
        ]
        if not candidates:
            raise RuntimeError("accepted evidence lost its supporting quote")
        return min(candidates, key=lambda quote: (len(quote), quote.casefold()))

    def _document_chunks(self, documents: list[SourceDocument]) -> list[list[SourceDocument]]:
        chunks: list[list[SourceDocument]] = []
        current: list[SourceDocument] = []
        current_bytes = 0
        for document in documents:
            size = len(
                json.dumps(self._source_payload(document), sort_keys=True, default=str).encode(
                    "utf-8"
                )
            )
            if current and (
                len(current) >= self._extraction_batch_size
                or current_bytes + size > self._extraction_max_prompt_bytes
            ):
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(document)
            current_bytes += size
        if current:
            chunks.append(current)
        return chunks

    def _source_payload(self, document: SourceDocument) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": document.source_id,
            "title": document.title,
            "url": str(document.url),
            "published_at": document.published_at.isoformat() if document.published_at else None,
            "source_type": document.source_type.value,
            "content": "",
            "content_hash": document.content_hash,
        }
        metadata_bytes = len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        available_content_bytes = max(
            0,
            self._extraction_max_prompt_bytes - metadata_bytes - 256,
        )
        encoded_content = document.content.encode("utf-8")[:available_content_bytes]
        payload["content"] = encoded_content.decode("utf-8", errors="ignore")
        return payload

    def resolve_source_id(self, source_id: str) -> str:
        """Resolve a historical registry alias for final trace reconciliation."""
        return self._registry.resolve_id(source_id)

    @staticmethod
    def _evidence_id(claim_id: str, source_id: str | None) -> str:
        return stable_hash(
            {"claim_id": claim_id, "source_id": source_id},
            prefix="ev_",
            length=24,
        )

    @classmethod
    def _value_fingerprint(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalized_fingerprint_text(value)
        if isinstance(value, Mapping):
            return {
                str(key): cls._value_fingerprint(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [cls._value_fingerprint(item) for item in value]
        return value

    @classmethod
    def _preferred_value(cls, first: Any, second: Any) -> Any:
        first_serialized = json.dumps(first, sort_keys=True, default=str)
        second_serialized = json.dumps(second, sort_keys=True, default=str)
        return first if first_serialized <= second_serialized else second
