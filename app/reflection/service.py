import json
import re
from collections import defaultdict
from datetime import timedelta

from app.config import Settings
from app.core.clock import SystemClock
from app.core.enums import AssessmentDecision, EvidenceStatus
from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import Clock, LLMClient, PromptRepository
from app.planner.service import ResearchPlanner
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
from app.utils.text import normalized_fingerprint_text

ASSESSMENT_DIMENSIONS = (
    "Company Identity",
    "Products",
    "Technology",
    "Pipeline",
    "Recent News",
    "Financial Signals",
    "Procurement Signals",
    "Cooperation Signals",
    "Evidence Quality",
    "Evidence Freshness",
    "Evidence Diversity",
)


def evidence_statement_for_claim_ids(
    claim_ids: list[str],
    evidence_by_id: dict[str, list[Evidence]],
) -> str:
    """Render cited evidence without retaining model-authored prose."""

    parts: list[str] = []
    for claim_id in claim_ids:
        representative = min(
            evidence_by_id[claim_id],
            key=lambda item: item.evidence_id,
        )
        if representative.status == EvidenceStatus.UNKNOWN:
            part = f"Unknown: {representative.claim}"
        else:
            value = str(representative.value)
            part = (
                representative.claim
                if "http://" in value.casefold() or "https://" in value.casefold()
                else f"{representative.claim}: {value}"
            )
            if representative.status == EvidenceStatus.INFERENCE:
                part = f"Inference: {part}"
        parts.append(part)
    return "; ".join(dict.fromkeys(parts))


def empty_assessment(reason: str) -> InformationAssessment:
    return InformationAssessment(
        dimensions=[
            DimensionAssessment(
                dimension=name,
                coverage_score=0.0,
                confidence=0.0,
                missing_items=[f"No assessment evidence is available for {name}."],
            )
            for name in ASSESSMENT_DIMENSIONS
        ],
        evidence_coverage=0.0,
        weak_evidence=[],
        conflicts=[],
        missing_information=[reason],
        recommended_followup_queries=[],
        decision=AssessmentDecision.CONTINUE_SEARCH,
        decision_reason=reason,
    )


_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Company Identity": ("identity", "company name", "headquarter", "founded", "website"),
    "Products": ("product", "drug", "medicine", "therapy"),
    "Technology": ("technology", "platform", "modality", "technical"),
    "Pipeline": ("pipeline", "clinical", "trial", "phase"),
    "Recent News": ("news", "announced", "recent", "launch", "approval"),
    "Financial Signals": ("financial", "funding", "revenue", "financing", "capital"),
    "Procurement Signals": ("procurement", "purchas", "supplier", "capacity", "facility"),
    "Cooperation Signals": (
        "partner",
        "partnership",
        "collaboration",
        "licensing",
        "agreement",
        "alliance",
    ),
    "Evidence Quality": ("evidence", "quality", "official", "regulatory", "primary"),
    "Evidence Freshness": ("evidence", "freshness", "latest", "current", "dated"),
    "Evidence Diversity": (
        "evidence",
        "diversity",
        "independent",
        "sources",
        "corroboration",
    ),
}
_FOLLOWUP_QUERY_TERMS: dict[str, frozenset[str]] = {
    "Company Identity": frozenset(
        {"identity", "corporate", "official", "website", "headquarters", "incorporation"}
    ),
    "Products": frozenset(
        {"product", "products", "drug", "drugs", "medicine", "medicines", "therapy", "portfolio"}
    ),
    "Technology": frozenset({"technology", "technical", "platform", "modality", "biotechnology"}),
    "Pipeline": frozenset({"pipeline", "clinical", "trial", "phase", "development", "stage"}),
    "Recent News": frozenset(
        {"news", "announced", "announcement", "recent", "launch", "approval", "press", "regulatory"}
    ),
    "Financial Signals": frozenset(
        {"financial", "funding", "revenue", "financing", "capital", "investor", "earnings"}
    ),
    "Procurement Signals": frozenset(
        {
            "procurement",
            "purchase",
            "purchasing",
            "supplier",
            "capacity",
            "facility",
            "manufacturing",
        }
    ),
    "Cooperation Signals": frozenset(
        {"partner", "partnership", "collaboration", "licensing", "agreement", "alliance"}
    ),
    "Evidence Quality": frozenset(
        {"evidence", "quality", "official", "regulatory", "primary", "source"}
    ),
    "Evidence Freshness": frozenset(
        {"evidence", "freshness", "latest", "current", "dated", "updated"}
    ),
    "Evidence Diversity": frozenset(
        {"evidence", "diversity", "independent", "sources", "corroboration"}
    ),
}
_SAFE_FOLLOWUP_QUERY_SUFFIX: dict[str, str] = {
    "Company Identity": "company identity official website headquarters",
    "Products": "pharmaceutical products drug medicine portfolio",
    "Technology": "pharmaceutical technology platform modality",
    "Pipeline": "drug pipeline clinical trial phase stage",
    "Recent News": "company news press announcement regulatory approval",
    "Financial Signals": "company financial revenue funding financing",
    "Procurement Signals": "pharmaceutical procurement supplier manufacturing capacity",
    "Cooperation Signals": "pharmaceutical partnership licensing collaboration agreement",
    "Evidence Quality": "official regulatory primary evidence sources company profile",
    "Evidence Freshness": "latest current dated company evidence updates",
    "Evidence Diversity": "independent regulatory industry evidence sources corroboration",
}
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
        "therapeutics",
    }
)


class ReflectionService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRepository,
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._settings = settings
        self._clock = clock or SystemClock()

    async def summarize(
        self,
        company: ResolvedCompany,
        bundle: EvidenceBundle,
    ) -> ResearchSummary:
        prompt = self._prompts.render(
            "reflection",
            {
                "resolved_company": "Supplied in the structured user JSON payload.",
                "evidence_bundle": "Supplied in the structured user JSON payload.",
                "prior_queries": "Supplied in the structured user JSON payload.",
            },
        )
        payload = {
            "task": "summarize",
            "resolved_company": company.model_dump(mode="json"),
            "evidence_bundle": bundle.model_dump(mode="json"),
            "prior_queries": [],
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=ResearchSummary,
                cache_namespace="research-summary",
            )
            proposed = result.value
        except BudgetExceededError:
            proposed = self._fallback_summary(bundle)
        except StructuredOutputError:
            proposed = self._fallback_summary(bundle)
        return self._sanitize_summary(proposed, bundle)

    async def assess(
        self,
        company: ResolvedCompany,
        bundle: EvidenceBundle,
        summary: ResearchSummary,
        prior_queries: list[str],
    ) -> InformationAssessment:
        prompt = self._prompts.render(
            "reflection",
            {
                "resolved_company": "Supplied in the structured user JSON payload.",
                "evidence_bundle": "Supplied in the structured user JSON payload.",
                "prior_queries": "Supplied in the structured user JSON payload.",
            },
        )
        payload = {
            "task": "assess",
            "resolved_company": company.model_dump(mode="json"),
            "evidence_bundle": bundle.model_dump(mode="json"),
            "research_summary": summary.model_dump(mode="json"),
            "prior_queries": prior_queries,
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=InformationAssessment,
                cache_namespace="information-assessment",
            )
            proposed = result.value
        except BudgetExceededError:
            proposed = self._fallback_assessment(summary)
        except StructuredOutputError:
            proposed = self._fallback_assessment(summary)
        return self._enforce_assessment_policy(proposed, bundle, summary)

    def reconcile_after_identity_constraint(
        self,
        bundle: EvidenceBundle,
    ) -> tuple[ResearchSummary, InformationAssessment]:
        """Rebuild public analytical products after final entity-scope filtering."""

        summary = self._sanitize_summary(ResearchSummary(), bundle)
        ceiling = InformationAssessment(
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
            decision_reason="Candidate assessment pending deterministic policy checks.",
        )
        assessment = self._enforce_assessment_policy(ceiling, bundle, summary)
        return summary, assessment

    async def generate_followup(
        self,
        company: ResolvedCompany,
        assessment: InformationAssessment,
        prior_queries: list[str],
        prior_query_fingerprints: set[str],
    ) -> FollowupPlan:
        prompt = self._prompts.render(
            "followup",
            {
                "resolved_company": "Supplied in the structured user JSON payload.",
                "assessment": "Supplied in the structured user JSON payload.",
                "prior_queries": "Supplied in the structured user JSON payload.",
            },
        )
        payload = {
            "resolved_company": company.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
            "prior_queries": prior_queries,
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=FollowupPlan,
                cache_namespace="followup",
            )
            proposed = result.value
        except BudgetExceededError:
            return FollowupPlan(queries=[], addressed_missing_items=[])
        except StructuredOutputError:
            proposed = FollowupPlan()

        missing = set(assessment.missing_information)
        for dimension_assessment in assessment.dimensions:
            missing.update(dimension_assessment.missing_items)
        missing_by_dimension: dict[str, list[str]] = defaultdict(list)
        for item in sorted(missing):
            missing_dimension = self._missing_item_dimension(item)
            if missing_dimension is not None:
                missing_by_dimension[missing_dimension].append(item)

        prioritized_dimensions: list[str] = []
        for query in proposed.queries:
            query_dimension, _ = self._infer_addressed_items(query, missing, company)
            if (
                query_dimension is not None
                and query_dimension in missing_by_dimension
                and query_dimension not in prioritized_dimensions
            ):
                prioritized_dimensions.append(query_dimension)
        prioritized_dimensions.extend(
            dimension
            for dimension in ASSESSMENT_DIMENSIONS
            if dimension in missing_by_dimension and dimension not in prioritized_dimensions
        )

        candidates = [
            SearchQuery(
                query_id="pending",
                query=(f"{company.canonical_name} {_SAFE_FOLLOWUP_QUERY_SUFFIX[query_dimension]}"),
                topic=query_dimension,
                priority=5,
                expected_evidence=(
                    f"Supported {query_dimension.casefold()} information for "
                    f"{company.canonical_name}."
                ),
                addresses_missing_items=missing_by_dimension[query_dimension],
            )
            for query_dimension in prioritized_dimensions
        ]
        prepared = ResearchPlanner.prepare_queries(candidates, prior_query_fingerprints)
        selected = prepared[: self._settings.max_queries_per_round]
        return FollowupPlan(
            queries=selected,
            addressed_missing_items=sorted(
                {item for query in selected for item in query.addresses_missing_items}
            ),
        )

    @classmethod
    def _infer_addressed_items(
        cls,
        query: SearchQuery,
        missing: set[str],
        company: ResolvedCompany,
    ) -> tuple[str | None, list[str]]:
        query_tokens = set(re.findall(r"[a-z0-9]+", normalized_fingerprint_text(query.query)))
        identifiers = [company.canonical_name, *company.aliases]
        targets_company = False
        identifier_query_tokens: set[str] = set()
        for identifier in identifiers:
            identifier_tokens = re.findall(r"[a-z0-9]+", identifier.casefold())
            distinctive = [
                token for token in identifier_tokens if token not in _GENERIC_COMPANY_WORDS
            ]
            required = distinctive or identifier_tokens
            if required and set(required).issubset(query_tokens):
                targets_company = True
                identifier_query_tokens.update(identifier_tokens)
                break
        if not targets_company:
            return None, []

        topical_tokens = query_tokens - identifier_query_tokens
        dimension_scores = {
            dimension: len(topical_tokens & terms)
            for dimension, terms in _FOLLOWUP_QUERY_TERMS.items()
        }
        best_score = max(dimension_scores.values(), default=0)
        best_dimensions = [
            dimension for dimension, score in dimension_scores.items() if score == best_score
        ]
        if best_score < 2 or len(best_dimensions) != 1:
            return None, []
        query_dimension = best_dimensions[0]

        addressed: list[str] = []
        for item in missing:
            if cls._missing_item_dimension(item) == query_dimension:
                addressed.append(item)
        return query_dimension, addressed

    @staticmethod
    def _missing_item_dimension(item: str) -> str | None:
        normalized_item = normalized_fingerprint_text(item)
        item_tokens = set(re.findall(r"[a-z0-9]+", normalized_item))
        for dimension in _FOLLOWUP_QUERY_TERMS:
            if normalized_fingerprint_text(dimension) in normalized_item:
                return dimension
        scores: dict[str, int] = {}
        for dimension, terms in _FOLLOWUP_QUERY_TERMS.items():
            scores[dimension] = len(item_tokens & terms)
            scores[dimension] += sum(
                1 for keyword in _DIMENSION_KEYWORDS[dimension] if keyword in normalized_item
            )
        best_score = max(scores.values(), default=0)
        best_dimensions = [dimension for dimension, score in scores.items() if score == best_score]
        return best_dimensions[0] if best_score > 0 and len(best_dimensions) == 1 else None

    def _enforce_assessment_policy(
        self,
        proposed: InformationAssessment,
        bundle: EvidenceBundle,
        summary: ResearchSummary,
    ) -> InformationAssessment:
        proposed_by_name = {item.dimension.casefold(): item for item in proposed.dimensions}
        dimensions: list[DimensionAssessment] = []
        for name in ASSESSMENT_DIMENSIONS:
            candidate = proposed_by_name.get(name.casefold())
            observed_coverage, observed_confidence = self._observed_score(name, bundle)
            if candidate is None:
                candidate = DimensionAssessment(
                    dimension=name,
                    coverage_score=0.0,
                    confidence=0.0,
                    missing_items=[f"No supported {name.lower()} information was found."],
                )
            coverage = min(candidate.coverage_score, observed_coverage)
            confidence = min(candidate.confidence, observed_confidence)
            dimension_target = max(
                self._settings.minimum_dimension_coverage,
                self._settings.sufficient_coverage_threshold,
            )
            missing_items = (
                [f"Additional evidence is required for {name}."]
                if coverage < dimension_target
                else []
            )
            dimensions.append(
                DimensionAssessment(
                    dimension=name,
                    coverage_score=coverage,
                    confidence=confidence,
                    missing_items=missing_items,
                )
            )

        coverage = sum(item.coverage_score for item in dimensions) / len(dimensions)
        meets_threshold = coverage >= self._settings.sufficient_coverage_threshold
        meets_minimums = all(
            item.coverage_score >= self._settings.minimum_dimension_coverage for item in dimensions
        )
        decision = (
            AssessmentDecision.STOP
            if meets_threshold and meets_minimums
            else AssessmentDecision.CONTINUE_SEARCH
        )
        missing = list(
            dict.fromkeys(
                [
                    *summary.missing_information,
                    *(item for dimension in dimensions for item in dimension.missing_items),
                ]
            )
        )
        return InformationAssessment(
            dimensions=dimensions,
            evidence_coverage=coverage,
            weak_evidence=self._deterministic_weak_evidence(bundle),
            conflicts=[conflict.description for conflict in bundle.conflicts],
            missing_information=missing,
            recommended_followup_queries=[],
            decision=decision,
            decision_reason=(
                "All deterministic coverage thresholds were met."
                if decision == AssessmentDecision.STOP
                else "One or more deterministic evidence coverage thresholds were not met."
            ),
        )

    def _observed_score(self, dimension: str, bundle: EvidenceBundle) -> tuple[float, float]:
        supported = [item for item in bundle.evidence if item.status != EvidenceStatus.UNKNOWN]
        if dimension == "Evidence Quality":
            if not supported:
                return 0.0, 0.0
            verified = sum(item.status == EvidenceStatus.VERIFIED_FACT for item in supported)
            ratio = verified / len(supported)
            return min(1.0, 0.35 + ratio * 0.65), min(0.95, 0.5 + ratio * 0.45)
        if dimension == "Evidence Diversity":
            domains = {item.source for item in supported if item.source}
            score = min(1.0, len(domains) / 4)
            return score, score
        if dimension == "Evidence Freshness":
            cutoff = self._clock.now() - timedelta(days=730)
            dated = [item for item in supported if item.published_at is not None]
            if not dated:
                return 0.0, 0.0
            recent = sum(item.published_at >= cutoff for item in dated if item.published_at)
            score = recent / len(dated)
            return score, min(0.9, 0.5 + 0.4 * score)

        keywords = _DIMENSION_KEYWORDS[dimension]
        matches = [
            item
            for item in supported
            if any(keyword in f"{item.claim} {item.value}".casefold() for keyword in keywords)
        ]
        if not matches:
            return 0.0, 0.0
        claim_ids = {item.claim_id for item in matches}
        verified = any(item.status == EvidenceStatus.VERIFIED_FACT for item in matches)
        sources = {item.source for item in matches if item.source}
        coverage = min(1.0, 0.35 + 0.15 * len(claim_ids) + (0.2 if verified else 0.0))
        confidence = min(0.95, 0.4 + 0.15 * len(sources) + (0.2 if verified else 0.0))
        return coverage, confidence

    def _sanitize_summary(
        self,
        proposed: ResearchSummary,
        bundle: EvidenceBundle,
    ) -> ResearchSummary:
        evidence_by_id: dict[str, list[Evidence]] = defaultdict(list)
        for item in bundle.evidence:
            evidence_by_id[item.claim_id].append(item)
        known: list[KnownFact] = []
        for fact in proposed.known_facts:
            valid_ids = list(
                dict.fromkeys(
                    claim_id
                    for claim_id in fact.claim_ids
                    if claim_id in evidence_by_id
                    and all(
                        item.status != EvidenceStatus.UNKNOWN for item in evidence_by_id[claim_id]
                    )
                )
            )
            if valid_ids:
                known.append(
                    fact.model_copy(
                        update={
                            "claim_ids": valid_ids,
                            "statement": evidence_statement_for_claim_ids(
                                valid_ids,
                                evidence_by_id,
                            ),
                        }
                    )
                )
        represented_claim_ids = {claim_id for fact in known for claim_id in fact.claim_ids}
        for claim_id in sorted(evidence_by_id):
            if claim_id in represented_claim_ids or any(
                item.status == EvidenceStatus.UNKNOWN for item in evidence_by_id[claim_id]
            ):
                continue
            known.append(
                KnownFact(
                    statement=evidence_statement_for_claim_ids(
                        [claim_id],
                        evidence_by_id,
                    ),
                    claim_ids=[claim_id],
                )
            )
        return proposed.model_copy(
            update={
                "known_facts": known,
                "missing_information": self._deterministic_missing_information(bundle),
                "conflicts": [conflict.description for conflict in bundle.conflicts],
                "weak_evidence": self._deterministic_weak_evidence(bundle),
                "unknowns": sorted(
                    {
                        item.claim
                        for item in bundle.evidence
                        if item.status == EvidenceStatus.UNKNOWN
                    }
                ),
            }
        )

    def _deterministic_missing_information(self, bundle: EvidenceBundle) -> list[str]:
        target = max(
            self._settings.minimum_dimension_coverage,
            self._settings.sufficient_coverage_threshold,
        )
        return [
            f"Additional evidence is required for {dimension}."
            for dimension in ASSESSMENT_DIMENSIONS
            if self._observed_score(dimension, bundle)[0] < target
        ]

    @staticmethod
    def _deterministic_weak_evidence(bundle: EvidenceBundle) -> list[str]:
        return sorted(
            {
                item.claim
                for item in bundle.evidence
                if item.status in {EvidenceStatus.SINGLE_SOURCE, EvidenceStatus.INFERENCE}
            }
        )

    @staticmethod
    def _fallback_summary(bundle: EvidenceBundle) -> ResearchSummary:
        facts: list[KnownFact] = []
        seen: set[str] = set()
        for item in bundle.evidence:
            if item.status == EvidenceStatus.UNKNOWN or item.claim_id in seen:
                continue
            seen.add(item.claim_id)
            facts.append(KnownFact(statement=item.claim, claim_ids=[item.claim_id]))
        weak = sorted(
            {
                item.claim
                for item in bundle.evidence
                if item.status in {EvidenceStatus.SINGLE_SOURCE, EvidenceStatus.INFERENCE}
            }
        )
        return ResearchSummary(
            known_facts=facts,
            missing_information=[],
            conflicts=[conflict.description for conflict in bundle.conflicts],
            weak_evidence=weak,
            unknowns=[],
        )

    @staticmethod
    def _fallback_assessment(summary: ResearchSummary) -> InformationAssessment:
        base = empty_assessment("Structured assessment was unavailable.")
        return base.model_copy(
            update={
                "weak_evidence": summary.weak_evidence,
                "conflicts": summary.conflicts,
                "missing_information": summary.missing_information,
            }
        )
