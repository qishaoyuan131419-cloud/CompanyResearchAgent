import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.core.enums import DimensionStatus, EvidenceStatus, GapReason, StopReason
from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import Clock, LLMClient, PromptRepository
from app.llm.errors import LLMProviderError
from app.reflection.service import evidence_statement_for_claim_ids
from app.schemas.company import ResolvedCompany
from app.schemas.evidence import Evidence, EvidenceBundle
from app.schemas.reflection import InformationAssessment
from app.schemas.research import (
    ResearchFinding,
    ResearchGap,
    ResearchGapSection,
    ResearchReport,
    ResearchSection,
)
from app.utils.hashing import stable_hash


@dataclass(frozen=True, slots=True)
class FinalizedResearch:
    report: ResearchReport
    evidence_bundle: EvidenceBundle


class ResearchFinalizer:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRepository,
        clock: Clock | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._clock = clock

    async def finalize(
        self,
        company: ResolvedCompany,
        bundle: EvidenceBundle,
        assessment: InformationAssessment,
        stop_reason: StopReason | None = None,
    ) -> FinalizedResearch:
        evidence_bundle = bundle
        prompt = self._prompts.render(
            "research",
            {
                "resolved_company": "Supplied in the structured user JSON payload.",
                "evidence_bundle": "Supplied in the structured user JSON payload.",
                "assessment": "Supplied in the structured user JSON payload.",
            },
        )
        payload = {
            "resolved_company": company.model_dump(mode="json"),
            "evidence_bundle": evidence_bundle.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=ResearchReport,
                cache_namespace="final-research",
            )
            report = self._sanitize_report(result.value, evidence_bundle)
        except (BudgetExceededError, LLMProviderError, StructuredOutputError):
            report = self._fallback_report(evidence_bundle, assessment)
        report = self._ensure_supported_findings(report, evidence_bundle)
        report = self._ensure_unknown_section(report, evidence_bundle, assessment, stop_reason)
        report = self._ensure_conflicts(report, evidence_bundle)
        report = self._ensure_recommendations(report, evidence_bundle)
        evidence_bundle = evidence_bundle.model_copy(
            update={"research_gap_count": len(report.unknowns.gaps)}
        )
        return FinalizedResearch(report=report, evidence_bundle=evidence_bundle)

    @staticmethod
    def _sanitize_report(report: ResearchReport, bundle: EvidenceBundle) -> ResearchReport:
        statuses_by_id: dict[str, set[EvidenceStatus]] = defaultdict(set)
        evidence_by_id: dict[str, list[Evidence]] = defaultdict(list)
        for item in bundle.evidence:
            statuses_by_id[item.claim_id].add(item.status)
            evidence_by_id[item.claim_id].append(item)
        routed_findings: dict[str, list[ResearchFinding]] = defaultdict(list)
        seen_claim_ids: set[str] = set()
        for section_name in type(report).model_fields:
            if section_name == "unknowns":
                continue
            for finding in getattr(report, section_name).findings:
                for claim_id in finding.claim_ids:
                    if claim_id in seen_claim_ids or claim_id not in statuses_by_id:
                        continue
                    statuses = statuses_by_id[claim_id]
                    if EvidenceStatus.UNKNOWN not in statuses:
                        representative = min(
                            evidence_by_id[claim_id],
                            key=lambda item: item.evidence_id,
                        )
                        target_section = ResearchFinalizer._section_for_claim(representative.claim)
                    else:
                        continue
                    seen_claim_ids.add(claim_id)
                    routed_findings[target_section].append(
                        ResearchFinding(
                            statement=evidence_statement_for_claim_ids(
                                [claim_id],
                                evidence_by_id,
                            ),
                            claim_ids=[claim_id],
                        )
                    )
        return ResearchReport(
            **{
                section_name: (
                    ResearchGapSection()
                    if section_name == "unknowns"
                    else ResearchSection(findings=routed_findings[section_name])
                )
                for section_name in type(report).model_fields
            }
        )

    @staticmethod
    def _statement_for_claim_ids(
        claim_ids: list[str],
        evidence_by_id: dict[str, list[Evidence]],
    ) -> str:
        return evidence_statement_for_claim_ids(claim_ids, evidence_by_id)

    @staticmethod
    def _fallback_report(
        bundle: EvidenceBundle,
        assessment: InformationAssessment,
    ) -> ResearchReport:
        del assessment
        sections: dict[str, list[ResearchFinding]] = defaultdict(list)
        seen: set[str] = set()
        for item in bundle.evidence:
            if item.claim_id in seen:
                continue
            seen.add(item.claim_id)
            section = ResearchFinalizer._section_for_claim(item.claim)
            value = str(item.value)
            statement = (
                item.claim
                if "http://" in value.casefold() or "https://" in value.casefold()
                else f"{item.claim}: {value}"
            )
            if item.status == EvidenceStatus.INFERENCE:
                statement = f"Inference: {statement}"
            sections[section].append(
                ResearchFinding(statement=statement, claim_ids=[item.claim_id])
            )

        for conflict in bundle.conflicts:
            sections["risks"].append(
                ResearchFinding(
                    statement=f"Conflicting evidence: {conflict.description}",
                    claim_ids=conflict.claim_ids,
                )
            )
        payload: dict[str, Any] = {
            field: (
                ResearchGapSection()
                if field == "unknowns"
                else ResearchSection(findings=sections[field])
            )
            for field in ResearchReport.model_fields
        }
        return ResearchReport(**payload)

    @staticmethod
    def _section_for_claim(claim: str) -> str:
        normalized = claim.casefold()
        mapping = (
            (("product", "drug", "medicine", "commercial"), "products"),
            (("technology", "platform", "modality"), "technology"),
            (("pipeline", "clinical", "trial", "phase"), "pipeline"),
            (("manufactur", "facility", "production"), "manufacturing"),
            (("news", "announc", "recent", "approval"), "recent_news"),
            (("fund", "financial", "revenue", "investment"), "financial_signals"),
            (("supply", "supplier", "logistics"), "supply_chain"),
            (("procure", "purchas", "capacity"), "potential_procurement_signals"),
            (("department", "contact", "executive", "officer"), "target_departments"),
            (("risk", "warning", "closed", "competition"), "risks"),
        )
        for keywords, section in mapping:
            if any(keyword in normalized for keyword in keywords):
                return section
        return "overview"

    @staticmethod
    def _ensure_unknown_section(
        report: ResearchReport,
        bundle: EvidenceBundle,
        assessment: InformationAssessment,
        stop_reason: StopReason | None,
    ) -> ResearchReport:
        reason_by_status = {
            DimensionStatus.NOT_SEARCHED: GapReason.NOT_SEARCHED,
            DimensionStatus.SEARCH_FAILED: GapReason.SEARCH_FAILED,
            DimensionStatus.SEARCHED_NO_RESULTS: GapReason.SEARCHED_NO_RESULTS,
            DimensionStatus.CONTENT_RETRIEVAL_FAILED: GapReason.CONTENT_RETRIEVAL_FAILED,
            DimensionStatus.EXTRACTION_FAILED: GapReason.EXTRACTION_FAILED,
            DimensionStatus.CONFLICTING_EVIDENCE: GapReason.SOURCE_CONFLICT,
        }
        gaps: list[ResearchGap] = []
        seen: set[tuple[str, str]] = set()
        for dimension in assessment.dimensions:
            descriptions = dimension.missing_items or (
                [f"Evidence remains insufficient for {dimension.dimension}."]
                if dimension.coverage_score < 1.0
                else []
            )
            for description in descriptions:
                key = (dimension.dimension, description)
                if key in seen:
                    continue
                seen.add(key)
                budget_reasons = {
                    StopReason.QUERY_BUDGET_REACHED,
                    StopReason.TOKEN_BUDGET_REACHED,
                    StopReason.COST_BUDGET_REACHED,
                    StopReason.TIME_BUDGET_REACHED,
                    StopReason.SOURCE_BUDGET_REACHED,
                }
                reason = (
                    GapReason.BUDGET_INTERRUPTED
                    if stop_reason in budget_reasons
                    else reason_by_status.get(
                        dimension.status,
                        GapReason.INSUFFICIENT_EVIDENCE,
                    )
                )
                gaps.append(
                    ResearchGap(
                        gap_id=stable_hash(
                            {"dimension": dimension.dimension, "description": description},
                            prefix="gap_",
                            length=24,
                        ),
                        dimension=dimension.dimension,
                        description=description,
                        reason=reason,
                        attempted_query_ids=dimension.searched_query_ids,
                        source_ids=dimension.source_ids,
                        processing_error_ids=[
                            error.error_id
                            for error in bundle.processing_errors
                            if (
                                dimension.status == DimensionStatus.EXTRACTION_FAILED
                                and error.stage.value in {"extraction", "validation"}
                            )
                            or (
                                dimension.status == DimensionStatus.SEARCH_FAILED
                                and error.stage.value == "search"
                            )
                        ],
                    )
                )
        return report.model_copy(update={"unknowns": ResearchGapSection(gaps=gaps)})

    @staticmethod
    def _ensure_supported_findings(
        report: ResearchReport,
        bundle: EvidenceBundle,
    ) -> ResearchReport:
        represented = {
            claim_id
            for section_name in type(report).model_fields
            if section_name not in {"unknowns", "recommendations"}
            for finding in getattr(report, section_name).findings
            for claim_id in finding.claim_ids
        }
        evidence_by_id: dict[str, list[Evidence]] = defaultdict(list)
        for item in bundle.evidence:
            evidence_by_id[item.claim_id].append(item)

        additions: dict[str, list[ResearchFinding]] = defaultdict(list)
        for claim_id in sorted(evidence_by_id):
            items = evidence_by_id[claim_id]
            if claim_id in represented or any(
                item.status == EvidenceStatus.UNKNOWN for item in items
            ):
                continue
            representative = min(items, key=lambda item: item.evidence_id)
            section_name = ResearchFinalizer._section_for_claim(representative.claim)
            additions[section_name].append(
                ResearchFinding(
                    statement=evidence_statement_for_claim_ids(
                        [claim_id],
                        evidence_by_id,
                    ),
                    claim_ids=[claim_id],
                )
            )

        if not additions:
            return report
        updates = {
            section_name: ResearchSection(
                findings=[
                    *getattr(report, section_name).findings,
                    *section_additions,
                ]
            )
            for section_name, section_additions in additions.items()
        }
        return report.model_copy(update=updates)

    @staticmethod
    def _ensure_conflicts(
        report: ResearchReport,
        bundle: EvidenceBundle,
    ) -> ResearchReport:
        represented_sets = [set(finding.claim_ids) for finding in report.risks.findings]
        additions = [
            ResearchFinding(
                statement=f"Conflicting evidence: {conflict.description}",
                claim_ids=conflict.claim_ids,
            )
            for conflict in bundle.conflicts
            if not any(set(conflict.claim_ids).issubset(ids) for ids in represented_sets)
        ]
        if not additions:
            return report
        return report.model_copy(
            update={"risks": ResearchSection(findings=[*report.risks.findings, *additions])}
        )

    @staticmethod
    def _ensure_recommendations(
        report: ResearchReport,
        bundle: EvidenceBundle,
    ) -> ResearchReport:
        recommendations: list[ResearchFinding] = []
        seen: set[str] = set()
        for conflict in bundle.conflicts:
            key = "|".join(conflict.claim_ids)
            if key in seen:
                continue
            seen.add(key)
            recommendations.append(
                ResearchFinding(
                    statement=f"Reconcile the conflicting evidence for {conflict.claim}.",
                    claim_ids=conflict.claim_ids,
                )
            )
        for item in bundle.evidence:
            if item.status != EvidenceStatus.UNKNOWN or item.claim_id in seen:
                continue
            seen.add(item.claim_id)
            recommendations.append(
                ResearchFinding(
                    statement=f"Perform targeted research for the unresolved item: {item.claim}.",
                    claim_ids=[item.claim_id],
                )
            )
        return report.model_copy(
            update={"recommendations": ResearchSection(findings=recommendations)}
        )
