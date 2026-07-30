import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.core.clock import SystemClock
from app.core.enums import EvidenceStatus
from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import Clock, LLMClient, PromptRepository
from app.reflection.service import evidence_statement_for_claim_ids
from app.schemas.company import ResolvedCompany
from app.schemas.evidence import Evidence, EvidenceBundle
from app.schemas.reflection import InformationAssessment
from app.schemas.research import ResearchFinding, ResearchReport, ResearchSection
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
        self._clock = clock or SystemClock()

    async def finalize(
        self,
        company: ResolvedCompany,
        bundle: EvidenceBundle,
        assessment: InformationAssessment,
    ) -> FinalizedResearch:
        bundle_with_unknowns = self._add_unknown_evidence(bundle, assessment)
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
            "evidence_bundle": bundle_with_unknowns.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=ResearchReport,
                cache_namespace="final-research",
            )
            report = self._sanitize_report(result.value, bundle_with_unknowns)
        except (BudgetExceededError, StructuredOutputError):
            report = self._fallback_report(bundle_with_unknowns, assessment)
        report = self._ensure_supported_findings(report, bundle_with_unknowns)
        report = self._ensure_unknown_section(report, bundle_with_unknowns)
        report = self._ensure_conflicts(report, bundle_with_unknowns)
        report = self._ensure_recommendations(report, bundle_with_unknowns)
        return FinalizedResearch(report=report, evidence_bundle=bundle_with_unknowns)

    def _add_unknown_evidence(
        self,
        bundle: EvidenceBundle,
        assessment: InformationAssessment,
    ) -> EvidenceBundle:
        existing_claim_ids = {item.claim_id for item in bundle.evidence}
        missing_items: list[str] = list(assessment.missing_information)
        for dimension in assessment.dimensions:
            missing_items.extend(dimension.missing_items)
        additions: list[Evidence] = []
        for missing in dict.fromkeys(item.strip() for item in missing_items if item.strip()):
            claim_id = stable_hash({"unknown": missing.casefold()}, prefix="clm_")
            if claim_id in existing_claim_ids:
                continue
            existing_claim_ids.add(claim_id)
            additions.append(
                Evidence(
                    evidence_id=stable_hash({"claim_id": claim_id, "source": None}, prefix="evd_"),
                    claim_id=claim_id,
                    claim=missing,
                    value="Unknown",
                    status=EvidenceStatus.UNKNOWN,
                    confidence=0.0,
                    retrieved_at=self._clock.now(),
                )
            )
        return bundle.model_copy(update={"evidence": [*bundle.evidence, *additions]})

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
            for finding in getattr(report, section_name).findings:
                for claim_id in finding.claim_ids:
                    if claim_id in seen_claim_ids or claim_id not in statuses_by_id:
                        continue
                    statuses = statuses_by_id[claim_id]
                    if statuses == {EvidenceStatus.UNKNOWN}:
                        target_section = "unknowns"
                    elif EvidenceStatus.UNKNOWN not in statuses:
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
                section_name: ResearchSection(findings=routed_findings[section_name])
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
            if item.status == EvidenceStatus.UNKNOWN:
                section = "unknowns"
                statement = f"Unknown: {item.claim}"
            else:
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
            field: ResearchSection(findings=sections[field])
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
    ) -> ResearchReport:
        represented = {
            claim_id for finding in report.unknowns.findings for claim_id in finding.claim_ids
        }
        additions: list[ResearchFinding] = []
        seen: set[str] = set()
        for item in bundle.evidence:
            if (
                item.status == EvidenceStatus.UNKNOWN
                and item.claim_id not in represented
                and item.claim_id not in seen
            ):
                seen.add(item.claim_id)
                additions.append(
                    ResearchFinding(
                        statement=f"Unknown: {item.claim}",
                        claim_ids=[item.claim_id],
                    )
                )
        if not additions:
            return report
        return report.model_copy(
            update={"unknowns": ResearchSection(findings=[*report.unknowns.findings, *additions])}
        )

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
