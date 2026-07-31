from pydantic import model_validator

from app.core.enums import ExecutionStatus
from app.schemas.base import StrictModel
from app.schemas.company import CompanySnapshot, ResolvedCompany
from app.schemas.evidence import EvidenceBundle
from app.schemas.research import ResearchReport
from app.schemas.trace import RunTrace


class ResearchRequest(CompanySnapshot):
    pass


class ResearchResponse(StrictModel):
    execution_status: ExecutionStatus
    company: ResolvedCompany
    research: ResearchReport
    evidence: EvidenceBundle
    run_trace: RunTrace

    @model_validator(mode="after")
    def validate_cross_references(self) -> "ResearchResponse":
        claim_ids = {item.claim_id for item in self.evidence.evidence}
        source_ids = {source.source_id for source in self.evidence.sources}
        for section_name in type(self.research).model_fields:
            section = getattr(self.research, section_name)
            for finding in getattr(section, "findings", []):
                missing = set(finding.claim_ids) - claim_ids
                if missing:
                    raise ValueError(
                        f"research finding references unknown claims: {sorted(missing)}"
                    )
            for gap in getattr(section, "gaps", []):
                missing_sources = set(gap.source_ids) - source_ids
                if missing_sources:
                    raise ValueError(
                        f"research gap references unknown sources: {sorted(missing_sources)}"
                    )
        gap_count = len(self.research.unknowns.gaps)
        if self.evidence.research_gap_count != gap_count:
            raise ValueError("research_gap_count must equal len(research.unknowns.gaps)")
        metric_names = (
            "source_count",
            "supported_claim_count",
            "verified_fact_count",
            "single_source_count",
            "inference_count",
            "research_gap_count",
            "rejected_claim_count",
            "processing_error_count",
        )
        for name in metric_names:
            if getattr(self.run_trace, name) != getattr(self.evidence, name):
                raise ValueError(f"run_trace.{name} is inconsistent with evidence.{name}")
        return self


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
