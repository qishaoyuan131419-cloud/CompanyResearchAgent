from pydantic import Field, model_validator

from app.schemas.base import StrictModel


class ResearchFinding(StrictModel):
    statement: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)


class ResearchSection(StrictModel):
    findings: list[ResearchFinding] = Field(default_factory=list)


class ResearchReport(StrictModel):
    overview: ResearchSection = Field(default_factory=ResearchSection)
    products: ResearchSection = Field(default_factory=ResearchSection)
    technology: ResearchSection = Field(default_factory=ResearchSection)
    pipeline: ResearchSection = Field(default_factory=ResearchSection)
    manufacturing: ResearchSection = Field(default_factory=ResearchSection)
    recent_news: ResearchSection = Field(default_factory=ResearchSection)
    financial_signals: ResearchSection = Field(default_factory=ResearchSection)
    supply_chain: ResearchSection = Field(default_factory=ResearchSection)
    potential_procurement_signals: ResearchSection = Field(default_factory=ResearchSection)
    target_departments: ResearchSection = Field(default_factory=ResearchSection)
    risks: ResearchSection = Field(default_factory=ResearchSection)
    unknowns: ResearchSection = Field(default_factory=ResearchSection)
    recommendations: ResearchSection = Field(default_factory=ResearchSection)

    @model_validator(mode="after")
    def reject_urls_in_research(self) -> "ResearchReport":
        for section_name in type(self).model_fields:
            section = getattr(self, section_name)
            for finding in section.findings:
                lowered = finding.statement.lower()
                if "http://" in lowered or "https://" in lowered:
                    raise ValueError("research statements must reference claim IDs, not URLs")
        return self
