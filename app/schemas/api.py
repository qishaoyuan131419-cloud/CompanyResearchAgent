from app.schemas.base import StrictModel
from app.schemas.company import CompanySnapshot, ResolvedCompany
from app.schemas.evidence import EvidenceBundle
from app.schemas.research import ResearchReport
from app.schemas.trace import RunTrace


class ResearchRequest(CompanySnapshot):
    pass


class ResearchResponse(StrictModel):
    company: ResolvedCompany
    research: ResearchReport
    evidence: EvidenceBundle
    run_trace: RunTrace


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
