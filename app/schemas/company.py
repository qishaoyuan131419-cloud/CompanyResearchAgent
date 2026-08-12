from typing import Annotated

from pydantic import AnyHttpUrl, Field, StringConstraints

from app.core.enums import IdentityStatus
from app.schemas.base import StrictModel

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
ShortHint = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]


class CompanySnapshot(StrictModel):
    canonical_name: NonEmptyString
    website: AnyHttpUrl | None = None
    linkedin: AnyHttpUrl | None = None
    country: ShortHint | None = None
    industry: ShortHint | None = None
    summary: str | None = Field(default=None, max_length=10_000)


class CompanyRelationship(StrictModel):
    relationship_type: ShortHint
    related_company: NonEmptyString
    status: ShortHint | None = None
    claim_ids: list[str] = Field(default_factory=list)


class CompanyCandidate(StrictModel):
    name: NonEmptyString
    website: AnyHttpUrl | None = None
    country: ShortHint | None = None
    distinguishing_details: list[NonEmptyString] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class ResolvedCompany(StrictModel):
    canonical_name: NonEmptyString
    identity_status: IdentityStatus = IdentityStatus.UNCONFIRMED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    aliases: list[NonEmptyString] = Field(default_factory=list)
    website: AnyHttpUrl | None = None
    country: ShortHint | None = None
    industry: ShortHint | None = None
    is_closed: bool | None = None
    candidates: list[CompanyCandidate] = Field(default_factory=list)
    relationships: list[CompanyRelationship] = Field(default_factory=list)
    verification_notes: list[NonEmptyString] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
