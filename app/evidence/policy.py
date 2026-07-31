"""Deterministic policy for assigning evidence status and confidence."""

from collections.abc import Collection
from dataclasses import dataclass, field

from app.core.enums import EvidenceStatus, SourceType
from app.schemas.search import SourceDocument
from app.utils.urls import source_domain

_DEFAULT_RELIABLE_SOURCE_TYPES = frozenset(
    {
        SourceType.OFFICIAL,
        SourceType.REGULATORY,
        SourceType.ACADEMIC,
        SourceType.DATABASE,
        SourceType.NEWS,
        SourceType.INDUSTRY,
    }
)
_AUTHORITATIVE_SINGLE_SOURCE_TYPES = frozenset(
    {
        SourceType.OFFICIAL,
        SourceType.REGULATORY,
    }
)


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Centralizes conservative, reproducible evidence classification rules.

    Source type is the only reliability signal exposed by the shared source schema.
    Callers may narrow ``reliable_source_types`` when they have a stricter trust
    policy. Social and unknown/other sources are deliberately excluded by default.
    """

    reliable_source_types: frozenset[SourceType] = field(
        default_factory=lambda: _DEFAULT_RELIABLE_SOURCE_TYPES
    )
    single_source_confidence_cap: float = 0.74
    inference_confidence_cap: float = 0.65

    def is_reliable(self, source: SourceDocument) -> bool:
        return source.source_type in self.reliable_source_types

    def status_for(
        self,
        *,
        sources: Collection[SourceDocument],
        is_inference: bool,
    ) -> EvidenceStatus:
        if not sources:
            return EvidenceStatus.UNKNOWN
        if is_inference:
            return EvidenceStatus.INFERENCE
        if any(source.source_type in _AUTHORITATIVE_SINGLE_SOURCE_TYPES for source in sources):
            return EvidenceStatus.VERIFIED_FACT
        reliable_domains = {
            source_domain(str(source.url)) for source in sources if self.is_reliable(source)
        }
        if len(reliable_domains) >= 2:
            return EvidenceStatus.VERIFIED_FACT
        if len(reliable_domains) == 1:
            return EvidenceStatus.SINGLE_SOURCE
        return EvidenceStatus.UNKNOWN

    def confidence_for(self, *, status: EvidenceStatus, extracted: float) -> float:
        confidence = min(1.0, max(0.0, extracted))
        if status == EvidenceStatus.UNKNOWN:
            return 0.0
        if status == EvidenceStatus.INFERENCE:
            return min(confidence, self.inference_confidence_cap)
        if status == EvidenceStatus.SINGLE_SOURCE:
            return min(confidence, self.single_source_confidence_cap)
        return confidence
