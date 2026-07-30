"""Evidence normalization, source lineage, and claim validation."""

from app.evidence.policy import EvidencePolicy
from app.evidence.processor import (
    ClaimExtractionResponse,
    EvidenceProcessingResult,
    EvidenceProcessor,
)
from app.evidence.registry import RegisteredSources, SourceRegistry

__all__ = [
    "ClaimExtractionResponse",
    "EvidencePolicy",
    "EvidenceProcessingResult",
    "EvidenceProcessor",
    "RegisteredSources",
    "SourceRegistry",
]
