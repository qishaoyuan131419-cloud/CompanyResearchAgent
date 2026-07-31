from dataclasses import dataclass

from app.core.enums import AssessmentDecision, StopReason
from app.schemas.reflection import InformationAssessment


@dataclass(frozen=True, slots=True)
class StopContext:
    assessment: InformationAssessment
    current_round: int
    max_rounds: int
    consecutive_no_new_evidence_rounds: int
    no_new_evidence_limit: int
    consecutive_failed_rounds: int
    failed_rounds_limit: int
    budget_stop_reason: StopReason | None = None
    budget_exhausted: bool = False


class StopPolicy:
    """Deterministic stop guards in documented precedence order."""

    def evaluate(self, context: StopContext) -> StopReason | None:
        if context.assessment.decision == AssessmentDecision.STOP:
            return StopReason.SUFFICIENT_INFORMATION
        if context.consecutive_failed_rounds >= context.failed_rounds_limit:
            return StopReason.SEARCH_UNAVAILABLE
        if context.budget_stop_reason is not None:
            return context.budget_stop_reason
        if context.budget_exhausted:
            return StopReason.QUERY_BUDGET_REACHED
        if context.current_round >= context.max_rounds:
            return StopReason.MAX_ROUNDS_REACHED
        if context.consecutive_no_new_evidence_rounds >= context.no_new_evidence_limit:
            return StopReason.NO_NEW_EVIDENCE
        return None
