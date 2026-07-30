import inspect
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TypeVar

from app.core.enums import AgentState, TransitionOutcome
from app.core.exceptions import InvalidStateTransitionError
from app.core.protocols import Clock, EventLogger
from app.schemas.trace import RunTrace, StateTransition

T = TypeVar("T")


LEGAL_TRANSITIONS: dict[AgentState | None, frozenset[AgentState]] = {
    None: frozenset({AgentState.RESOLVE_COMPANY}),
    AgentState.RESOLVE_COMPANY: frozenset(
        {AgentState.PLAN_RESEARCH, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.PLAN_RESEARCH: frozenset(
        {AgentState.GENERATE_SEARCH_QUERIES, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.GENERATE_SEARCH_QUERIES: frozenset(
        {AgentState.PARALLEL_SEARCH, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.PARALLEL_SEARCH: frozenset(
        {
            AgentState.PROCESS_EVIDENCE,
            AgentState.REFINE_COMPANY_IDENTITY,
            AgentState.FINALIZE,
        }
    ),
    AgentState.PROCESS_EVIDENCE: frozenset(
        {AgentState.SUMMARIZE, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.SUMMARIZE: frozenset(
        {AgentState.ASSESS_INFORMATION, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.ASSESS_INFORMATION: frozenset(
        {AgentState.GENERATE_FOLLOWUP_QUERIES, AgentState.REFINE_COMPANY_IDENTITY}
    ),
    AgentState.GENERATE_FOLLOWUP_QUERIES: frozenset(
        {
            AgentState.PARALLEL_SEARCH,
            AgentState.REFINE_COMPANY_IDENTITY,
            AgentState.FINALIZE,
        }
    ),
    AgentState.REFINE_COMPANY_IDENTITY: frozenset(
        {AgentState.GENERATE_FOLLOWUP_QUERIES, AgentState.FINALIZE}
    ),
    AgentState.FINALIZE: frozenset({AgentState.RETURN_RESULT}),
    AgentState.RETURN_RESULT: frozenset(),
}


class ObservableStateMachine:
    def __init__(self, *, trace: RunTrace, clock: Clock, logger: EventLogger) -> None:
        self._trace = trace
        self._clock = clock
        self._logger = logger
        self._state: AgentState | None = None

    @property
    def state(self) -> AgentState | None:
        return self._state

    async def execute(
        self,
        to_state: AgentState,
        *,
        round_number: int,
        action: Callable[[], Awaitable[T] | T],
        details: dict[str, object] | None = None,
    ) -> T:
        if to_state not in LEGAL_TRANSITIONS[self._state]:
            raise InvalidStateTransitionError(
                f"illegal state transition: {self._state!s} -> {to_state.value}"
            )

        from_state = self._state
        started_at = self._clock.now()
        started_counter = perf_counter()
        error: str | None = None
        outcome = TransitionOutcome.SUCCEEDED
        try:
            pending = action()
            result = await pending if inspect.isawaitable(pending) else pending
            self._state = to_state
            return result
        except BaseException as exc:
            outcome = TransitionOutcome.FAILED
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            completed_at = self._clock.now()
            event = StateTransition(
                sequence=len(self._trace.transitions) + 1,
                from_state=from_state,
                to_state=to_state,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=(perf_counter() - started_counter) * 1000,
                round=round_number,
                details=dict(details or {}),
                outcome=outcome,
                error=error,
            )
            self._trace.transitions.append(event)
            fields = {
                "run_id": self._trace.run_id,
                "round": round_number,
                "from_state": from_state.value if from_state else None,
                "to_state": to_state.value,
                "duration_ms": event.duration_ms,
                "outcome": outcome.value,
                "error": error,
            }
            try:
                if error:
                    self._logger.error("state_transition", **fields)
                else:
                    self._logger.info("state_transition", **fields)
            except Exception:
                # Observability must never change workflow semantics or mask the
                # original state-action exception.
                pass
