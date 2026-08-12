import asyncio
from dataclasses import dataclass

from app.core.exceptions import BudgetExceededError
from app.core.protocols import LLMUsage


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    total_tokens: int
    estimated_cost_usd: float
    queries_used: int
    token_limit: int
    cost_limit_usd: float
    query_limit: int
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0
    reservation_blocked: bool = False

    @property
    def llm_exhausted(self) -> bool:
        return (
            self.reservation_blocked
            or self.total_tokens + self.reserved_tokens >= self.token_limit
            or (
                self.cost_limit_usd > 0
                and self.estimated_cost_usd + self.reserved_cost_usd >= self.cost_limit_usd
            )
        )

    @property
    def query_exhausted(self) -> bool:
        return self.queries_used >= self.query_limit

    @property
    def exhausted(self) -> bool:
        return self.llm_exhausted or self.query_exhausted


class BudgetLedger:
    """Concurrency-safe accounting for search and model budgets."""

    def __init__(self, *, token_limit: int, cost_limit_usd: float, query_limit: int) -> None:
        self._token_limit = token_limit
        self._cost_limit_usd = cost_limit_usd
        self._query_limit = query_limit
        self._tokens = 0
        self._cost = 0.0
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
        self._queries = 0
        self._reservation_blocked = False
        self._lock = asyncio.Lock()

    async def allow_queries(self, requested: int) -> int:
        if requested < 0:
            raise ValueError("requested query count cannot be negative")
        async with self._lock:
            remaining = max(0, self._query_limit - self._queries)
            allowed = min(requested, remaining)
            self._queries += allowed
            return allowed

    async def ensure_llm_capacity(self) -> None:
        async with self._lock:
            if self._reservation_blocked:
                raise BudgetExceededError("LLM budget was already exhausted")
            if self._tokens >= self._token_limit:
                raise BudgetExceededError("token budget reached")
            if self._cost_limit_usd > 0 and self._cost >= self._cost_limit_usd:
                raise BudgetExceededError("cost budget reached")

    async def reserve_llm(self, estimate: LLMUsage) -> LLMUsage:
        """Atomically reserve the conservative upper bound for one model call."""

        tokens = max(0, estimate.total_tokens)
        cost = max(0.0, estimate.estimated_cost_usd)
        async with self._lock:
            if self._reservation_blocked:
                raise BudgetExceededError("LLM budget was already exhausted")
            if self._tokens + self._reserved_tokens + tokens > self._token_limit:
                self._reservation_blocked = True
                raise BudgetExceededError("insufficient remaining token budget for model call")
            if (
                self._cost_limit_usd > 0
                and self._cost + self._reserved_cost + cost > self._cost_limit_usd
            ):
                self._reservation_blocked = True
                raise BudgetExceededError("insufficient remaining cost budget for model call")
            self._reserved_tokens += tokens
            self._reserved_cost += cost
        return LLMUsage(
            input_tokens=max(0, estimate.input_tokens),
            output_tokens=max(0, estimate.output_tokens),
            estimated_cost_usd=cost,
        )

    async def settle_llm(self, reservation: LLMUsage, actual: LLMUsage) -> None:
        """Replace a reservation with provider-reported usage after a completed call."""

        async with self._lock:
            self._reserved_tokens = max(0, self._reserved_tokens - max(0, reservation.total_tokens))
            self._reserved_cost = max(
                0.0,
                self._reserved_cost - max(0.0, reservation.estimated_cost_usd),
            )
            self._tokens += max(0, actual.total_tokens)
            self._cost += max(0.0, actual.estimated_cost_usd)

    async def commit_llm_reservation(self, reservation: LLMUsage) -> None:
        """Conservatively charge a call when the provider gives no usage after failure."""

        await self.settle_llm(reservation, reservation)

    async def record_llm_usage(self, usage: LLMUsage) -> None:
        async with self._lock:
            self._tokens += max(0, usage.total_tokens)
            self._cost += max(0.0, usage.estimated_cost_usd)

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(
                total_tokens=self._tokens,
                estimated_cost_usd=self._cost,
                queries_used=self._queries,
                token_limit=self._token_limit,
                cost_limit_usd=self._cost_limit_usd,
                query_limit=self._query_limit,
                reserved_tokens=self._reserved_tokens,
                reserved_cost_usd=self._reserved_cost,
                reservation_blocked=self._reservation_blocked,
            )
