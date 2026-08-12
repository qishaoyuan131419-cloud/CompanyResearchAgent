import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.core.protocols import Clock
from app.schemas.api import ResearchJobSnapshot, ResearchRequest, ResearchResponse

ProgressCallback = Callable[[str, Mapping[str, Any]], Awaitable[None]]
ResearchRunner = Callable[
    [ResearchRequest, str, ProgressCallback],
    Awaitable[ResearchResponse],
]


@dataclass(frozen=True, slots=True)
class _JobEvent:
    event_id: int
    event_type: str
    created_at: datetime
    data: dict[str, Any]


@dataclass(slots=True)
class _ResearchJob:
    run_id: str
    request: ResearchRequest
    created_at: datetime
    updated_at: datetime
    status: str = "queued"
    result: ResearchResponse | None = None
    error: str | None = None
    events: list[_JobEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None


class ResearchJobManager:
    """Small in-process job registry for async/SSE clients.

    Production deployments can replace this component with the host project's
    durable queue without changing the HTTP contract.
    """

    def __init__(
        self,
        *,
        runner: ResearchRunner,
        clock: Clock,
        max_jobs: int = 100,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self._runner = runner
        self._clock = clock
        self._max_jobs = max_jobs
        self._heartbeat_seconds = heartbeat_seconds
        self._jobs: dict[str, _ResearchJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, request: ResearchRequest) -> str:
        run_id = f"run_{uuid4().hex}"
        now = self._clock.now()
        job = _ResearchJob(run_id=run_id, request=request, created_at=now, updated_at=now)
        async with self._lock:
            self._discard_oldest_completed_if_needed()
            self._jobs[run_id] = job
        await self._publish(job, "queued", {"status": "queued"})
        job.task = asyncio.create_task(self._execute(job), name=f"research-job-{run_id}")
        return run_id

    async def snapshot(self, run_id: str) -> ResearchJobSnapshot | None:
        async with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return None
        return ResearchJobSnapshot(
            run_id=job.run_id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            result=job.result,
            error=job.error,
        )

    async def stream(self, run_id: str, *, after_event_id: int = 0) -> AsyncIterator[str]:
        async with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return
        cursor = max(0, after_event_id)
        while True:
            pending = [event for event in job.events if event.event_id > cursor]
            if pending:
                for event in pending:
                    cursor = event.event_id
                    payload = {
                        "run_id": run_id,
                        "event_id": event.event_id,
                        "created_at": event.created_at.isoformat(),
                        **event.data,
                    }
                    yield (
                        f"id: {event.event_id}\n"
                        f"event: {event.event_type}\n"
                        f"data: {json.dumps(payload, sort_keys=True, default=str)}\n\n"
                    )
                continue
            if job.status in {"completed", "failed", "cancelled"}:
                return
            try:
                async with job.condition:
                    await asyncio.wait_for(
                        job.condition.wait(),
                        timeout=self._heartbeat_seconds,
                    )
            except TimeoutError:
                yield ": heartbeat\n\n"

    async def close(self) -> None:
        async with self._lock:
            tasks = [job.task for job in self._jobs.values() if job.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, job: _ResearchJob) -> None:
        job.status = "running"
        job.updated_at = self._clock.now()
        await self._publish(job, "running", {"status": "running"})

        async def progress(event_type: str, data: Mapping[str, Any]) -> None:
            await self._publish(job, event_type, dict(data))

        try:
            job.result = await self._runner(job.request, job.run_id, progress)
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "Research job was cancelled."
            job.updated_at = self._clock.now()
            await self._publish(job, "cancelled", {"status": "cancelled"})
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: research job failed"
            job.updated_at = self._clock.now()
            await self._publish(
                job,
                "failed",
                {"status": "failed", "error": job.error},
            )
        else:
            job.status = "completed"
            job.updated_at = self._clock.now()
            await self._publish(
                job,
                "completed",
                {
                    "status": "completed",
                    "execution_status": job.result.execution_status.value,
                    "source_count": job.result.evidence.source_count,
                    "supported_claim_count": job.result.evidence.supported_claim_count,
                    "verified_fact_count": job.result.evidence.verified_fact_count,
                },
            )

    async def _publish(
        self,
        job: _ResearchJob,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        event = _JobEvent(
            event_id=len(job.events) + 1,
            event_type=event_type,
            created_at=self._clock.now(),
            data=data,
        )
        job.events.append(event)
        job.updated_at = event.created_at
        async with job.condition:
            job.condition.notify_all()

    def _discard_oldest_completed_if_needed(self) -> None:
        if len(self._jobs) < self._max_jobs:
            return
        completed = [
            job for job in self._jobs.values() if job.status in {"completed", "failed", "cancelled"}
        ]
        if not completed:
            return
        oldest = min(completed, key=lambda job: job.updated_at)
        self._jobs.pop(oldest.run_id, None)
