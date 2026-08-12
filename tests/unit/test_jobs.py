import asyncio

from app.core.clock import SystemClock
from app.schemas.api import ResearchRequest
from app.services.jobs import ResearchJobManager


async def test_job_manager_streams_progress_and_terminal_failure() -> None:
    async def failing_runner(request, run_id, progress):  # type: ignore[no-untyped-def]
        assert request.canonical_name == "Acme Pharma"
        assert run_id.startswith("run_")
        await progress("company_resolved", {"status": "running", "company": {"name": "Acme"}})
        raise RuntimeError("provider detail must not be exposed")

    manager = ResearchJobManager(runner=failing_runner, clock=SystemClock())
    run_id = await manager.create(ResearchRequest(canonical_name="Acme Pharma"))
    for _ in range(20):
        snapshot = await manager.snapshot(run_id)
        if snapshot is not None and snapshot.status == "failed":
            break
        await asyncio.sleep(0)

    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.error == "RuntimeError: research job failed"
    events = [event async for event in manager.stream(run_id)]
    rendered = "".join(events)
    assert "event: queued" in rendered
    assert "event: running" in rendered
    assert "event: company_resolved" in rendered
    assert "event: failed" in rendered
    assert "provider detail must not be exposed" not in rendered
    await manager.close()
