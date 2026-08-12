import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.dependencies import get_container
from app.schemas.api import (
    HealthResponse,
    ResearchJobCreated,
    ResearchJobSnapshot,
    ResearchRequest,
    ResearchResponse,
)
from app.services.container import ApplicationContainer

health_router = APIRouter(tags=["health"])
research_router = APIRouter(tags=["research"])


@health_router.get("/health/live", response_model=HealthResponse)
async def liveness(container: ApplicationContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=container.settings.app_name,
        version=__version__,
    )


@health_router.get("/health/ready", response_model=HealthResponse)
async def readiness(container: ApplicationContainer = Depends(get_container)) -> HealthResponse:
    if not container.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required providers are not configured.",
        )
    return HealthResponse(
        status="ready",
        service=container.settings.app_name,
        version=__version__,
    )


@research_router.post(
    "/research",
    response_model=ResearchResponse,
    response_model_exclude_none=True,
)
async def research_company(
    request: ResearchRequest,
    container: ApplicationContainer = Depends(get_container),
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> ResearchResponse:
    _require_api_key(container, x_api_key)
    return await container.research(request)


@research_router.post(
    "/research-jobs",
    response_model=ResearchJobCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_research_job(
    request: ResearchRequest,
    container: ApplicationContainer = Depends(get_container),
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> ResearchJobCreated:
    _require_api_key(container, x_api_key)
    run_id = await container.jobs.create(request)
    prefix = container.settings.api_prefix.rstrip("/")
    return ResearchJobCreated(
        run_id=run_id,
        status_url=f"{prefix}/research-jobs/{run_id}",
        events_url=f"{prefix}/research-jobs/{run_id}/events",
    )


@research_router.get("/research-jobs/{run_id}", response_model=ResearchJobSnapshot)
async def get_research_job(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> ResearchJobSnapshot:
    _require_api_key(container, x_api_key)
    snapshot = await container.jobs.snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found.")
    return snapshot


@research_router.get("/research-jobs/{run_id}/events")
async def stream_research_job(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    _require_api_key(container, x_api_key)
    snapshot = await container.jobs.snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found.")
    try:
        cursor = int(last_event_id or "0")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be an integer.",
        ) from exc
    return StreamingResponse(
        container.jobs.stream(run_id, after_event_id=cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _require_api_key(container: ApplicationContainer, x_api_key: str | None) -> None:
    configured_key = container.settings.service_api_key
    if configured_key is not None:
        expected = configured_key.get_secret_value()
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid service API key is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
