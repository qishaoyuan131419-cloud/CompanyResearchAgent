import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app import __version__
from app.api.dependencies import get_container
from app.schemas.api import HealthResponse, ResearchRequest, ResearchResponse
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
    configured_key = container.settings.service_api_key
    if configured_key is not None:
        expected = configured_key.get_secret_value()
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid service API key is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return await container.research(request)
