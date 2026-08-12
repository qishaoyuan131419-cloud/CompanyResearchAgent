from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import __version__
from app.api.errors import (
    configuration_error_handler,
    http_error_handler,
    research_error_handler,
)
from app.api.routes import health_router, research_router
from app.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, ResearchAgentError
from app.logging.setup import configure_logging
from app.services.container import ApplicationContainer


def create_app(
    settings: Settings | None = None,
    *,
    container: ApplicationContainer | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    owns_container = container is None
    resolved_container = container or ApplicationContainer(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = resolved_container
        try:
            yield
        finally:
            if owns_container:
                await resolved_container.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Evidence-first pharmaceutical company research with a complete run trace.",
        lifespan=lifespan,
    )
    application.add_exception_handler(ConfigurationError, configuration_error_handler)
    application.add_exception_handler(ResearchAgentError, research_error_handler)
    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(health_router)
    application.include_router(research_router, prefix=resolved_settings.api_prefix)
    return application


app = create_app()
