from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import ConfigurationError, ResearchAgentError, RunTimeoutError


async def http_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request
    if not isinstance(exc, HTTPException):
        raise exc
    code = {
        status.HTTP_401_UNAUTHORIZED: "authentication_required",
        status.HTTP_503_SERVICE_UNAVAILABLE: "service_not_ready",
    }.get(exc.status_code, "http_error")
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": {
                "code": code,
                "message": str(exc.detail),
            }
        },
    )


async def configuration_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request
    if not isinstance(exc, ConfigurationError):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": {
                "code": "service_not_configured",
                "message": str(exc),
            }
        },
    )


async def research_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request
    if not isinstance(exc, ResearchAgentError):
        raise exc
    if isinstance(exc, RunTimeoutError):
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": {
                    "code": "research_timeout",
                    "message": str(exc),
                }
            },
        )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": {
                "code": "research_provider_error",
                "message": str(exc),
            }
        },
    )
