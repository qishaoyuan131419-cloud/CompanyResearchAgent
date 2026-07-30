from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_liveness_does_not_require_provider_credentials() -> None:
    app = create_app(Settings(environment="test", cache_enabled=False))
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_missing_providers() -> None:
    app = create_app(Settings(environment="test", cache_enabled=False))
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_not_ready"


def test_research_request_requires_canonical_name() -> None:
    app = create_app(Settings(environment="test", cache_enabled=False))
    with TestClient(app) as client:
        response = client.post("/v1/research", json={"country": "US"})
    assert response.status_code == 422


def test_research_reports_missing_configuration_without_network_calls() -> None:
    app = create_app(Settings(environment="test", cache_enabled=False))
    with TestClient(app) as client:
        response = client.post("/v1/research", json={"canonical_name": "Acme Pharma"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_not_configured"


def test_configured_service_key_is_required_before_paid_route_dispatch() -> None:
    app = create_app(
        Settings(
            environment="test",
            cache_enabled=False,
            service_api_key="test-service-secret",
        )
    )
    with TestClient(app) as client:
        missing = client.post("/v1/research", json={"canonical_name": "Acme Pharma"})
        wrong = client.post(
            "/v1/research",
            json={"canonical_name": "Acme Pharma"},
            headers={"X-API-Key": "wrong"},
        )
        authorized = client.post(
            "/v1/research",
            json={"canonical_name": "Acme Pharma"},
            headers={"X-API-Key": "test-service-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert authorized.status_code == 503
    assert missing.json()["error"]["code"] == "authentication_required"
    assert wrong.json()["error"]["code"] == "authentication_required"


def test_blank_llm_key_is_not_ready() -> None:
    app = create_app(
        Settings(
            environment="test",
            cache_enabled=False,
            llm_api_key="   ",
            llm_model="model",
            llm_base_url="http://localhost:9998",
            exa_mcp_url="http://localhost:9999/mcp",
        )
    )
    with TestClient(app) as client:
        readiness = client.get("/health/ready")
        research = client.post(
            "/v1/research",
            json={"canonical_name": "Acme Pharma"},
        )

    assert readiness.status_code == 503
    assert research.status_code == 503
    assert research.json()["error"]["code"] == "service_not_configured"
