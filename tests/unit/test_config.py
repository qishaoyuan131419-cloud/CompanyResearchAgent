import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.enums import SourceType


def production_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "service_api_key": "service-secret",
        "llm_api_key": "llm-secret",
        "llm_base_url": "https://llm.example/v1",
        "llm_model": "model",
        "exa_mcp_url": "https://mcp.exa.ai/mcp?tools=web_search_advanced_exa",
        "llm_input_cost_per_million": 1.0,
        "llm_output_cost_per_million": 1.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_accepts_https_endpoints_and_requires_service_api_key() -> None:
    settings = production_settings()

    assert settings.service_api_key is not None
    assert settings.service_api_key.get_secret_value() == "service-secret"

    with pytest.raises(ValidationError, match="CRA_SERVICE_API_KEY"):
        production_settings(service_api_key="")


@pytest.mark.parametrize("field", ["llm_base_url", "exa_mcp_url"])
def test_production_rejects_http_endpoints(field: str) -> None:
    with pytest.raises(ValidationError, match="must use HTTPS in production"):
        production_settings(**{field: "http://localhost:8000/v1"})


@pytest.mark.parametrize("environment", ["development", "test"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8000/v1",
        "http://localhost.:8000/v1",
        "http://127.0.0.2:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_nonproduction_allows_only_loopback_http(environment: str, endpoint: str) -> None:
    settings = Settings(
        _env_file=None,
        environment=environment,
        llm_base_url=endpoint,
        exa_mcp_url=endpoint,
    )

    assert settings.llm_base_url == endpoint
    assert settings.exa_mcp_url == endpoint


@pytest.mark.parametrize("field", ["llm_base_url", "exa_mcp_url"])
@pytest.mark.parametrize("environment", ["development", "test"])
def test_nonproduction_rejects_non_loopback_http(field: str, environment: str) -> None:
    with pytest.raises(ValidationError, match="HTTP only for loopback"):
        Settings(
            _env_file=None,
            environment=environment,
            **{field: "http://internal.example/v1"},
        )


@pytest.mark.parametrize("field", ["llm_base_url", "exa_mcp_url"])
@pytest.mark.parametrize(
    "endpoint,error",
    [
        ("https://user:password@example.com/v1", "must not contain user information"),
        ("https://example.com/v1#credentials", "must not contain a URL fragment"),
    ],
)
def test_endpoints_reject_userinfo_and_fragments(field: str, endpoint: str, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        Settings(_env_file=None, environment="test", **{field: endpoint})


@pytest.mark.parametrize(
    "query",
    [
        "api_key=secret",
        "api%5Fkey=secret",
        "access_token=secret",
        "x-api-key=secret",
        "password=secret",
        "authorization=Bearer",
    ],
)
def test_exa_endpoint_rejects_secret_query_parameters(query: str) -> None:
    with pytest.raises(ValidationError, match="credentials in query parameters"):
        Settings(
            _env_file=None,
            environment="test",
            exa_mcp_url=f"https://mcp.exa.ai/mcp?{query}",
        )


def test_llm_endpoint_rejects_all_query_parameters() -> None:
    for endpoint in (
        "https://llm.example/v1?api-version=2026-01-01",
        "https://llm.example/v1?",
    ):
        with pytest.raises(ValidationError, match="must not contain query parameters"):
            Settings(
                _env_file=None,
                environment="test",
                llm_base_url=endpoint,
            )


def test_run_limits_are_bounded_and_configurable() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        max_concurrent_runs=7,
        run_timeout_seconds=123.5,
    )

    assert settings.max_concurrent_runs == 7
    assert settings.run_timeout_seconds == 123.5

    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_concurrent_runs=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, run_timeout_seconds=0)


@pytest.mark.parametrize("domain", ["com", "co.uk", "https://example.com", "localhost"])
def test_source_authority_rules_reject_public_suffixes_and_non_domains(domain: str) -> None:
    with pytest.raises(ValidationError, match="publisher domains"):
        Settings(
            _env_file=None,
            environment="test",
            source_type_domain_rules={domain: SourceType.OFFICIAL},
        )


def test_source_authority_rules_are_normalized() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        source_type_domain_rules={"WWW.Example.COM.": SourceType.OFFICIAL},
    )

    assert settings.source_type_domain_rules == {"example.com": SourceType.OFFICIAL}
