from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import SourceType
from app.utils.urls import is_public_suffix_only

_SECRET_QUERY_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def _is_loopback_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _contains_secret_query_parameter(query: str) -> bool:
    for key, _ in parse_qsl(query, keep_blank_values=True):
        normalized = "".join(character for character in key.casefold() if character.isalnum())
        if normalized in {"auth", "key"} or any(
            marker in normalized for marker in _SECRET_QUERY_KEY_MARKERS
        ):
            return True
    return False


def _validate_endpoint(
    value: str | None,
    *,
    setting_name: str,
    environment: Literal["development", "test", "production"],
    allow_query: bool,
) -> None:
    if value is None:
        return
    if (
        not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{setting_name} must be a valid absolute HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{setting_name} must be a valid absolute HTTP(S) URL") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{setting_name} must be a valid absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{setting_name} must not contain user information")
    if parsed.fragment or "#" in value:
        raise ValueError(f"{setting_name} must not contain a URL fragment")
    if not allow_query and "?" in value:
        raise ValueError(f"{setting_name} must not contain query parameters")
    if parsed.query and _contains_secret_query_parameter(parsed.query):
        raise ValueError(f"{setting_name} must not contain credentials in query parameters")
    if environment == "production" and scheme != "https":
        raise ValueError(f"{setting_name} must use HTTPS in production")
    if scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            f"{setting_name} may use HTTP only for loopback endpoints in development or test"
        )


def _secret_is_configured(value: SecretStr | None) -> bool:
    return value is not None and bool(value.get_secret_value().strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CRA_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Company Research Agent"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    service_api_key: SecretStr | None = None
    max_concurrent_runs: int = Field(default=4, ge=1, le=100)
    run_timeout_seconds: float = Field(default=900.0, gt=0.0)

    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=4_096, ge=1)
    llm_openai_max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
        "max_completion_tokens"
    )
    llm_input_cost_per_million: float = Field(default=0.0, ge=0.0)
    llm_output_cost_per_million: float = Field(default=0.0, ge=0.0)

    exa_mcp_url: str | None = None
    exa_api_key: SecretStr | None = None
    exa_search_tool: str = "web_search_advanced_exa"
    exa_results_per_query: int = Field(default=8, ge=1, le=100)
    exa_max_text_characters: int = Field(default=20_000, ge=1, le=500_000)
    source_type_domain_rules: dict[str, SourceType] = Field(default_factory=dict)

    max_search_rounds: int = Field(default=3, ge=1, le=10)
    search_timeout_seconds: float = Field(default=315.0, gt=0.0)
    search_max_retries: int = Field(default=1, ge=0, le=10)
    search_backoff_base_seconds: float = Field(default=0.5, ge=0.0)
    search_max_concurrency: int = Field(default=8, ge=1, le=100)
    max_queries_per_round: int = Field(default=30, ge=1, le=100)
    max_total_queries: int = Field(default=50, ge=1, le=500)
    max_consecutive_failed_rounds: int = Field(default=2, ge=1, le=10)
    evidence_limit: int = Field(default=500, ge=1, le=10_000)
    extraction_batch_size: int = Field(default=8, ge=1, le=100)
    extraction_max_prompt_bytes: int = Field(default=80_000, ge=1, le=2_000_000)
    no_new_evidence_rounds: int = Field(default=1, ge=1, le=5)
    token_budget: int = Field(default=100_000, ge=1)
    cost_budget_usd: float = Field(default=25.0, ge=0.0)
    sufficient_coverage_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    minimum_dimension_coverage: float = Field(default=0.40, ge=0.0, le=1.0)

    cache_enabled: bool = True
    cache_path: Path = Path(".cache/research-agent.sqlite3")
    search_cache_ttl_seconds: int = Field(default=86_400, ge=0)
    page_cache_ttl_seconds: int = Field(default=86_400, ge=0)
    llm_cache_ttl_seconds: int = Field(default=604_800, ge=0)

    prompt_directory: Path = Path(__file__).resolve().parent / "prompts" / "templates"

    @model_validator(mode="after")
    def ensure_production_credentials(self) -> "Settings":
        normalized_domain_rules: dict[str, SourceType] = {}
        for raw_domain, source_type in self.source_type_domain_rules.items():
            domain = raw_domain.casefold().strip().strip(".").removeprefix("www.")
            if (
                not domain
                or "." not in domain
                or any(character.isspace() for character in domain)
                or any(character in domain for character in "/:@?#")
                or is_public_suffix_only(domain)
            ):
                raise ValueError(
                    "CRA_SOURCE_TYPE_DOMAIN_RULES keys must identify publisher domains, "
                    "not URLs or public suffixes"
                )
            normalized_domain_rules[domain] = source_type
        self.source_type_domain_rules = normalized_domain_rules
        _validate_endpoint(
            self.llm_base_url,
            setting_name="CRA_LLM_BASE_URL",
            environment=self.environment,
            allow_query=False,
        )
        _validate_endpoint(
            self.exa_mcp_url,
            setting_name="CRA_EXA_MCP_URL",
            environment=self.environment,
            allow_query=True,
        )
        if self.environment == "production":
            missing: list[str] = []
            if (
                not _secret_is_configured(self.llm_api_key)
                or not self.llm_model.strip()
                or not self.llm_base_url
            ):
                missing.append("CRA_LLM_API_KEY/CRA_LLM_MODEL/CRA_LLM_BASE_URL")
            if not self.exa_mcp_url:
                missing.append("CRA_EXA_MCP_URL")
            if not _secret_is_configured(self.service_api_key):
                missing.append("CRA_SERVICE_API_KEY")
            if self.cost_budget_usd > 0 and (
                self.llm_input_cost_per_million <= 0 or self.llm_output_cost_per_million <= 0
            ):
                missing.append("CRA_LLM_INPUT_COST_PER_MILLION/CRA_LLM_OUTPUT_COST_PER_MILLION")
            if missing:
                raise ValueError(f"missing production configuration: {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
