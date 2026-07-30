import asyncio

from app.cache import NullCache, SearchResultCache, SQLiteTTLCache
from app.config import Settings
from app.core.budget import BudgetLedger
from app.core.clock import SystemClock
from app.core.exceptions import ConfigurationError, RunTimeoutError
from app.core.protocols import AsyncCache
from app.evidence.processor import EvidenceProcessor
from app.evidence.registry import SourceRegistry
from app.llm.factory import build_llm_client
from app.logging.setup import StructuredLogger
from app.planner.service import ResearchPlanner
from app.prompts.repository import FilePromptRepository
from app.reflection.service import ReflectionService
from app.resolver.service import CompanyResolver
from app.schemas.api import ResearchRequest, ResearchResponse
from app.search.exa_mcp import ExaMCPSearchClient
from app.search.executor import SearchExecutor
from app.services.finalizer import ResearchFinalizer
from app.services.orchestrator import ResearchOrchestrator
from app.utils.hashing import stable_hash


class ApplicationContainer:
    """Composition root for shared infrastructure and isolated per-run state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clock = SystemClock()
        self.logger = StructuredLogger("company_research_agent")
        self.prompts = FilePromptRepository(settings.prompt_directory)
        self.cache: AsyncCache = (
            SQLiteTTLCache(settings.cache_path) if settings.cache_enabled else NullCache()
        )
        self._run_semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
        self._search_executor: SearchExecutor | None = None
        if settings.exa_mcp_url:
            search_client = ExaMCPSearchClient.from_settings(settings)
            search_cache = (
                SearchResultCache(
                    self.cache,
                    ttl_seconds=settings.search_cache_ttl_seconds,
                    page_ttl_seconds=settings.page_cache_ttl_seconds,
                    provider=stable_hash(
                        {
                            "adapter": "exa-mcp-json-v2",
                            "endpoint": settings.exa_mcp_url,
                            "tool": settings.exa_search_tool,
                            "max_text_characters": settings.exa_max_text_characters,
                            "source_type_domain_rules": sorted(
                                (domain.casefold(), source_type.value)
                                for domain, source_type in settings.source_type_domain_rules.items()
                            ),
                        },
                        prefix="provider_",
                        length=40,
                    ),
                )
                if settings.cache_enabled
                else None
            )
            self._search_executor = SearchExecutor.from_settings(
                search_client,
                settings,
                result_cache=search_cache,
                clock=self.clock,
                logger=self.logger,
            )

    @property
    def ready(self) -> bool:
        api_key = self.settings.llm_api_key
        return bool(
            self._search_executor is not None
            and api_key is not None
            and api_key.get_secret_value().strip()
            and self.settings.llm_model.strip()
            and self.settings.llm_base_url
        )

    async def research(self, request: ResearchRequest) -> ResearchResponse:
        try:
            async with asyncio.timeout(self.settings.run_timeout_seconds):
                async with self._run_semaphore:
                    return await self._research_once(request)
        except TimeoutError as exc:
            raise RunTimeoutError(
                "The research run exceeded the configured end-to-end deadline."
            ) from exc

    async def _research_once(self, request: ResearchRequest) -> ResearchResponse:
        if self._search_executor is None:
            raise ConfigurationError("CRA_EXA_MCP_URL is required to run research")
        if not self.settings.llm_base_url:
            raise ConfigurationError("CRA_LLM_BASE_URL is required to run research")

        budget = BudgetLedger(
            token_limit=self.settings.token_budget,
            cost_limit_usd=self.settings.cost_budget_usd,
            query_limit=self.settings.max_total_queries,
        )
        llm = build_llm_client(self.settings, budget=budget, cache=self.cache)
        resolver = CompanyResolver(llm=llm, prompts=self.prompts)
        planner = ResearchPlanner(
            llm=llm,
            prompts=self.prompts,
            max_queries_per_round=self.settings.max_queries_per_round,
        )
        registry = SourceRegistry(clock=self.clock)
        evidence = EvidenceProcessor(
            llm_client=llm,
            prompt_repository=self.prompts,
            registry=registry,
            subject_identifiers=(request.canonical_name,),
            evidence_limit=self.settings.evidence_limit,
            extraction_batch_size=self.settings.extraction_batch_size,
            extraction_max_prompt_bytes=self.settings.extraction_max_prompt_bytes,
        )
        reflection = ReflectionService(
            llm=llm,
            prompts=self.prompts,
            settings=self.settings,
            clock=self.clock,
        )
        finalizer = ResearchFinalizer(llm=llm, prompts=self.prompts, clock=self.clock)
        orchestrator = ResearchOrchestrator(
            settings=self.settings,
            resolver=resolver,
            planner=planner,
            search_executor=self._search_executor,
            evidence_processor=evidence,
            reflection=reflection,
            finalizer=finalizer,
            budget=budget,
            logger=self.logger,
            clock=self.clock,
        )
        try:
            return await orchestrator.run(request)
        finally:
            await llm.aclose()

    async def close(self) -> None:
        await self.cache.close()
