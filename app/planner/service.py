import json
from typing import Any

from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import LLMClient, PromptRepository
from app.llm.errors import LLMProviderError
from app.schemas.company import CompanySnapshot, ResolvedCompany
from app.schemas.planning import (
    QueryPlan,
    ResearchPlan,
    ResearchPlanAndQueries,
    ResearchTopic,
    SearchQuery,
)
from app.utils.hashing import stable_hash
from app.utils.text import normalized_fingerprint_text, queries_are_near_duplicates

_DEFAULT_TOPICS = (
    ("Company Identity", 5, "Disambiguate the legal and operating entity."),
    ("Company Overview", 5, "Establish current scope and business focus."),
    ("Products", 5, "Identify commercially available products."),
    ("Technology", 4, "Identify platforms and technical capabilities."),
    ("Pipeline and Clinical Stage", 5, "Identify development assets and stages."),
    ("Manufacturing", 4, "Identify internal and external manufacturing capabilities."),
    ("Recent News", 4, "Find material recent developments."),
    ("Financial Signals", 3, "Find funding and financial signals."),
    ("Partnerships", 4, "Find collaboration and licensing relationships."),
    ("Supply Chain", 4, "Identify suppliers, dependencies, and capacity signals."),
    ("Procurement Signals", 4, "Identify evidence of purchasing needs or expansion."),
    ("Target Departments", 3, "Identify evidence-backed organizational functions."),
    ("Risks and Competition", 3, "Identify material risks and competitors."),
)


class ResearchPlanner:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRepository,
        max_queries_per_round: int,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._max_queries = max_queries_per_round

    async def build_plan_and_queries(
        self,
        company: ResolvedCompany,
        prior_query_fingerprints: set[str],
        snapshot: CompanySnapshot | None = None,
    ) -> tuple[ResearchPlan, QueryPlan]:
        """Produce the initial plan and distinct high-yield queries in one model call."""

        prompt = self._prompts.render(
            "planner",
            {
                "task": "plan_and_queries",
                "resolved_company": "Supplied in the structured user JSON payload.",
                "research_plan": "Produce topics and queries together.",
                "max_queries": str(self._max_queries),
            },
        )
        payload: dict[str, Any] = {
            "task": "plan_and_queries",
            "max_queries": self._max_queries,
            "resolved_company": company.model_dump(mode="json"),
            "company_snapshot_hints": snapshot.model_dump(mode="json") if snapshot else None,
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=ResearchPlanAndQueries,
                cache_namespace="planner-plan-and-queries",
            )
            topics = self._deduplicate_topics(result.value.topics)
            candidates = result.value.queries
        except BudgetExceededError:
            topics = self._default_plan().topics
            candidates = []
        except (LLMProviderError, StructuredOutputError):
            topics = self._default_plan().topics
            candidates = self._fallback_queries(company, ResearchPlan(topics=topics))
        if not any(topic.topic.casefold() == "company identity" for topic in topics):
            topics.insert(0, self._default_plan().topics[0])
        plan = ResearchPlan(topics=topics)
        queries = self.prepare_queries(candidates, prior_query_fingerprints)[: self._max_queries]
        return plan, QueryPlan(queries=queries)

    async def build_plan(
        self,
        company: ResolvedCompany,
        snapshot: CompanySnapshot | None = None,
    ) -> ResearchPlan:
        prompt = self._prompts.render(
            "planner",
            {
                "task": "topics",
                "resolved_company": "Supplied in the structured user JSON payload.",
                "research_plan": "Supplied in the structured user JSON payload.",
                "max_queries": str(self._max_queries),
            },
        )
        payload: dict[str, Any] = {
            "task": "topics",
            "resolved_company": company.model_dump(mode="json"),
            "company_snapshot_hints": snapshot.model_dump(mode="json") if snapshot else None,
            "research_plan": [],
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=ResearchPlan,
                cache_namespace="planner-topics",
            )
            plan = result.value
        except BudgetExceededError:
            plan = self._default_plan()
        except (LLMProviderError, StructuredOutputError):
            plan = self._default_plan()
        topics = self._deduplicate_topics(plan.topics)
        if not any(topic.topic.casefold() == "company identity" for topic in topics):
            topics.insert(0, self._default_plan().topics[0])
        return ResearchPlan(topics=topics)

    async def generate_queries(
        self,
        company: ResolvedCompany,
        plan: ResearchPlan,
        prior_query_fingerprints: set[str],
        snapshot: CompanySnapshot | None = None,
    ) -> QueryPlan:
        prompt = self._prompts.render(
            "planner",
            {
                "task": "queries",
                "resolved_company": "Supplied in the structured user JSON payload.",
                "research_plan": "Supplied in the structured user JSON payload.",
                "max_queries": str(self._max_queries),
            },
        )
        payload: dict[str, Any] = {
            "task": "queries",
            "resolved_company": company.model_dump(mode="json"),
            "company_snapshot_hints": snapshot.model_dump(mode="json") if snapshot else None,
            "research_plan": plan.model_dump(mode="json"),
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(payload, sort_keys=True, default=str),
                response_model=QueryPlan,
                cache_namespace="planner-queries",
            )
            candidates = result.value.queries
        except BudgetExceededError:
            candidates = self._fallback_queries(company, plan)
        except (LLMProviderError, StructuredOutputError):
            candidates = self._fallback_queries(company, plan)
        return QueryPlan(
            queries=self.prepare_queries(candidates, prior_query_fingerprints)[: self._max_queries]
        )

    @staticmethod
    def prepare_queries(
        candidates: list[SearchQuery],
        prior_query_fingerprints: set[str],
    ) -> list[SearchQuery]:
        prepared: list[SearchQuery] = []
        seen = set(prior_query_fingerprints)
        seen_intents: set[str] = set()
        for candidate in candidates:
            fingerprint = normalized_fingerprint_text(candidate.query)
            intent = normalized_fingerprint_text(candidate.intent or "")
            if (
                not fingerprint
                or any(queries_are_near_duplicates(fingerprint, previous) for previous in seen)
                or (intent and intent in seen_intents)
            ):
                continue
            seen.add(fingerprint)
            if intent:
                seen_intents.add(intent)
            prepared.append(
                candidate.model_copy(
                    update={
                        "query_id": stable_hash(fingerprint, prefix="qry_"),
                        "query": " ".join(candidate.query.split()),
                    }
                )
            )
        return prepared

    @staticmethod
    def _deduplicate_topics(topics: list[ResearchTopic]) -> list[ResearchTopic]:
        unique: dict[str, ResearchTopic] = {}
        for topic in topics:
            key = normalized_fingerprint_text(topic.topic)
            current = unique.get(key)
            if current is None or topic.priority > current.priority:
                unique[key] = topic
        return sorted(unique.values(), key=lambda topic: (-topic.priority, topic.topic.casefold()))

    @staticmethod
    def _default_plan() -> ResearchPlan:
        return ResearchPlan(
            topics=[
                ResearchTopic(
                    topic=name,
                    priority=priority,
                    reason=reason,
                    expected_output=f"Evidence-backed findings for {name}; Unknown if unavailable.",
                    estimated_value=priority / 5,
                )
                for name, priority, reason in _DEFAULT_TOPICS
            ]
        )

    @staticmethod
    def _fallback_queries(
        company: ResolvedCompany,
        plan: ResearchPlan,
    ) -> list[SearchQuery]:
        queries: list[SearchQuery] = []
        for topic in plan.topics:
            for qualifier in ("official", "pharmaceutical"):
                query_text = f'"{company.canonical_name}" {topic.topic} {qualifier}'
                queries.append(
                    SearchQuery(
                        query_id="pending",
                        query=query_text,
                        topic=topic.topic,
                        priority=topic.priority,
                        expected_evidence=topic.expected_output,
                        language="en",
                        source_preference=[
                            "official",
                            "regulatory",
                            "reputable industry sources",
                        ],
                    )
                )
        return queries
