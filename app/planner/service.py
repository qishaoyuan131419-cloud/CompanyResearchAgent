import json
from typing import Any

from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import LLMClient, PromptRepository
from app.llm.errors import LLMProviderError
from app.schemas.company import CompanySnapshot, ResolvedCompany
from app.schemas.planning import QueryPlan, ResearchPlan, ResearchTopic, SearchQuery
from app.utils.hashing import stable_hash
from app.utils.text import normalized_fingerprint_text

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
        candidates = self._ensure_multiple_queries_per_topic(company, plan, candidates)
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
        for candidate in candidates:
            fingerprint = normalized_fingerprint_text(candidate.query)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
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

    @classmethod
    def _ensure_multiple_queries_per_topic(
        cls,
        company: ResolvedCompany,
        plan: ResearchPlan,
        candidates: list[SearchQuery],
    ) -> list[SearchQuery]:
        counts: dict[str, int] = {}
        for candidate in candidates:
            key = normalized_fingerprint_text(candidate.topic)
            counts[key] = counts.get(key, 0) + 1
        supplemented = list(candidates)
        fallbacks_by_topic: dict[str, list[SearchQuery]] = {}
        for fallback in cls._fallback_queries(company, plan):
            fallbacks_by_topic.setdefault(normalized_fingerprint_text(fallback.topic), []).append(
                fallback
            )
        for topic in plan.topics:
            key = normalized_fingerprint_text(topic.topic)
            needed = max(0, 2 - counts.get(key, 0))
            supplemented.extend(fallbacks_by_topic.get(key, [])[:needed])
        return supplemented
