from enum import StrEnum


class AgentState(StrEnum):
    RESOLVE_COMPANY = "resolve_company"
    PLAN_RESEARCH = "plan_research"
    GENERATE_SEARCH_QUERIES = "generate_search_queries"
    PARALLEL_SEARCH = "parallel_search"
    PROCESS_EVIDENCE = "process_evidence"
    SUMMARIZE = "summarize"
    ASSESS_INFORMATION = "assess_information"
    GENERATE_FOLLOWUP_QUERIES = "generate_followup_queries"
    REFINE_COMPANY_IDENTITY = "refine_company_identity"
    FINALIZE = "finalize"
    RETURN_RESULT = "return_result"


class StopReason(StrEnum):
    SUFFICIENT_INFORMATION = "sufficient_information"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    QUERY_BUDGET_REACHED = "query_budget_reached"
    TOKEN_BUDGET_REACHED = "token_budget_reached"
    COST_BUDGET_REACHED = "cost_budget_reached"
    TIME_BUDGET_REACHED = "time_budget_reached"
    SOURCE_BUDGET_REACHED = "source_budget_reached"
    NO_NEW_EVIDENCE = "no_new_evidence"
    SEARCH_UNAVAILABLE = "search_unavailable"
    EXTRACTION_FAILURE = "extraction_failure"
    COMPANY_OUTSIDE_SCOPE = "company_outside_scope"
    # Compatibility aliases serialize to precise values; the obsolete literal
    # "budget_reached" remains invalid.
    BUDGET_REACHED = "query_budget_reached"
    CONTINUOUS_SEARCH_FAILURE = "search_unavailable"


class TransitionOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"


class ExtractionMethod(StrEnum):
    STRUCTURED_LLM = "structured_llm"
    REPAIRED_LLM = "repaired_llm"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    MANUAL_FIXTURE = "manual_fixture"


class ProcessingStage(StrEnum):
    SEARCH = "search"
    NORMALIZATION = "normalization"
    CONTENT_RETRIEVAL = "content_retrieval"
    DEDUPLICATION = "deduplication"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    ASSESSMENT = "assessment"
    FINALIZATION = "finalization"


class GapReason(StrEnum):
    NOT_SEARCHED = "not_searched"
    SEARCH_FAILED = "search_failed"
    SEARCHED_NO_RESULTS = "searched_no_results"
    SOURCE_ACCESS_FAILED = "source_access_failed"
    CONTENT_RETRIEVAL_FAILED = "content_retrieval_failed"
    EXTRACTION_FAILED = "extraction_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SOURCE_CONFLICT = "source_conflict"
    BUDGET_INTERRUPTED = "budget_interrupted"
    NOT_PUBLICLY_DISCLOSED = "not_publicly_disclosed"


class DimensionStatus(StrEnum):
    NOT_SEARCHED = "not_searched"
    SEARCH_FAILED = "search_failed"
    SEARCHED_NO_RESULTS = "searched_no_results"
    SOURCES_FOUND = "sources_found"
    CONTENT_RETRIEVAL_FAILED = "content_retrieval_failed"
    EXTRACTION_FAILED = "extraction_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVIDENCE_AVAILABLE = "evidence_available"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    NOT_APPLICABLE = "not_applicable"


class EvidenceStatus(StrEnum):
    VERIFIED_FACT = "verified_fact"
    SINGLE_SOURCE = "single_source"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


class IdentityStatus(StrEnum):
    CONFIRMED = "confirmed"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    UNCONFIRMED = "unconfirmed"
    AMBIGUOUS = "ambiguous"


class SourceType(StrEnum):
    OFFICIAL = "official"
    REGULATORY = "regulatory"
    ACADEMIC = "academic"
    DATABASE = "database"
    NEWS = "news"
    INDUSTRY = "industry"
    SOCIAL = "social"
    OTHER = "other"


class AssessmentDecision(StrEnum):
    CONTINUE_SEARCH = "continue_search"
    STOP = "stop"


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
