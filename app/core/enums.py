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
    BUDGET_REACHED = "budget_reached"
    NO_NEW_EVIDENCE = "no_new_evidence"
    CONTINUOUS_SEARCH_FAILURE = "continuous_search_failure"


class TransitionOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
