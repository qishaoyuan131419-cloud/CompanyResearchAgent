class ResearchAgentError(Exception):
    """Base exception for expected agent failures."""


class ConfigurationError(ResearchAgentError):
    """Raised when a configured dependency cannot be constructed."""


class StructuredOutputError(ResearchAgentError):
    """Raised when an LLM response cannot be validated safely."""


class SearchError(ResearchAgentError):
    """Raised when a search adapter cannot complete a request."""


class BudgetExceededError(ResearchAgentError):
    """Raised before further paid work when the configured budget is exhausted."""


class InvalidStateTransitionError(ResearchAgentError):
    """Raised when orchestration attempts an undeclared state transition."""


class RunTimeoutError(ResearchAgentError):
    """Raised when an end-to-end research run exceeds its configured deadline."""
