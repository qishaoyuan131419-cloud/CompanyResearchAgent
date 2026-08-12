from __future__ import annotations

from app.core.exceptions import ResearchAgentError


class LLMProviderError(ResearchAgentError):
    """A safe, provider-neutral error raised for an unsuccessful model request."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the configured credential."""


class LLMProviderResponseError(LLMProviderError):
    """The provider returned a successful HTTP response with an invalid envelope."""
