from app.core.exceptions import SearchError


class SearchClientError(SearchError):
    """Expected search-client failure with an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class RetryableSearchError(SearchClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class SearchResponseError(SearchClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
