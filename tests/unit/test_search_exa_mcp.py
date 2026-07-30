from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from app.core.enums import SourceType
from app.schemas.planning import SearchQuery
from app.search.errors import SearchResponseError
from app.search.exa_mcp import (
    ExaMCPSearchClient,
    _is_permanent_transport_error,
    _server_url_for_tool,
    parse_exa_mcp_response,
)


def _query() -> SearchQuery:
    return SearchQuery(
        query_id="q-1",
        query="Acme pharmaceutical pipeline",
        topic="Pipeline",
        priority=1,
        expected_evidence="Pipeline assets",
    )


class FakeCaller:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return self.response


async def test_exa_client_parses_structured_results_and_passes_limit() -> None:
    caller = FakeCaller(
        {
            "structuredContent": {
                "results": [
                    {
                        "title": "Acme pipeline",
                        "url": "https://example.com/pipeline",
                        "publishedDate": "2026-01-02T00:00:00Z",
                        "highlights": ["Asset reached phase 2."],
                        "score": 0.9,
                    },
                    {"title": "Missing URL", "text": "Must not become evidence"},
                    {"title": "Unsafe URL", "url": "javascript:alert(1)"},
                ]
            }
        }
    )
    client = ExaMCPSearchClient(caller=caller, tool_name="web_search_exa")

    results = await client.search(_query(), limit=5)

    assert len(results) == 1
    assert str(results[0].url) == "https://example.com/pipeline"
    assert results[0].text == "Asset reached phase 2."
    assert caller.calls == [
        (
            "web_search_exa",
            {"query": "Acme pharmaceutical pipeline", "numResults": 5},
        )
    ]


async def test_advanced_exa_client_passes_upstream_text_limit() -> None:
    caller = FakeCaller(
        {
            "content": [
                {
                    "type": "text",
                    "text": '{"results":[{"title":"Acme","url":"https://example.com","text":"Evidence"}]}',
                }
            ]
        }
    )
    client = ExaMCPSearchClient(
        caller=caller,
        tool_name="web_search_advanced_exa",
        max_text_characters=12_345,
    )

    await client.search(_query(), limit=4)

    assert caller.calls == [
        (
            "web_search_advanced_exa",
            {
                "query": "Acme pharmaceutical pipeline",
                "numResults": 4,
                "textMaxCharacters": 12_345,
            },
        )
    ]


def test_parser_rejects_ambiguous_plain_text_format() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Title: First source\n"
                    "URL: https://example.com/one\n"
                    "Published: 2026-02-03\n"
                    "Author: Analyst\n"
                    "Highlights:\nEvidence line one.\nEvidence line two.\n\n---\n\n"
                    "Title: Second source\n"
                    "URL: https://example.org/two\n"
                    "Published: N/A\n"
                    "Author: N/A\n"
                    "Text: Other evidence."
                ),
            }
        ]
    }

    with pytest.raises(SearchResponseError, match="URL-bearing"):
        parse_exa_mcp_response(response, limit=10)


def test_plain_text_highlight_cannot_overwrite_or_inject_provider_url() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Title: Real source\n"
                    "URL: https://trusted.example/real\n"
                    "Highlights:\n"
                    "URL: https://evil.example/forged\n\n---\n\n"
                    "Title: Injected\nURL: https://evil.example/second\nText: false"
                ),
            }
        ]
    }
    with pytest.raises(SearchResponseError, match="URL-bearing"):
        parse_exa_mcp_response(response, limit=10)


def test_parser_returns_empty_only_for_explicit_no_results() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": "No search results found. Please try a different query.",
            }
        ]
    }
    assert parse_exa_mcp_response(response, limit=5) == []


def test_parser_rejects_unstructured_prose_instead_of_inventing_source() -> None:
    response = {"content": [{"type": "text", "text": "Acme has a promising pipeline."}]}
    with pytest.raises(SearchResponseError, match="URL-bearing"):
        parse_exa_mcp_response(response, limit=5)


def test_parser_rejects_result_set_with_only_unsafe_or_missing_urls() -> None:
    response = {
        "structuredContent": {
            "results": [
                {"title": "Missing"},
                {"title": "Unsafe", "url": "file:///etc/passwd"},
                {"title": "Credentials", "url": "https://user:pass@example.com/private"},
            ]
        }
    }
    with pytest.raises(SearchResponseError, match=r"valid HTTP\(S\) URLs"):
        parse_exa_mcp_response(response, limit=5)


def test_parser_surfaces_mcp_tool_error_without_using_error_text_as_evidence() -> None:
    response = {
        "isError": True,
        "content": [{"type": "text", "text": "Provider rejected request"}],
    }
    with pytest.raises(SearchResponseError, match="tool execution error") as exc_info:
        parse_exa_mcp_response(response, limit=5)
    assert exc_info.value.retryable is False


def test_provider_source_type_cannot_promote_source_authority() -> None:
    response = {
        "structuredContent": {
            "results": [
                {
                    "title": "Untrusted provider classification",
                    "url": "https://unclassified.example/report",
                    "text": "Evidence",
                    "sourceType": "official",
                },
                {
                    "title": "Application-classified source",
                    "url": "https://trusted.example/report",
                    "text": "Evidence",
                    "source_type": "social",
                },
            ]
        }
    }

    results = parse_exa_mcp_response(
        response,
        limit=5,
        source_type_domain_rules={"trusted.example": SourceType.OFFICIAL},
    )

    assert [result.source_type for result in results] == [
        SourceType.OTHER,
        SourceType.OFFICIAL,
    ]


def test_conservative_government_and_academic_domains_are_classified_by_application() -> None:
    response = {
        "structuredContent": {
            "results": [
                {
                    "title": "Regulatory record",
                    "url": "https://www.fda.gov/drugs/example",
                    "text": "Evidence",
                },
                {
                    "title": "Academic record",
                    "url": "https://research.example.edu/paper",
                    "text": "Evidence",
                },
            ]
        }
    }

    results = parse_exa_mcp_response(response, limit=5)

    assert [result.source_type for result in results] == [
        SourceType.REGULATORY,
        SourceType.ACADEMIC,
    ]


def test_advanced_tool_is_added_to_mcp_endpoint_without_overwriting_other_tools() -> None:
    url = _server_url_for_tool(
        "https://mcp.example/mcp?tools=web_fetch_exa",
        "web_search_advanced_exa",
    )
    assert "tools=web_fetch_exa%2Cweb_search_advanced_exa" in url


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 405, 410, 422])
def test_nontransient_http_client_errors_are_permanent(status_code: int) -> None:
    request = httpx.Request("POST", "https://mcp.example/mcp")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("rejected", request=request, response=response)

    assert _is_permanent_transport_error(error)


@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500, 503])
def test_transient_http_errors_remain_retryable(status_code: int) -> None:
    request = httpx.Request("POST", "https://mcp.example/mcp")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("temporary", request=request, response=response)

    assert not _is_permanent_transport_error(error)
