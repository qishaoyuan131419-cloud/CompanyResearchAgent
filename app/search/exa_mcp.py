import asyncio
import importlib
import json
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.core.enums import SourceType
from app.core.exceptions import ConfigurationError
from app.schemas.planning import SearchQuery
from app.schemas.search import SearchResult
from app.search.errors import RetryableSearchError, SearchResponseError

_NO_RESULTS_PREFIX = "no search results found"
_DEFAULT_MAX_TEXT_CHARACTERS = 100_000
_ADVANCED_SEARCH_TOOL = "web_search_advanced_exa"


class MCPToolCaller(Protocol):
    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class StreamableHTTPMCPToolCaller:
    """Small MCP SDK boundary for one Streamable HTTP tool call.

    SDK imports are intentionally lazy so fake-backed unit tests and cache-only
    processes do not require MCP transport initialization.
    """

    def __init__(self, server_url: str, *, api_key: str | None = None) -> None:
        self._server_url = _validated_server_url(server_url)
        self._api_key = api_key
        self._verified_tools: set[str] = set()
        self._verification_lock = asyncio.Lock()

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            mcp_module = importlib.import_module("mcp")
            transport_module = importlib.import_module("mcp.client.streamable_http")
            client_session = mcp_module.ClientSession
            transport_factory = getattr(
                transport_module,
                "streamable_http_client",
                getattr(transport_module, "streamablehttp_client", None),
            )
            if transport_factory is None:
                raise AttributeError("MCP Streamable HTTP client is unavailable")
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            raise ConfigurationError("the installed MCP SDK lacks Streamable HTTP support") from exc

        try:
            headers = {"User-Agent": "company-research-agent/0.1"}
            if self._api_key:
                headers["x-api-key"] = self._api_key
            timeout = httpx.Timeout(timeout=None, connect=10.0)
            async with httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
            ) as http_client:
                async with transport_factory(
                    self._server_url,
                    http_client=http_client,
                ) as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with client_session(read_stream, write_stream) as session:
                        await session.initialize()
                        await self._verify_tool(session, name)
                        return await session.call_tool(name, arguments=dict(arguments))
        except (ConfigurationError, SearchResponseError):
            raise
        except Exception as exc:
            if _is_permanent_transport_error(exc):
                raise SearchResponseError(
                    "Exa MCP authentication or request was permanently rejected"
                ) from exc
            raise RetryableSearchError("Exa MCP transport request failed") from exc

    async def _verify_tool(self, session: Any, name: str) -> None:
        async with self._verification_lock:
            if name in self._verified_tools:
                return
            cursor: str | None = None
            while True:
                response = await session.list_tools(cursor=cursor)
                for tool in response.tools:
                    if tool.name != name:
                        continue
                    schema = tool.inputSchema
                    properties = schema.get("properties") if isinstance(schema, dict) else None
                    if not isinstance(properties, dict) or "query" not in properties:
                        raise SearchResponseError(
                            f"configured Exa tool {name!r} has an incompatible input schema"
                        )
                    self._verified_tools.add(name)
                    return
                cursor = response.nextCursor
                if not cursor:
                    break
            raise SearchResponseError(
                f"configured Exa tool {name!r} is not exposed by the MCP server"
            )


class ExaMCPSearchClient:
    """``SearchClient`` adapter for Exa's MCP search tool."""

    def __init__(
        self,
        *,
        tool_name: str,
        caller: MCPToolCaller | None = None,
        server_url: str | None = None,
        api_key: str | None = None,
        source_type_domain_rules: Mapping[str, SourceType] | None = None,
        max_text_characters: int = _DEFAULT_MAX_TEXT_CHARACTERS,
    ) -> None:
        if not tool_name.strip():
            raise ConfigurationError("Exa MCP search tool name cannot be empty")
        if max_text_characters < 1:
            raise ValueError("max_text_characters must be at least 1")
        if caller is None:
            if not server_url:
                raise ConfigurationError("Exa MCP URL is not configured")
            caller = StreamableHTTPMCPToolCaller(
                _server_url_for_tool(server_url, tool_name),
                api_key=api_key,
            )
        self._caller = caller
        self._tool_name = tool_name
        self._source_type_domain_rules = dict(source_type_domain_rules or {})
        self._max_text_characters = max_text_characters

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        caller: MCPToolCaller | None = None,
    ) -> "ExaMCPSearchClient":
        api_key = settings.exa_api_key.get_secret_value() if settings.exa_api_key else None
        return cls(
            tool_name=settings.exa_search_tool,
            caller=caller,
            server_url=settings.exa_mcp_url,
            api_key=api_key,
            source_type_domain_rules=settings.source_type_domain_rules,
            max_text_characters=settings.exa_max_text_characters,
        )

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        arguments: dict[str, Any] = {"query": query.query, "numResults": limit}
        if self._tool_name == _ADVANCED_SEARCH_TOOL:
            arguments["textMaxCharacters"] = self._max_text_characters
        try:
            response = await self._caller.call_tool(
                self._tool_name,
                arguments,
            )
        except (ConfigurationError, RetryableSearchError, SearchResponseError):
            raise
        except Exception as exc:
            raise RetryableSearchError("Exa MCP tool call failed") from exc
        return parse_exa_mcp_response(
            response,
            limit=limit,
            source_type_domain_rules=self._source_type_domain_rules,
            max_text_characters=self._max_text_characters,
        )


ExaSearchClient = ExaMCPSearchClient


def parse_exa_mcp_response(
    response: Any,
    *,
    limit: int,
    source_type_domain_rules: Mapping[str, SourceType] | None = None,
    max_text_characters: int = _DEFAULT_MAX_TEXT_CHARACTERS,
) -> list[SearchResult]:
    """Safely parse structured or official text-formatted Exa MCP responses.

    Records without an explicit provider-supplied HTTP(S) URL are discarded. The
    parser never synthesizes a URL or turns arbitrary prose into an evidence source.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1")
    if _read_field(response, "isError", "is_error") is True:
        raise SearchResponseError("Exa MCP reported a tool execution error")

    structured = _read_field(response, "structuredContent", "structured_content")
    if structured is not None:
        items = _find_result_items(structured)
        if items is not None:
            return _parse_items(
                items,
                limit=limit,
                source_type_domain_rules=source_type_domain_rules,
                max_text_characters=max_text_characters,
            )

    direct_items = _find_result_items(response)
    if direct_items is not None:
        return _parse_items(
            direct_items,
            limit=limit,
            source_type_domain_rules=source_type_domain_rules,
            max_text_characters=max_text_characters,
        )

    texts = _extract_text_blocks(response)
    if not texts:
        raise SearchResponseError("Exa MCP response contained no parseable content")

    parsed_items: list[Mapping[str, Any]] = []
    saw_explicit_no_results = False
    for text in texts:
        stripped = text.strip()
        if stripped.casefold().startswith(_NO_RESULTS_PREFIX):
            saw_explicit_no_results = True
            continue
        json_items = _items_from_json_text(stripped)
        if json_items is not None:
            parsed_items.extend(json_items)

    if not parsed_items:
        if saw_explicit_no_results:
            return []
        raise SearchResponseError("Exa MCP text response contained no URL-bearing results")
    return _parse_items(
        parsed_items,
        limit=limit,
        source_type_domain_rules=source_type_domain_rules,
        max_text_characters=max_text_characters,
    )


def _parse_items(
    items: list[Any],
    *,
    limit: int,
    source_type_domain_rules: Mapping[str, SourceType] | None,
    max_text_characters: int,
) -> list[SearchResult]:
    if not items:
        return []
    results: list[SearchResult] = []
    for item in items:
        result = _parse_item(
            item,
            source_type_domain_rules=source_type_domain_rules,
            max_text_characters=max_text_characters,
        )
        if result is not None:
            results.append(result)
        if len(results) >= limit:
            break
    if not results:
        raise SearchResponseError("Exa MCP returned results without any valid HTTP(S) URLs")
    return results


def _parse_item(
    item: Any,
    *,
    source_type_domain_rules: Mapping[str, SourceType] | None,
    max_text_characters: int,
) -> SearchResult | None:
    if not isinstance(item, Mapping):
        return None

    raw_url = _first_value(item, "url", "link")
    if not isinstance(raw_url, str) or not _is_safe_source_url(raw_url):
        return None

    title = _string_value(_first_value(item, "title", "name"))
    author = _optional_string(_first_value(item, "author", "byline"))
    published_at = _first_value(
        item,
        "published_at",
        "publishedAt",
        "publishedDate",
        "published_date",
    )
    score = _number_value(_first_value(item, "score", "relevanceScore"))
    source_type = _source_type(raw_url, source_type_domain_rules or {})
    text = _result_text(item)[:max_text_characters]

    values: dict[str, Any] = {
        "title": title,
        "url": raw_url,
        "text": text,
        "published_at": published_at,
        "author": author,
        "score": score,
        "source_type": source_type,
    }
    try:
        return SearchResult.model_validate(values)
    except ValidationError:
        values["published_at"] = None
        try:
            return SearchResult.model_validate(values)
        except ValidationError:
            return None


def _find_result_items(value: Any, *, depth: int = 0) -> list[Any] | None:
    if depth > 3:
        return None
    if isinstance(value, list):
        return value
    if not isinstance(value, Mapping):
        return None

    for key in ("results", "items"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return candidate
    for key in ("data", "output", "result"):
        candidate = value.get(key)
        if candidate is not None:
            nested = _find_result_items(candidate, depth=depth + 1)
            if nested is not None:
                return nested
    return None


def _extract_text_blocks(response: Any) -> list[str]:
    content = _read_field(response, "content")
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        block_type = _read_field(block, "type")
        text = _read_field(block, "text")
        if block_type == "text" and isinstance(text, str):
            texts.append(text)
    return texts


def _items_from_json_text(text: str) -> list[Any] | None:
    if not text or text[0] not in "[{":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _find_result_items(payload)


def _result_text(item: Mapping[str, Any]) -> str:
    for key in ("text", "summary", "snippet"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        return "\n".join(value for value in highlights if isinstance(value, str))
    return ""


def _read_field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _first_value(item: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return None


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.casefold() == "n/a":
        return None
    return stripped


def _number_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _source_type(
    url: str,
    domain_rules: Mapping[str, SourceType],
) -> SourceType:
    host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    normalized_rules = sorted(
        (
            (domain.casefold().removeprefix("www.").strip("."), source_type)
            for domain, source_type in domain_rules.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for domain, source_type in normalized_rules:
        if host == domain or host.endswith(f".{domain}"):
            return source_type
    if host.endswith(".gov") or host == "gov":
        return SourceType.REGULATORY
    if host.endswith(".edu") or host == "edu":
        return SourceType.ACADEMIC
    return SourceType.OTHER


def _is_safe_source_url(value: str) -> bool:
    if any(character.isspace() or ord(character) < 32 for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
    )


def _is_permanent_transport_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            if _is_permanent_http_status(current.response.status_code):
                return True
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and _is_permanent_http_status(status_code):
            return True
        message = " ".join(str(current).casefold().split())
        if any(
            marker in message
            for marker in (
                "invalid api key",
                "unauthorized",
                "forbidden",
                "(401)",
                "http 401",
                "http 403",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_permanent_http_status(status_code: int) -> bool:
    return 400 <= status_code < 500 and status_code not in {408, 409, 425, 429}


def _validated_server_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError as exc:
        raise ConfigurationError("Exa MCP URL is invalid") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("Exa MCP URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("Exa MCP URL cannot contain user information")

    return value.strip()


def _server_url_for_tool(value: str, tool_name: str) -> str:
    parsed = urlsplit(_validated_server_url(value))
    query = parse_qsl(parsed.query, keep_blank_values=True)
    tools: list[str] = []
    retained: list[tuple[str, str]] = []
    for key, item in query:
        if key.casefold() == "tools":
            tools.extend(part.strip() for part in item.split(",") if part.strip())
        else:
            retained.append((key, item))
    if tool_name not in tools:
        tools.append(tool_name)
    retained.append(("tools", ",".join(dict.fromkeys(tools))))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(retained), parsed.fragment)
    )
