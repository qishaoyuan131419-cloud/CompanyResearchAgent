from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import StructuredOutputError

TModel = TypeVar("TModel", bound=BaseModel)
_URL_PATTERN = re.compile(r"(?i)(?:https?://|\bwww\.)")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _object_without_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _find_url_path(value: Any, *, path: str = "$") -> str | None:
    if isinstance(value, str):
        return path if _URL_PATTERN.search(value) else None
    if isinstance(value, list):
        for index, item in enumerate(value):
            match = _find_url_path(item, path=f"{path}[{index}]")
            if match is not None:
                return match
    elif isinstance(value, dict):
        for key, item in value.items():
            match = _find_url_path(item, path=f"{path}.{key}")
            if match is not None:
                return match
    return None


def _validation_summary(error: ValidationError) -> str:
    summaries: list[str] = []
    for detail in error.errors(include_url=False, include_context=False)[:5]:
        location = ".".join(str(part) for part in detail.get("loc", ())) or "$"
        summaries.append(f"{location}: {detail.get('type', 'validation_error')}")
    suffix = "" if error.error_count() <= 5 else f" (+{error.error_count() - 5} more)"
    return "; ".join(summaries) + suffix


def parse_structured_output(
    json_text: str,
    response_model: type[TModel],
    *,
    reject_urls: bool = True,
) -> TModel:
    """Parse exactly one standards-compliant JSON object and validate it strictly.

    Markdown fences, duplicate keys, NaN/Infinity, trailing prose, schema coercions,
    and URL-shaped strings are deliberately rejected. Provider output is never
    echoed in an exception, which prevents untrusted source text leaking to logs.
    """

    if not isinstance(json_text, str) or not json_text.strip():
        raise StructuredOutputError("LLM returned an empty structured response")
    try:
        payload = json.loads(
            json_text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StructuredOutputError("LLM response was not strict JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError("LLM structured response must be a JSON object")
    if reject_urls and (url_path := _find_url_path(payload)) is not None:
        raise StructuredOutputError(f"LLM response contained a prohibited URL at {url_path}")

    # Re-serialize before strict validation so Pydantic applies its JSON-mode rules
    # (for example, string-valued enums) without allowing Python-side coercions.
    canonical_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        return response_model.model_validate_json(canonical_json, strict=True)
    except ValidationError as exc:
        summary = _validation_summary(exc)
        raise StructuredOutputError(f"LLM response failed schema validation: {summary}") from exc
