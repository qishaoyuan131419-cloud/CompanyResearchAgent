from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import StructuredOutputError

TModel = TypeVar("TModel", bound=BaseModel)
# Bare hostnames are allowed for explicitly stated company domains (for example,
# ``pfizer.com`` or ``www.pfizer.com``). Reject transport URLs and hostnames
# carrying a path, which are the URL-shaped values the LLM must not provide.
_URL_PATTERN = re.compile(r"(?i)(?:https?://|\bwww\.[^\s/]+/)")


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
            # Some response models intentionally contain provider-validated
            # URL fields (for example ResolvedCompany.website). Those fields
            # are sanitized against evidence lineage downstream; URL-shaped
            # strings elsewhere remain prohibited.
            if key in {"website", "quote"}:
                continue
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
    item_level_recovery = bool(
        getattr(response_model, "allow_prohibited_urls_for_item_recovery", False)
    )
    if (
        reject_urls
        and not item_level_recovery
        and (url_path := _find_url_path(payload)) is not None
    ):
        raise StructuredOutputError(f"LLM response contained a prohibited URL at {url_path}")

    # Re-serialize before strict validation so Pydantic applies its JSON-mode rules
    # (for example, string-valued enums) without allowing Python-side coercions.
    canonical_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        return response_model.model_validate_json(canonical_json, strict=True)
    except ValidationError as exc:
        summary = _validation_summary(exc)
        raise StructuredOutputError(f"LLM response failed schema validation: {summary}") from exc
