from __future__ import annotations

from enum import StrEnum

import pytest
from pydantic import BaseModel, ConfigDict

from app.core.exceptions import StructuredOutputError
from app.llm.parsing import parse_structured_output


class Decision(StrEnum):
    KEEP = "keep"


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    count: int
    decision: Decision


def test_parse_structured_output_accepts_exact_json_schema() -> None:
    result = parse_structured_output(
        '{"statement":"supported fact","count":2,"decision":"keep"}', Answer
    )

    assert result == Answer(statement="supported fact", count=2, decision=Decision.KEEP)


@pytest.mark.parametrize(
    "payload",
    [
        '```json\n{"statement":"x","count":1,"decision":"keep"}\n```',
        '{"statement":"x","count":1,"decision":"keep"} trailing',
        '{"statement":"x","statement":"y","count":1,"decision":"keep"}',
        '{"statement":"x","count":NaN,"decision":"keep"}',
        '[{"statement":"x","count":1,"decision":"keep"}]',
        '{"statement":"x","count":"1","decision":"keep"}',
        '{"statement":"x","count":1,"decision":"keep","extra":true}',
    ],
)
def test_parse_structured_output_rejects_non_strict_or_schema_invalid_json(payload: str) -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured_output(payload, Answer)


@pytest.mark.parametrize(
    "value",
    ["https://example.test/source", "HTTP://example.test", "see www.example.test/source"],
)
def test_parse_structured_output_rejects_url_shaped_strings(value: str) -> None:
    payload = f'{{"statement":"{value}","count":1,"decision":"keep"}}'

    with pytest.raises(StructuredOutputError, match="prohibited URL"):
        parse_structured_output(payload, Answer)


def test_parse_error_never_echoes_untrusted_provider_text() -> None:
    secret = "SENSITIVE_UNTRUSTED_SOURCE_TEXT"

    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_output(f"not-json-{secret}", Answer)

    assert secret not in str(caught.value)
