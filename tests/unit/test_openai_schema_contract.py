from typing import Any

from app.evidence.processor import ClaimExtractionResponse
from app.llm.openai_compatible import to_openai_strict_schema
from app.schemas.company import ResolvedCompany
from app.schemas.planning import QueryPlan, ResearchPlan
from app.schemas.reflection import FollowupPlan, InformationAssessment, ResearchSummary
from app.schemas.research import ResearchReport

_RESPONSE_MODELS = (
    ResolvedCompany,
    ResearchPlan,
    QueryPlan,
    ClaimExtractionResponse,
    ResearchSummary,
    InformationAssessment,
    FollowupPlan,
    ResearchReport,
)


def _assert_strict_subset(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _assert_strict_subset(item)
        return
    if not isinstance(value, dict):
        return
    assert "default" not in value
    assert "format" not in value
    properties = value.get("properties")
    if isinstance(properties, dict):
        assert value.get("additionalProperties") is False
        assert value.get("required") == list(properties)
    for item in value.values():
        _assert_strict_subset(item)


def test_every_llm_response_schema_normalizes_to_openai_strict_contract() -> None:
    for response_model in _RESPONSE_MODELS:
        normalized = to_openai_strict_schema(response_model.model_json_schema(mode="validation"))
        _assert_strict_subset(normalized)
