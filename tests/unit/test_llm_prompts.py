from __future__ import annotations

from pathlib import Path

import pytest

from app.prompts.repository import FilePromptRepository

_PROMPT_DIRECTORY = Path(__file__).parents[2] / "app" / "prompts" / "templates"


@pytest.mark.parametrize(
    ("name", "variables"),
    [
        ("resolver", {"company_snapshot": "{}", "source_documents": "[]"}),
        (
            "planner",
            {"task": "topics", "resolved_company": "{}", "research_plan": "[]"},
        ),
        (
            "extractor",
            {
                "resolved_company": "{}",
                "source_documents": "[]",
                "existing_claims": "[]",
            },
        ),
        (
            "reflection",
            {"resolved_company": "{}", "evidence_bundle": "{}", "prior_queries": "[]"},
        ),
        (
            "followup",
            {"resolved_company": "{}", "assessment": "{}", "prior_queries": "[]"},
        ),
        (
            "research",
            {"resolved_company": "{}", "evidence_bundle": "{}", "assessment": "{}"},
        ),
    ],
)
def test_required_prompts_render_and_enforce_evidence_boundaries(
    name: str, variables: dict[str, str]
) -> None:
    prompt = FilePromptRepository(_PROMPT_DIRECTORY).render(name, variables)
    lowered = prompt.lower()

    assert "$" not in prompt
    assert "source_id" in lowered or "source id" in lowered
    assert "url" in lowered
    assert "untrusted" in lowered
    assert "invent" in lowered or "guess" in lowered


def test_planner_template_supports_topics_and_query_generation() -> None:
    repository = FilePromptRepository(_PROMPT_DIRECTORY)

    topics = repository.render(
        "planner", {"task": "topics", "resolved_company": "{}", "research_plan": "[]"}
    )
    queries = repository.render(
        "planner",
        {"task": "queries", "resolved_company": "{}", "research_plan": "[topic]"},
    )

    assert "Task mode: topics" in topics
    assert "Task mode: queries" in queries
    assert "When `task` is `topics`" in topics
    assert "When `task` is `queries`" in queries


def test_extractor_forbids_free_form_inference() -> None:
    prompt = FilePromptRepository(_PROMPT_DIRECTORY).render(
        "extractor",
        {
            "resolved_company": "{}",
            "source_documents": "[]",
            "existing_claims": "[claim_1]",
        },
    )

    assert "claim_1" in prompt
    lowered = prompt.casefold()
    assert "free-form model inference is rejected" in lowered
    assert "non-negated predicate-value relation" in lowered
    assert "target company" in lowered
