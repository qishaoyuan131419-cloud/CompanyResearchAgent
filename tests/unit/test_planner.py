import pytest
from pydantic import ValidationError

from app.planner.service import ResearchPlanner
from app.schemas.planning import SearchQuery
from app.utils.text import normalized_fingerprint_text


def query(text: str, topic: str = "Products") -> SearchQuery:
    return SearchQuery(
        query_id="llm-generated",
        query=text,
        topic=topic,
        priority=5,
        expected_evidence="Official product information",
    )


def test_query_preparation_filters_history_and_in_batch_duplicates() -> None:
    history = {normalized_fingerprint_text("Acme pipeline")}
    prepared = ResearchPlanner.prepare_queries(
        [
            query(" ACME   PIPELINE "),
            query("Acme products"),
            query(" acme  products "),
            query("Acme technology", topic="Technology"),
        ],
        history,
    )
    assert [item.query for item in prepared] == ["Acme products", "Acme technology"]
    assert all(item.query_id.startswith("qry_") for item in prepared)


def test_failed_query_history_is_still_filtered() -> None:
    previous = normalized_fingerprint_text("Acme manufacturing")
    assert ResearchPlanner.prepare_queries([query("acme MANUFACTURING")], {previous}) == []


def test_query_preparation_filters_trivial_paraphrases_but_preserves_distinct_intents() -> None:
    prepared = ResearchPlanner.prepare_queries(
        [
            query("Pfizer official company pipeline clinical trials"),
            query("Pfizer clinical trial pipeline official"),
            query("Pfizer SEC ticker and incorporation", topic="Company Identity"),
        ],
        set(),
    )

    assert [item.query for item in prepared] == [
        "Pfizer official company pipeline clinical trials",
        "Pfizer SEC ticker and incorporation",
    ]


def test_search_query_rejects_embedded_url() -> None:
    with pytest.raises(ValidationError, match="must not contain URLs"):
        query("Acme site https://fabricated.example")
