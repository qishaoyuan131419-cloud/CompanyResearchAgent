from datetime import UTC, datetime

from app.core.enums import SourceType
from app.evidence.registry import SourceRegistry
from app.schemas.search import (
    QueryExecution,
    SearchBatch,
    SearchResult,
    SearchStatistics,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


def make_result(
    url: str,
    text: str,
    *,
    title: str = "Source title",
    source_type: SourceType = SourceType.NEWS,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,
        text=text,
        source_type=source_type,
    )


def make_batch(*query_results: tuple[str, list[SearchResult]]) -> SearchBatch:
    executions = [
        QueryExecution(
            query_id=query_id,
            query=query_id,
            started_at=NOW,
            completed_at=NOW,
            duration_ms=0,
            attempts=1,
            results=results,
        )
        for query_id, results in query_results
    ]
    return SearchBatch(
        executions=executions,
        statistics=SearchStatistics(
            query_count=len(executions),
            successful_queries=len(executions),
            failed_queries=0,
            cache_hits=0,
            total_results=sum(len(results) for _, results in query_results),
            total_duration_ms=0,
        ),
    )


def test_evidence_registry_deduplicates_url_variants_across_rounds() -> None:
    registry = SourceRegistry(clock=FixedClock())

    first = registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://Example.com/company/?utm_source=newsletter#team",
                        "Acme develops medicine X.",
                    )
                ],
            )
        )
    )
    second = registry.register_batch(
        make_batch(
            (
                "q2",
                [make_result("https://example.com/company", "Updated page content")],
            )
        )
    )

    assert len(registry) == 1
    assert len(first.new_source_ids) == 1
    assert second.new_source_ids == ()
    assert second.duplicate_count == 1
    source = registry.sources[0]
    assert str(source.url) == "https://example.com/company"
    assert source.query_ids == ["q1", "q2"]


def test_evidence_registry_deduplicates_normalized_content_across_urls() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registered = registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://news.example/acme",
                        "Acme\n  completed   its trial.",
                    ),
                    make_result(
                        "https://official.example/release",
                        "acme completed its trial.",
                        source_type=SourceType.OFFICIAL,
                    ),
                ],
            )
        )
    )

    assert len(registry) == 1
    assert len(registered.new_source_ids) == 1
    assert registered.duplicate_count == 1
    source = registry.sources[0]
    assert source.source_type == SourceType.OFFICIAL
    assert str(source.url) == "https://official.example/release"


def test_evidence_registry_does_not_over_deduplicate_changed_fact() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result("https://a.example/item", "The trial is Phase 2."),
                    make_result("https://b.example/item", "The trial is Phase 3."),
                ],
            )
        )
    )

    assert len(registry) == 2


def test_evidence_registry_does_not_merge_empty_documents_by_content() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result("https://a.example/item", ""),
                    make_result("https://b.example/item", "  \n "),
                ],
            )
        )
    )

    assert len(registry) == 2


def test_evidence_registry_is_deterministic_for_result_order() -> None:
    results = [
        make_result(
            "https://z.example/story",
            "Shared source body",
            title="Z syndication",
            source_type=SourceType.NEWS,
        ),
        make_result(
            "https://a.example/release",
            " shared   source body ",
            title="Official release",
            source_type=SourceType.OFFICIAL,
        ),
    ]
    first = SourceRegistry(clock=FixedClock())
    second = SourceRegistry(clock=FixedClock())

    first.register_batch(make_batch(("q1", results)))
    second.register_batch(make_batch(("q1", list(reversed(results)))))

    assert [source.model_dump(mode="json") for source in first.sources] == [
        source.model_dump(mode="json") for source in second.sources
    ]


def test_evidence_registry_late_bridge_preserves_immutable_source_lineage() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result("https://a.example/page", "First body"),
                    make_result("https://b.example/page", "Second body"),
                ],
            )
        )
    )
    original_ids = [source.source_id for source in registry.sources]

    bridge = registry.register_batch(
        make_batch(
            (
                "q2",
                [make_result("https://a.example/page", "Second body")],
            )
        )
    )

    assert len(registry) == 2
    assert bridge.new_source_ids == ()
    assert bridge.duplicate_count == 1
    assert registry.resolve_id(original_ids[0]) != registry.resolve_id(original_ids[1])
    source_a = next(source for source in registry.sources if "a.example" in str(source.url))
    source_b = next(source for source in registry.sources if "b.example" in str(source.url))
    assert source_a.content == "First body"
    assert source_a.query_ids == ["q1", "q2"]
    assert source_b.content == "Second body"
    assert source_b.query_ids == ["q1"]


def test_late_content_duplicate_cannot_rewrite_original_source_metadata() -> None:
    registry = SourceRegistry(clock=FixedClock())
    registry.register_batch(
        make_batch(
            (
                "q1",
                [
                    make_result(
                        "https://news.example/story",
                        "Shared body",
                        title="Original news story",
                        source_type=SourceType.NEWS,
                    )
                ],
            )
        )
    )

    registry.register_batch(
        make_batch(
            (
                "q2",
                [
                    make_result(
                        "https://official.example/release",
                        "Shared body",
                        title="Later official copy",
                        source_type=SourceType.OFFICIAL,
                    )
                ],
            )
        )
    )

    source = registry.sources[0]
    assert str(source.url) == "https://news.example/story"
    assert source.title == "Original news story"
    assert source.source_type == SourceType.NEWS
    assert source.query_ids == ["q1", "q2"]
