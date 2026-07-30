"""Persistent, deterministic registry for normalized search sources."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.protocols import Clock
from app.schemas.search import SearchBatch, SearchResult, SourceDocument
from app.utils.hashing import stable_hash
from app.utils.text import normalize_text, normalized_fingerprint_text
from app.utils.urls import canonicalize_url


@dataclass(frozen=True, slots=True)
class RegisteredSources:
    """Outcome of adding one search batch to a registry."""

    new_source_ids: tuple[str, ...]
    duplicate_count: int


class SourceRegistry:
    """Normalizes and deduplicates sources across all rounds of one run.

    URL and non-empty normalized-content fingerprints are both indexes into the
    same registry. Registered URL snapshots are immutable: a later observation
    cannot bridge two prior sources or replace citation content. New URLs with
    identical content still deduplicate deterministically.
    """

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock
        self._sources: dict[str, SourceDocument] = {}
        self._url_index: dict[str, str] = {}
        self._content_index: dict[str, str] = {}
        self._aliases: dict[str, str] = {}

    def register_batch(self, batch: SearchBatch) -> RegisteredSources:
        entries = [
            (execution.query_id, result)
            for execution in batch.executions
            for result in execution.results
        ]
        entries.sort(key=lambda item: self._result_sort_key(item[0], item[1]))

        roots_before = set(self._root_ids())
        duplicate_count = 0
        touched_ids: set[str] = set()
        mutable_roots: set[str] = set()
        for query_id, result in entries:
            source_id, is_duplicate = self.register(
                query_id=query_id,
                result=result,
                mutable_roots=mutable_roots,
            )
            touched_ids.add(source_id)
            duplicate_count += int(is_duplicate)
            resolved = self.resolve_id(source_id)
            if resolved not in roots_before:
                mutable_roots.add(resolved)

        roots_after = set(self._root_ids())
        new_ids = {
            self.resolve_id(source_id)
            for source_id in touched_ids
            if self.resolve_id(source_id) not in roots_before
        }
        # A union can replace an old root with a lexicographically smaller root.
        # Such a connected component is not new merely because its root changed.
        for source_id in tuple(new_ids):
            source = self.get(source_id)
            if any(
                self._same_registered_component(source, old_id)
                for old_id in roots_before
                if old_id not in roots_after
            ):
                new_ids.discard(source_id)

        return RegisteredSources(
            new_source_ids=tuple(sorted(new_ids)),
            duplicate_count=duplicate_count,
        )

    def register(
        self,
        *,
        query_id: str,
        result: SearchResult,
        mutable_roots: set[str] | None = None,
    ) -> tuple[str, bool]:
        canonical_url = canonicalize_url(str(result.url))
        content = normalize_text(result.text)
        content_hash = stable_hash(normalized_fingerprint_text(content), prefix="cnt_", length=32)

        url_match = self._url_index.get(canonical_url)
        resolved_url_match = self.resolve_id(url_match) if url_match is not None else None
        resolved_content_match: str | None = None
        if content:
            content_match = self._content_index.get(content_hash)
            if content_match is not None:
                resolved_content_match = self.resolve_id(content_match)

        candidate_id = self._candidate_source_id(
            canonical_url=canonical_url,
            content=content,
            content_hash=content_hash,
        )
        candidate = SourceDocument(
            source_id=candidate_id,
            title=normalize_text(result.title),
            url=canonical_url,
            published_at=result.published_at,
            retrieved_at=self._clock.now(),
            source_type=result.source_type,
            summary=content[:500],
            content=content,
            content_hash=content_hash,
            query_ids=[query_id],
        )

        # A URL is an immutable citation snapshot once registered. If the same
        # URL later serves different content (including content already seen at
        # another URL), retain the original document and only add the query ID.
        # This prevents a bridge observation from rewriting old quote lineage.
        if resolved_url_match is not None:
            existing = self._sources[resolved_url_match]
            self._sources[resolved_url_match] = existing.model_copy(
                update={"query_ids": sorted({*existing.query_ids, query_id})}
            )
            self._url_index[canonical_url] = resolved_url_match
            if content_hash == existing.content_hash:
                self._content_index[content_hash] = resolved_url_match
            return resolved_url_match, True

        is_duplicate = resolved_content_match is not None
        if resolved_content_match is None:
            target_id = candidate_id
            self._sources[target_id] = candidate
        else:
            target_id = resolved_content_match
            existing = self._sources[target_id]
            if mutable_roots is not None and target_id in mutable_roots:
                self._sources[target_id] = self._merge_documents(
                    target_id,
                    [candidate, existing],
                )
            else:
                self._sources[target_id] = existing.model_copy(
                    update={"query_ids": sorted({*existing.query_ids, query_id})}
                )

        self._url_index[canonical_url] = target_id
        if content:
            self._content_index[content_hash] = target_id
        return target_id, is_duplicate

    def resolve_id(self, source_id: str) -> str:
        path: list[str] = []
        current = source_id
        while current in self._aliases:
            path.append(current)
            current = self._aliases[current]
        for alias in path:
            self._aliases[alias] = current
        return current

    def contains(self, source_id: str) -> bool:
        return self.resolve_id(source_id) in self._sources

    def get(self, source_id: str) -> SourceDocument:
        return self._sources[self.resolve_id(source_id)]

    @property
    def sources(self) -> list[SourceDocument]:
        return [self._sources[source_id] for source_id in sorted(self._root_ids())]

    def __len__(self) -> int:
        return len(self._sources)

    def _root_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))

    def _same_registered_component(self, source: SourceDocument, old_id: str) -> bool:
        try:
            return self.resolve_id(old_id) == source.source_id
        except KeyError:
            return False

    @staticmethod
    def _candidate_source_id(*, canonical_url: str, content: str, content_hash: str) -> str:
        identity = {"content_hash": content_hash} if content else {"url": canonical_url}
        return stable_hash(identity, prefix="src_", length=24)

    @staticmethod
    def _result_sort_key(query_id: str, result: SearchResult) -> tuple[str, str, str, str]:
        return (
            canonicalize_url(str(result.url)),
            normalized_fingerprint_text(result.text),
            normalized_fingerprint_text(result.title),
            query_id,
        )

    @classmethod
    def _merge_documents(cls, source_id: str, documents: list[SourceDocument]) -> SourceDocument:
        representative = min(documents, key=cls._document_preference)
        query_ids = sorted({query_id for document in documents for query_id in document.query_ids})
        retrieved_at = min(document.retrieved_at for document in documents)
        return representative.model_copy(
            update={
                "source_id": source_id,
                "query_ids": query_ids,
                "retrieved_at": retrieved_at,
            }
        )

    @staticmethod
    def _document_preference(document: SourceDocument) -> tuple[int, int, str, str]:
        source_type_rank = {
            "regulatory": 0,
            "official": 1,
            "academic": 2,
            "database": 3,
            "industry": 4,
            "news": 5,
            "social": 6,
            "other": 7,
        }
        return (
            source_type_rank[document.source_type.value],
            -len(document.content),
            canonicalize_url(str(document.url)),
            normalized_fingerprint_text(document.title),
        )
