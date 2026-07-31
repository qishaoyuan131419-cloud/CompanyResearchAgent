import json
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from pydantic import AnyHttpUrl

from app.core.enums import EvidenceStatus, IdentityStatus
from app.core.exceptions import BudgetExceededError, StructuredOutputError
from app.core.protocols import LLMClient, PromptRepository
from app.llm.errors import LLMProviderError
from app.schemas.company import CompanyCandidate, CompanySnapshot, ResolvedCompany
from app.schemas.evidence import Evidence, EvidenceBundle
from app.utils.text import (
    has_organization_legal_suffix,
    normalize_organization_name,
    normalized_fingerprint_text,
    organization_short_name,
)
from app.utils.urls import canonicalize_url, source_domain


class CompanyResolver:
    """Conservatively resolves identity; input hints alone never become confirmed facts."""

    def __init__(self, *, llm: LLMClient, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def resolve(
        self,
        snapshot: CompanySnapshot,
        evidence_bundle: EvidenceBundle | None = None,
    ) -> ResolvedCompany:
        if evidence_bundle is None:
            return self._unconfirmed(snapshot)

        prompt = self._prompts.render(
            "resolver",
            {
                "company_snapshot": "Supplied in the structured user JSON payload.",
                "source_documents": "Supplied in the structured user JSON payload.",
            },
        )
        user_payload = {
            "company_snapshot_hints": snapshot.model_dump(mode="json"),
            "evidence_bundle": evidence_bundle.model_dump(mode="json"),
        }
        try:
            result = await self._llm.generate_structured(
                system_prompt=prompt,
                user_prompt=json.dumps(user_payload, sort_keys=True, default=str),
                response_model=ResolvedCompany,
                cache_namespace="resolver",
            )
            proposed = result.value
        except BudgetExceededError:
            proposed = self._unconfirmed(snapshot)
        except (LLMProviderError, StructuredOutputError):
            proposed = self._unconfirmed(snapshot)
        return self._sanitize_with_evidence(snapshot, proposed, evidence_bundle)

    @staticmethod
    def _unconfirmed(snapshot: CompanySnapshot) -> ResolvedCompany:
        return ResolvedCompany(
            canonical_name=snapshot.canonical_name,
            identity_status=IdentityStatus.UNCONFIRMED,
            confidence=0.0,
            verification_notes=["Identity has not yet been verified against external evidence."],
        )

    def _sanitize_with_evidence(
        self,
        snapshot: CompanySnapshot,
        proposed: ResolvedCompany,
        bundle: EvidenceBundle,
    ) -> ResolvedCompany:
        by_claim: dict[str, list[Evidence]] = defaultdict(list)
        for item in bundle.evidence:
            by_claim[item.claim_id].append(item)

        supported_claim_ids = {
            claim_id
            for claim_id, items in by_claim.items()
            if any(item.status != EvidenceStatus.UNKNOWN for item in items)
        }
        all_identity_claim_ids = [
            claim_id
            for claim_id in sorted(supported_claim_ids)
            if self._is_identity_claim(by_claim[claim_id])
        ]
        identity_claim_ids = list(
            dict.fromkeys(
                claim_id
                for claim_id in proposed.claim_ids
                if claim_id in supported_claim_ids and self._is_identity_claim(by_claim[claim_id])
            )
        )
        if not identity_claim_ids:
            identity_claim_ids = list(all_identity_claim_ids)
        relationships = []
        for relationship in proposed.relationships:
            valid_relationship_claims = [
                claim_id for claim_id in relationship.claim_ids if claim_id in supported_claim_ids
            ]
            relationship_tokens = [
                token
                for token in relationship.relationship_type.replace("_", " ").casefold().split()
                if len(token) >= 4 and token not in {"company", "relationship"}
            ]
            supporting_relationship_claims = [
                claim_id
                for claim_id in valid_relationship_claims
                if self._has_exact_claim_value(
                    [claim_id],
                    by_claim,
                    relationship.related_company,
                    claim_terms=tuple(relationship_tokens),
                )
            ]
            if supporting_relationship_claims and relationship_tokens:
                status_claim_ids = (
                    self._relationship_status_claim_ids(
                        valid_relationship_claims,
                        by_claim,
                        relationship_tokens=tuple(relationship_tokens),
                        related_company=relationship.related_company,
                        status=relationship.status,
                    )
                    if relationship.status
                    else []
                )
                relationships.append(
                    relationship.model_copy(
                        update={
                            "claim_ids": list(
                                dict.fromkeys([*supporting_relationship_claims, *status_claim_ids])
                            ),
                            "status": relationship.status if status_claim_ids else None,
                        }
                    )
                )

        candidates: list[CompanyCandidate] = []
        for candidate in proposed.candidates:
            candidate_claim_ids = [
                claim_id
                for claim_id in candidate.claim_ids
                if claim_id in supported_claim_ids and self._is_identity_claim(by_claim[claim_id])
            ]
            if candidate_claim_ids and self._has_exact_claim_value(
                candidate_claim_ids,
                by_claim,
                candidate.name,
                claim_kind="name",
            ):
                candidates.append(
                    candidate.model_copy(
                        update={
                            "website": None,
                            "country": (
                                candidate.country
                                if candidate.country
                                and self._has_exact_claim_value(
                                    candidate_claim_ids,
                                    by_claim,
                                    candidate.country,
                                    claim_kind="country",
                                )
                                else None
                            ),
                            "distinguishing_details": [
                                detail
                                for detail in candidate.distinguishing_details
                                if self._has_exact_claim_value(
                                    candidate_claim_ids,
                                    by_claim,
                                    detail,
                                )
                            ],
                            "claim_ids": candidate_claim_ids,
                        }
                    )
                )
        candidates = sorted(
            {
                (
                    normalize_organization_name(candidate.name),
                    " ".join((candidate.country or "").casefold().split()),
                    tuple(
                        sorted(
                            " ".join(detail.casefold().split())
                            for detail in candidate.distinguishing_details
                        )
                    ),
                ): candidate
                for candidate in candidates
            }.values(),
            key=lambda candidate: (
                candidate.name.casefold(),
                (candidate.country or "").casefold(),
                tuple(detail.casefold() for detail in candidate.distinguishing_details),
            ),
        )

        reliable_domains_by_claim: dict[str, set[str]] = defaultdict(set)
        verified_name_domains: set[str] = set()
        verified_name_confidence = 0.0
        proposed_name = proposed.canonical_name.strip()
        source_by_id = {source.source_id: source for source in bundle.sources}
        for claim_id in identity_claim_ids:
            for item in by_claim[claim_id]:
                item_domains = {
                    source_domain(str(source_by_id[source_id].url))
                    for source_id in item.source_ids
                    if source_id in source_by_id
                }
                reliable_domains_by_claim[claim_id].update(item_domains)
                values = item.value if isinstance(item.value, list) else [item.value]
                if (
                    proposed_name
                    and item.status == EvidenceStatus.VERIFIED_FACT
                    and self._identity_claim_kind(item.claim) == "name"
                    and item_domains
                    and any(
                        isinstance(value, str)
                        and self._organization_names_match(value, proposed_name)
                        for value in values
                    )
                ):
                    verified_name_domains.update(item_domains)
                    verified_name_confidence = max(verified_name_confidence, item.confidence)

        source_count = (
            len(set().union(*reliable_domains_by_claim.values())) if identity_claim_ids else 0
        )
        has_ambiguity_conflict = any(
            not self._equivalent_name_conflict(conflict.values)
            and any(
                self._identity_claim_kind(item.claim) in {"country", "name", "website"}
                for claim_id in conflict.claim_ids
                if claim_id in by_claim
                for item in by_claim[claim_id]
            )
            for conflict in bundle.conflicts
        )
        if len(candidates) >= 2 or has_ambiguity_conflict:
            status = IdentityStatus.AMBIGUOUS
            confidence = min(proposed.confidence, 0.70)
        elif verified_name_domains:
            status = IdentityStatus.CONFIRMED
            confidence = min(max(proposed.confidence, verified_name_confidence), 0.95)
        elif identity_claim_ids:
            status = IdentityStatus.PARTIALLY_CONFIRMED
            confidence = min(proposed.confidence, 0.70)
        else:
            status = IdentityStatus.UNCONFIRMED
            confidence = min(proposed.confidence, 0.25)

        website = self._evidence_backed_website(
            snapshot,
            identity_claim_ids,
            by_claim,
            bundle,
        )

        canonical_name = self._preferred_canonical_name(
            snapshot.canonical_name,
            proposed_name,
            identity_claim_ids,
            by_claim,
        )
        evidence_name_values = self._identity_name_values(identity_claim_ids, by_claim)
        alias_candidates = [
            alias
            for alias in [
                *proposed.aliases,
                *evidence_name_values,
                organization_short_name(canonical_name),
            ]
            if self._has_exact_claim_value(
                identity_claim_ids,
                by_claim,
                alias,
                claim_kind="name",
            )
            and normalized_fingerprint_text(alias) != normalized_fingerprint_text(canonical_name)
            and self._organization_names_match(alias, canonical_name)
        ]
        aliases = sorted(
            {
                normalized_fingerprint_text(alias): min(
                    (
                        candidate
                        for candidate in alias_candidates
                        if normalized_fingerprint_text(candidate)
                        == normalized_fingerprint_text(alias)
                    ),
                    key=lambda candidate: (
                        candidate.isupper(),
                        candidate.casefold(),
                        candidate,
                    ),
                )
                for alias in alias_candidates
            }.values(),
            key=lambda alias: (alias.casefold(), alias),
        )
        country = (
            proposed.country
            if proposed.country
            and self._has_exact_claim_value(
                identity_claim_ids,
                by_claim,
                proposed.country,
                claim_kind="country",
            )
            else None
        )
        industry = (
            proposed.industry
            if proposed.industry
            and self._has_exact_claim_value(
                identity_claim_ids,
                by_claim,
                proposed.industry,
                claim_kind="industry",
            )
            else None
        )
        is_closed = (
            proposed.is_closed
            if proposed.is_closed is not None
            and self._supports_closed_value(identity_claim_ids, by_claim, proposed.is_closed)
            else None
        )
        notes = [
            f"Identity resolution retained {len(identity_claim_ids)} supported claim(s) "
            f"from {source_count} independent source domain(s)."
        ]
        if status == IdentityStatus.AMBIGUOUS:
            notes.append("Multiple evidence-linked candidate identities remain unresolved.")
        if status in {IdentityStatus.AMBIGUOUS, IdentityStatus.UNCONFIRMED}:
            website = None
            country = None
            industry = None
            is_closed = None
            aliases = []
            relationships = []

        retained_claim_ids = list(
            dict.fromkeys(
                [
                    *identity_claim_ids,
                    *all_identity_claim_ids,
                    *(claim_id for candidate in candidates for claim_id in candidate.claim_ids),
                    *(
                        claim_id
                        for relationship in relationships
                        for claim_id in relationship.claim_ids
                    ),
                ]
            )
        )

        return proposed.model_copy(
            update={
                "canonical_name": canonical_name,
                "identity_status": status,
                "confidence": confidence,
                "website": website,
                "country": country,
                "industry": industry,
                "is_closed": is_closed,
                "aliases": aliases,
                "candidates": candidates,
                "relationships": relationships,
                "claim_ids": retained_claim_ids,
                "verification_notes": notes,
            }
        )

    @staticmethod
    def _evidence_backed_website(
        snapshot: CompanySnapshot,
        valid_claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
        bundle: EvidenceBundle,
    ) -> AnyHttpUrl | None:
        identity_source_ids = {
            source_id
            for claim_id in valid_claim_ids
            for item in by_claim[claim_id]
            if item.status != EvidenceStatus.UNKNOWN
            for source_id in item.source_ids
        }
        identity_sources = [
            source for source in bundle.sources if source.source_id in identity_source_ids
        ]
        if snapshot.website is not None:
            hinted = canonicalize_url(str(snapshot.website))
            hinted_domain = source_domain(hinted)
            has_explicit_domain_claim = any(
                item.status != EvidenceStatus.UNKNOWN
                and CompanyResolver._identity_claim_kind(item.claim) == "website"
                and any(
                    CompanyResolver._value_matches_domain(value, hinted_domain)
                    for value in (item.value if isinstance(item.value, list) else [item.value])
                )
                for claim_id in valid_claim_ids
                for item in by_claim[claim_id]
            )
            if not has_explicit_domain_claim:
                return None
            matching_sources = [
                source
                for source in identity_sources
                if source.source_type.value == "official"
                and source_domain(str(source.url)) == hinted_domain
            ]
            if matching_sources:
                return min(
                    matching_sources,
                    key=lambda source: canonicalize_url(str(source.url)),
                ).url

        return None

    @staticmethod
    def _value_matches_domain(value: object, expected_domain: str) -> bool:
        if not isinstance(value, str):
            return False
        candidate = value.casefold().strip().strip("./")
        if candidate.startswith(("http://", "https://")):
            try:
                candidate = source_domain(candidate)
            except ValueError:
                return False
        return candidate.removeprefix("www.") == expected_domain

    @classmethod
    def _has_exact_claim_value(
        cls,
        claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
        proposed_value: str,
        *,
        claim_terms: tuple[str, ...] = (),
        claim_kind: str | None = None,
    ) -> bool:
        expected = normalized_fingerprint_text(proposed_value)
        if not expected:
            return False
        for claim_id in claim_ids:
            for item in by_claim[claim_id]:
                if item.status == EvidenceStatus.UNKNOWN:
                    continue
                claim = " ".join(item.claim.casefold().split())
                claim_tokens = set(re.findall(r"[a-z0-9]+", claim))
                if claim_kind and cls._identity_claim_kind(claim) != claim_kind:
                    continue
                if claim_terms and not all(term in claim_tokens for term in claim_terms):
                    continue
                values = item.value if isinstance(item.value, list) else [item.value]
                if any(
                    isinstance(value, str)
                    and (
                        cls._organization_names_match(value, proposed_value)
                        if claim_kind == "name"
                        else normalized_fingerprint_text(value) == expected
                    )
                    for value in values
                ):
                    return True
        return False

    @staticmethod
    def _organization_names_match(first: str, second: str) -> bool:
        first_normalized = normalize_organization_name(first)
        return bool(first_normalized and first_normalized == normalize_organization_name(second))

    @classmethod
    def _equivalent_name_conflict(cls, values: Sequence[object]) -> bool:
        if len(values) < 2 or not all(isinstance(value, str) for value in values):
            return False
        return len({normalize_organization_name(str(value)) for value in values}) == 1

    @classmethod
    def _identity_name_values(
        cls,
        claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value
                for claim_id in claim_ids
                for item in by_claim[claim_id]
                if cls._identity_claim_kind(item.claim) == "name"
                for value in (item.value if isinstance(item.value, list) else [item.value])
                if isinstance(value, str)
            )
        )

    @classmethod
    def _preferred_canonical_name(
        cls,
        snapshot_name: str,
        proposed_name: str,
        claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
    ) -> str:
        supported_names = cls._identity_name_values(claim_ids, by_claim)
        reference_name = proposed_name or snapshot_name
        equivalent = [
            value
            for value in supported_names
            if cls._organization_names_match(value, reference_name)
        ]
        if not equivalent:
            return snapshot_name
        equivalent.extend(
            value
            for value in (snapshot_name, proposed_name)
            if value
            and any(cls._organization_names_match(value, supported) for supported in equivalent)
        )
        return max(
            equivalent,
            key=lambda value: (
                has_organization_legal_suffix(value),
                len(re.findall(r"[a-z0-9]+", value.casefold())),
                len(value),
                value,
            ),
        )

    @classmethod
    def _is_identity_claim(cls, items: list[Evidence]) -> bool:
        return any(
            item.status != EvidenceStatus.UNKNOWN
            and cls._identity_claim_kind(item.claim) is not None
            for item in items
        )

    _IDENTITY_CLAIM_PATTERNS: ClassVar[dict[str, frozenset[tuple[str, ...]]]] = {
        "name": frozenset(
            {
                ("company", "identity"),
                ("company", "name"),
                ("corporate", "identity"),
                ("corporate", "name"),
                ("legal", "company", "name"),
                ("legal", "name"),
                ("official", "company", "name"),
                ("company", "alias"),
                ("company", "aliases"),
                ("former", "company", "name"),
            }
        ),
        "country": frozenset(
            {
                ("company", "country"),
                ("company", "domicile"),
                ("country", "of", "incorporation"),
                ("corporate", "domicile"),
                ("headquarters", "country"),
                ("headquarter", "country"),
            }
        ),
        "industry": frozenset(
            {
                ("company", "industry"),
                ("company", "sector"),
                ("corporate", "industry"),
                ("corporate", "sector"),
                ("business", "focus"),
                ("industry",),
            }
        ),
        "website": frozenset(
            {
                ("company", "website"),
                ("company", "domain"),
                ("official", "website"),
                ("official", "domain"),
                ("website",),
            }
        ),
        "status": frozenset(
            {
                ("company", "operating", "status"),
                ("company", "status"),
                ("corporate", "status"),
                ("operating", "status"),
                ("company", "closed"),
                ("company", "dissolved"),
            }
        ),
    }

    @classmethod
    def _identity_claim_kind(cls, claim: str) -> str | None:
        tokens = tuple(re.findall(r"[a-z0-9]+", claim.casefold()))
        for kind, patterns in cls._IDENTITY_CLAIM_PATTERNS.items():
            if tokens in patterns:
                return kind
        token_set = set(tokens)
        # LLM extractors often phrase direct identity facts as complete
        # sentences (for example, "Pfizer Inc. is a biopharmaceutical company")
        # rather than using the short canonical labels above. Recognize only
        # strong lexical indicators; values and source lineage remain subject
        # to the existing evidence-backed sanitization below.
        if "website" in token_set or "domain" in token_set:
            return "website"
        if "incorporated" in token_set and ({"state", "country", "jurisdiction"} & token_set):
            return "country"
        if "incorporated" in token_set and "name" in token_set:
            return "name"
        if {"industry", "sector", "biopharmaceutical"} & token_set:
            return "industry"
        return None

    @classmethod
    def _relationship_status_claim_ids(
        cls,
        claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
        *,
        relationship_tokens: tuple[str, ...],
        related_company: str,
        status: str,
    ) -> list[str]:
        expected_status = " ".join(status.casefold().split())
        related_tokens = tuple(re.findall(r"[a-z0-9]+", related_company.casefold()))
        matches: list[str] = []
        for claim_id in claim_ids:
            for item in by_claim[claim_id]:
                if item.status == EvidenceStatus.UNKNOWN:
                    continue
                claim = " ".join(item.claim.casefold().split())
                ordered_claim_tokens = tuple(re.findall(r"[a-z0-9]+", claim))
                claim_tokens = set(ordered_claim_tokens)
                allowed_claim_tokens = {
                    *related_tokens,
                    *relationship_tokens,
                    "company",
                    "for",
                    "relationship",
                    "status",
                    "with",
                }
                values = item.value if isinstance(item.value, list) else [item.value]
                if (
                    "status" in claim_tokens
                    and claim_tokens.issubset(allowed_claim_tokens)
                    and all(token in claim_tokens for token in relationship_tokens)
                    and related_tokens
                    and any(
                        ordered_claim_tokens[index : index + len(related_tokens)] == related_tokens
                        for index in range(len(ordered_claim_tokens) - len(related_tokens) + 1)
                    )
                    and any(
                        isinstance(value, str)
                        and " ".join(value.casefold().split()) == expected_status
                        for value in values
                    )
                ):
                    matches.append(claim_id)
                    break
        return list(dict.fromkeys(matches))

    @classmethod
    def _supports_closed_value(
        cls,
        claim_ids: list[str],
        by_claim: dict[str, list[Evidence]],
        proposed: bool,
    ) -> bool:
        for claim_id in claim_ids:
            for item in by_claim[claim_id]:
                claim_text = item.claim.casefold()
                if cls._identity_claim_kind(claim_text) != "status":
                    continue
                if isinstance(item.value, bool) and item.value is proposed:
                    return True
                value = " ".join(str(item.value).casefold().split())
                if proposed and value in {"closed", "inactive", "dissolved"}:
                    return True
                if not proposed and value in {"active", "operating", "open"}:
                    return True
        return False
