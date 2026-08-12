import re

_WHITESPACE = re.compile(r"\s+")
_QUERY_TOKEN = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "company",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
)
_ORGANIZATION_LEGAL_SUFFIXES = frozenset(
    {
        "ag",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "ltd",
        "plc",
    }
)
_ORGANIZATION_LEGAL_SUFFIX_PATTERN = re.compile(
    r"(?:[\s,.-]+)(?:ag|corp|corporation|inc|incorporated|limited|llc|ltd|plc)\.?$",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def normalized_fingerprint_text(value: str) -> str:
    return normalize_text(value).casefold()


def query_tokens(value: str) -> frozenset[str]:
    """Return stable, low-noise tokens for deterministic query comparison."""

    return frozenset(
        _normalize_query_token(token)
        for token in _QUERY_TOKEN.findall(normalized_fingerprint_text(value))
        if token not in _QUERY_STOP_WORDS
    )


def _normalize_query_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss") and token != "news":
        return token[:-1]
    return token


def queries_are_near_duplicates(first: str, second: str, *, threshold: float = 0.82) -> bool:
    """Conservatively reject trivial paraphrases while preserving distinct intents."""

    normalized_first = normalized_fingerprint_text(first)
    normalized_second = normalized_fingerprint_text(second)
    if normalized_first == normalized_second:
        return True
    first_tokens = query_tokens(first)
    second_tokens = query_tokens(second)
    if not first_tokens or not second_tokens:
        return False
    intersection = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    jaccard = intersection / union
    containment = intersection / min(len(first_tokens), len(second_tokens))
    return jaccard >= threshold or (
        containment >= 0.9 and abs(len(first_tokens) - len(second_tokens)) <= 1
    )


def normalize_organization_name(value: str) -> str:
    """Normalize an organization name for identity equivalence checks."""

    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    while tokens and tokens[-1] in _ORGANIZATION_LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def has_organization_legal_suffix(value: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return bool(tokens and tokens[-1] in _ORGANIZATION_LEGAL_SUFFIXES)


def organization_short_name(value: str) -> str:
    return normalize_text(_ORGANIZATION_LEGAL_SUFFIX_PATTERN.sub("", value)).strip(" ,.-")
