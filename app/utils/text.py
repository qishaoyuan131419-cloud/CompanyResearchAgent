import re

_WHITESPACE = re.compile(r"\s+")
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
