import re

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def normalized_fingerprint_text(value: str) -> str:
    return normalize_text(value).casefold()
