from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract

_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
_DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_PARAMETERS
        )
    )
    return urlunsplit((scheme, host, path, query, ""))


def source_domain(value: str) -> str:
    host = (urlsplit(value).hostname or "").casefold()
    extracted = _DOMAIN_EXTRACTOR(host)
    return extracted.top_domain_under_public_suffix or host.removeprefix("www.")


def is_public_suffix_only(value: str) -> bool:
    """Return whether a domain rule names a public suffix instead of a publisher."""

    host = value.casefold().strip().strip(".").removeprefix("www.")
    extracted = _DOMAIN_EXTRACTOR(host)
    return bool(extracted.suffix) and not extracted.domain
