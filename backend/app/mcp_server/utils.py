"""
General utility functions for web searching and text processing.
"""
import re
from urllib.parse import urlparse


def normalized_tokens(text: str, stopwords: set[str]) -> list[str]:
    """
    Splits text into lowercase alphanumeric tokens and removes stopwords.

    Args:
        text: The input string.
        stopwords: A set of words to exclude from the result.

    Returns:
        A list of cleaned, normalized tokens.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in stopwords]


def extract_domain(url: str) -> str:
    """
    Parses a URL to extract the network location (domain), stripping "www.".

    Args:
        url: The URL to parse.

    Returns:
        The extracted domain (e.g., "example.com").
    """
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_au_domain(domain: str, au_tlds: tuple[str, ...]) -> bool:
    """
    Checks if a domain is an Australian top-level domain.

    Args:
        domain: The domain to check.
        au_tlds: A tuple of Australian TLD suffixes.

    Returns:
        True if the domain is an Australian domain, False otherwise.
    """
    value = (domain or "").lower()
    return any(value.endswith(suffix) for suffix in au_tlds)


def is_same_domain(url: str, domain: str) -> bool:
    """
    Checks if a URL belongs to a given domain or its subdomains.

    Args:
        url: The URL to check.
        domain: The base domain to compare against.

    Returns:
        True if the URL's domain is the same as or a subdomain of the given domain.
    """
    netloc = extract_domain(url)
    return netloc == domain or netloc.endswith("." + domain)
