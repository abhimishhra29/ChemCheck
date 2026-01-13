"""
Utilities for handling and validating file uploads, and for text/URL processing.
"""
import re
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException, UploadFile

from app.models import FileInfo

# A set of allowed image MIME types for validation.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def validate_image(file: UploadFile) -> None:
    """
    Validates an uploaded image file.

    Checks for a filename and ensures the content type is one of the allowed
    image formats.

    Args:
        file: The uploaded file from FastAPI.

    Raises:
        HTTPException: If the file has no name or has an unsupported content type.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PNG or JPEG.",
        )


def to_file_info(file: Optional[UploadFile]) -> Optional[FileInfo]:
    """
    Converts a FastAPI UploadFile to a Pydantic FileInfo model.

    If the input file is None, this function returns None.

    Args:
        file: The optional uploaded file.

    Returns:
        A FileInfo object or None.
    """
    if not file:
        return None
    return FileInfo(filename=file.filename, content_type=file.content_type)


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


def normalize_url(url: str) -> str:
    """
    Ensures a URL string includes a scheme (e.g., "https://").

    Args:
        url: The URL string to normalize.

    Returns:
        A URL string with "https://" prepended if no scheme was present.
    """
    value = url.strip()
    if not value:
        return value
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = f"https://{value}"
    return value
