from __future__ import annotations

import re

import httpx

from app.services.search_service import SearchService
from app.workflow.utils import (
    MAX_CONTENT_SNIPPET_CHARS,
    MAX_SDS_CONTENT_CHARS,
    PDF_SAMPLE_BYTES,
    SDSFlowState,
)


async def fetch_sds_node(
    state: SDSFlowState,
    search_service: SearchService,
    http_client: httpx.AsyncClient,
) -> dict:
    """
    Tries to find and fetch a Safety Data Sheet (SDS) for a product.

    This node iterates through different search strategies (by CAS number,
    product code, then name) using the SearchService. Once a candidate URL is
    found, it attempts to download the content.

    Args:
        state: The current state of the SDS workflow.
        search_service: The service for finding the SDS URL.
        http_client: The HTTP client for downloading the SDS content.

    Returns:
        A dictionary with the SDS URL, content, and other metadata.
    """
    manufacturer_url = state.get("manufacturer_url")
    if not manufacturer_url:
        return {"sds_url": None, "sds_content": None}

    ocr = state["ocr"]
    attempts = list(state.get("sds_attempts", []))
    flagged = set(state.get("flagged_urls", []))
    flagged_normalized = {url.rstrip("/") for url in flagged if url}

    search_order = [
        ("cas_number", ocr.cas_number),
        ("product_code", ocr.product_code),
        ("product_name", ocr.product_name),
    ]

    last_query = None
    for field_name, value in search_order:
        if not value:
            continue

        params = {
            "manufacturer_url": manufacturer_url,
            "cas_number": value if field_name == "cas_number" else None,
            "product_code": value if field_name == "product_code" else None,
            "product_name": value if field_name == "product_name" else None,
            "exclude_urls": sorted(list(flagged_normalized)),
        }
        result = await search_service.get_sds_url(**params)
        sds_url = result.get("sds_url")
        last_query = result.get("query")
        attempt = {
            "field": field_name,
            "value": value,
            "query": last_query,
            "sds_url": sds_url,
        }

        if sds_url and sds_url.rstrip("/") in flagged_normalized:
            attempt["skipped"] = "flagged"
            attempts.append(attempt)
            continue
        attempts.append(attempt)

        if sds_url:
            content = await _fetch_sds_content(sds_url, http_client)
            return {
                "sds_url": sds_url,
                "sds_content": content.get("content"),
                "sds_is_pdf": content.get("is_pdf"),
                "sds_query": last_query,
                "manufacturer_domain": result.get("manufacturer_domain")
                or state.get("manufacturer_domain"),
                "sds_attempts": attempts,
            }

    return {
        "sds_url": None,
        "sds_content": None,
        "sds_query": last_query,
        "manufacturer_domain": state.get("manufacturer_domain"),
        "sds_attempts": attempts,
    }


async def _fetch_sds_content(url: str, http_client: httpx.AsyncClient) -> dict:
    """Fetches the SDS content and returns the text and PDF status."""
    sample = await _fetch_url_sample(url, http_client)
    content = (sample.get("text") or "")[:MAX_SDS_CONTENT_CHARS]
    return {
        "content": content,
        "is_pdf": sample.get("is_pdf"),
        "error": sample.get("error"),
    }


async def _fetch_url_sample(url: str, http_client: httpx.AsyncClient) -> dict:
    """
    Fetches a sample of a URL's content asynchronously.

    It downloads up to `PDF_SAMPLE_BYTES` to determine the content type
    and extract initial text.
    """
    try:
        async with http_client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            data = b""
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                data += chunk
                if len(data) >= PDF_SAMPLE_BYTES:
                    break
    except httpx.RequestError as exc:
        return {"is_pdf": None, "text": "", "snippet": "", "error": str(exc)}

    is_pdf = "pdf" in content_type or data.startswith(b"%PDF")
    text = _decode_sample_text(data)
    snippet = text[:MAX_CONTENT_SNIPPET_CHARS]
    return {"is_pdf": is_pdf, "text": text, "snippet": snippet, "error": None}


def _decode_sample_text(data: bytes) -> str:
    """Decodes a byte sample into a string, ignoring errors."""
    text = data.decode("latin-1", errors="ignore")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
