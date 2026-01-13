from __future__ import annotations

import re

import requests

from app.workflow.utils import (
    DEFAULT_TIMEOUT_S,
    MAX_CONTENT_SNIPPET_CHARS,
    MAX_SDS_CONTENT_CHARS,
    PDF_SAMPLE_BYTES,
    SDSFlowState,
    get_sds_url,
)


def fetch_sds_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state.get("manufacturer_url")
    if not manufacturer_url:
        return {"sds_url": None, "sds_content": None}

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
            "exclude_urls": sorted(flagged_normalized),
        }
        result = get_sds_url(**params)
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
            content = _fetch_sds_content(sds_url)
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


def _fetch_sds_content(url: str) -> dict:
    sample = _fetch_url_sample(url)
    content = (sample.get("text") or "")[:MAX_SDS_CONTENT_CHARS]
    return {
        "content": content,
        "is_pdf": sample.get("is_pdf"),
        "error": sample.get("error"),
    }


def _fetch_url_sample(url: str) -> dict:
    try:
        resp = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"is_pdf": None, "text": "", "snippet": "", "error": str(exc)}

    content_type = (resp.headers.get("Content-Type") or "").lower()
    data = b""
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            data += chunk
            if len(data) >= PDF_SAMPLE_BYTES:
                break
    finally:
        resp.close()

    is_pdf = "pdf" in content_type or data.startswith(b"%PDF")
    text = _decode_sample_text(data)
    snippet = text[:MAX_CONTENT_SNIPPET_CHARS]
    return {"is_pdf": is_pdf, "text": text, "snippet": snippet}


def _decode_sample_text(data: bytes) -> str:
    text = data.decode("latin-1", errors="ignore")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
