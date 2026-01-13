from __future__ import annotations

import json
import os
from typing import Literal, Optional

import requests

from app.agents.ocr_agent import OCRResult
from app.workflow.utils import (
    DEFAULT_TIMEOUT_S,
    MAX_CONTENT_SNIPPET_CHARS,
    SDSFlowState,
    extract_domain,
    is_same_domain,
    normalize_url,
)

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_PDF_BYTES = 8_000_000
MAX_TEXT_CHARS = 10_000


def validation_node(state: SDSFlowState) -> dict:
    sds_url = state.get("sds_url")
    sds_content = state.get("sds_content")

    if not sds_url or not sds_content:
        return {
            "validation_status": "not_found",
            "validation": {
                "status": "not_found",
                "reason": "Missing SDS URL or content.",
            },
        }

    ocr = state["ocr"]
    manufacturer_domain = state.get("manufacturer_domain")
    if not manufacturer_domain:
        manufacturer_url = state.get("manufacturer_url") or ""
        manufacturer_domain = (
            extract_domain(normalize_url(manufacturer_url)) if manufacturer_url else None
        )

    validation = _validate_sds(
        sds_url=sds_url,
        sds_content=sds_content,
        ocr=ocr,
        manufacturer_domain=manufacturer_domain,
        sds_is_pdf=state.get("sds_is_pdf"),
    )
    status = validation.get("status")
    if status not in {"pass", "fail", "not_found"}:
        status = "fail"
        validation["status"] = status

    return {"validation_status": status, "validation": validation}


def route_after_validation(state: SDSFlowState) -> Literal["finalize", "retry"]:
    if state.get("validation_status") == "pass":
        return "finalize"
    if state.get("retry_count", 0) < 2:
        return "retry"
    return "finalize"


def _validate_sds(
    *,
    sds_url: str,
    sds_content: str,
    ocr: OCRResult,
    manufacturer_domain: Optional[str],
    sds_is_pdf: Optional[bool],
) -> dict:
    warnings = []
    domain_match = None

    if manufacturer_domain:
        domain_match = is_same_domain(sds_url, manufacturer_domain)
        if domain_match is False:
            warnings.append("SDS URL is not on the manufacturer domain.")

    if sds_is_pdf is False:
        result = {
            "status": "fail",
            "confidence": "high",
            "reason": "SDS URL is not a PDF.",
            "method": "gate",
        }
        return _attach_meta(result, sds_content, sds_is_pdf, domain_match, warnings)

    if not any([ocr.cas_number, ocr.product_code, ocr.product_name, ocr.description]):
        result = {
            "status": "fail",
            "confidence": "low",
            "reason": "No identifiers available for validation.",
            "method": "none",
        }
        return _attach_meta(result, sds_content, sds_is_pdf, domain_match, warnings)

    mistral_result = _validate_with_mistral(sds_url, ocr)
    return _attach_meta(mistral_result, sds_content, sds_is_pdf, domain_match, warnings)


def _attach_meta(
    result: dict,
    sds_content: str,
    sds_is_pdf: Optional[bool],
    domain_match: Optional[bool],
    warnings: list[str],
) -> dict:
    result["is_pdf"] = sds_is_pdf
    result["domain_match"] = domain_match
    if warnings:
        result["warnings"] = warnings
    return result


def _validate_with_mistral(sds_url: str, ocr: OCRResult) -> dict:
    api_key = os.getenv(MISTRAL_API_KEY_ENV)
    if not api_key:
        return {
            "status": "fail",
            "confidence": "low",
            "reason": "Mistral API is not configured.",
            "method": "mistral",
        }

    try:
        pdf_bytes = _download_pdf(sds_url)
        pdf_text = _extract_first_page_text(pdf_bytes)
    except Exception as exc:
        return {
            "status": "fail",
            "confidence": "low",
            "reason": f"PDF processing failed: {exc}",
            "method": "mistral",
        }

    if not pdf_text.strip():
        return {
            "status": "fail",
            "confidence": "medium",
            "reason": "No text extracted from PDF.",
            "method": "mistral",
        }

    try:
        validation_result = _ask_mistral_to_validate(api_key, pdf_text, ocr)
        return validation_result
    except Exception as exc:
        return {
            "status": "fail",
            "confidence": "low",
            "reason": f"Mistral API error: {exc}",
            "method": "mistral",
        }


def _download_pdf(url: str) -> bytes:
    resp = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT_S)
    resp.raise_for_status()

    data = bytearray()
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            data.extend(chunk)
            if len(data) > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds size limit")
    return bytes(data)


def _extract_first_page_text(pdf_bytes: bytes) -> str:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pymupdf required") from exc

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        doc.close()
        return ""

    text = doc.load_page(0).get_text()
    doc.close()

    return text[:MAX_TEXT_CHARS]


def _ask_mistral_to_validate(api_key: str, pdf_text: str, ocr: OCRResult) -> dict:
    looking_for = []
    if ocr.cas_number:
        looking_for.append(f"CAS Number: {ocr.cas_number}")
    if ocr.product_code:
        looking_for.append(f"Product Code: {ocr.product_code}")
    if ocr.product_name:
        looking_for.append(f"Product Name: {ocr.product_name}")
    if ocr.description:
        looking_for.append(f"Description: {ocr.description}")

    looking_for_str = "\n".join(looking_for)

    prompt = f"""You are validating a Safety Data Sheet (SDS) document.

Expected product information:
{looking_for_str}

First page of SDS:
{pdf_text}

Question: Does this SDS match the expected product?

Check if ANY of the expected identifiers (CAS number, product code, product name, or description) match what's in the SDS text. Be flexible with formatting and minor variations.

Respond ONLY with JSON:
{{
    \"is_match\": true/false,
    \"matched_fields\": [\"list of what matched, e.g. 'CAS number', 'product code'\"],
    \"explanation\": \"brief explanation of what you found\"
}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        MISTRAL_API_URL,
        headers=headers,
        json=payload,
        timeout=DEFAULT_TIMEOUT_S,
    )
    resp.raise_for_status()

    result = resp.json()
    content = json.loads(result["choices"][0]["message"]["content"])

    if content.get("is_match"):
        return {
            "status": "pass",
            "confidence": "high",
            "reason": f"Matched: {', '.join(content.get('matched_fields', []))}",
            "method": "mistral",
            "explanation": content.get("explanation"),
        }

    return {
        "status": "fail",
        "confidence": "high",
        "reason": "No matching identifiers found",
        "method": "mistral",
        "explanation": content.get("explanation"),
    }
