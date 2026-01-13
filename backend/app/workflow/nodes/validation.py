from __future__ import annotations

import json
import os
import re
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

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5-mini"
RAG_CHUNK_CHARS = 800
RAG_CHUNK_OVERLAP = 120
RAG_TOP_K = 4
SDS_KEYWORDS = ("sds", "safety data sheet", "safety datasheet")


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

    manufacturer_domain = state.get("manufacturer_domain")
    if not manufacturer_domain:
        manufacturer_url = state.get("manufacturer_url") or ""
        manufacturer_domain = extract_domain(normalize_url(manufacturer_url))

    validation = _validate_sds_with_rag(
        sds_url=sds_url,
        manufacturer_domain=manufacturer_domain,
        ocr=state["ocr"],
        sds_content=sds_content,
        sds_is_pdf=state.get("sds_is_pdf"),
        manufacturer_url=state.get("manufacturer_url"),
    )
    status = validation.get("status")
    if status not in {"pass", "fail", "not_found"}:
        status = "fail"
    return {"validation_status": status, "validation": validation}


def route_after_validation(state: SDSFlowState) -> Literal["finalize", "retry"]:
    if state.get("validation_status") == "pass":
        return "finalize"
    if state.get("retry_count", 0) < 2:
        return "retry"
    return "finalize"


def _validate_sds_with_rag(
    *,
    sds_url: str,
    manufacturer_domain: Optional[str],
    ocr: OCRResult,
    sds_content: str,
    sds_is_pdf: Optional[bool],
    manufacturer_url: Optional[str],
) -> dict:
    rag_chunks = _select_rag_chunks(sds_content, ocr)
    domain_match = False
    if manufacturer_domain:
        domain_match = is_same_domain(sds_url, manufacturer_domain)

    llm_result = _call_openai_validation(
        sds_url=sds_url,
        manufacturer_url=manufacturer_url,
        manufacturer_domain=manufacturer_domain,
        domain_match=domain_match,
        ocr=ocr,
        rag_chunks=rag_chunks,
    )

    if llm_result:
        status = _normalize_validation_status(
            llm_result.get("status") or llm_result.get("validation_status")
        )
        if status:
            return {
                "status": status,
                "reason": (llm_result.get("reason") or "").strip(),
                "model": OPENAI_MODEL,
                "domain_match": domain_match,
                "rag_chunks_used": len(rag_chunks),
            }

    validation = _validate_sds_candidate_from_content(
        sds_url=sds_url,
        manufacturer_domain=manufacturer_domain,
        ocr=ocr,
        sds_content=sds_content,
        sds_is_pdf=sds_is_pdf,
    )
    validation["rag_chunks_used"] = len(rag_chunks)
    return validation


def _call_openai_validation(
    *,
    sds_url: str,
    manufacturer_url: Optional[str],
    manufacturer_domain: Optional[str],
    domain_match: bool,
    ocr: OCRResult,
    rag_chunks: list[dict],
) -> Optional[dict]:
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        return None

    system_prompt = (
        "You validate whether an SDS matches OCR product details. "
        "Use only the provided RAG context and metadata. "
        "Return ONLY JSON with keys: status ('pass' or 'fail'), reason (short). "
        "If evidence is insufficient, return status='fail'."
    )

    payload = {
        "sds_url": sds_url,
        "manufacturer_url": manufacturer_url,
        "manufacturer_domain": manufacturer_domain,
        "domain_match": domain_match,
        "ocr": {
            "product_name": ocr.product_name,
            "product_code": ocr.product_code,
            "cas_number": ocr.cas_number,
            "manufacturer_name": ocr.manufacturer_name,
            "description": ocr.description,
        },
        "rag_context": [
            {"id": chunk.get("id"), "text": chunk.get("text")}
            for chunk in rag_chunks
        ],
    }

    request = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        "temperature": 0,
    }

    resp = requests.post(
        OPENAI_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request,
        timeout=DEFAULT_TIMEOUT_S,
    )
    if not resp.ok:
        return None

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return None

    try:
        return _parse_llm_json_strict(content)
    except ValueError:
        return None


def _normalize_validation_status(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"pass", "fail", "not_found"}:
        return normalized
    return None


def _select_rag_chunks(text: str, ocr: OCRResult) -> list[dict]:
    chunks = _chunk_text(text, RAG_CHUNK_CHARS, RAG_CHUNK_OVERLAP)
    if not chunks:
        return []

    query_tokens = _build_query_tokens(ocr)
    scored = []
    for idx, chunk in enumerate(chunks):
        score = _score_rag_chunk(chunk, ocr, query_tokens)
        scored.append((score, idx, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [item for item in scored if item[0] > 0][:RAG_TOP_K]
    if not picked:
        picked = scored[: min(len(scored), RAG_TOP_K)]

    return [
        {"id": idx, "score": score, "text": chunk}
        for score, idx, chunk in picked
    ]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks = []
    length = len(text)
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - overlap)
        if start >= length:
            break
    return chunks


def _build_query_tokens(ocr: OCRResult) -> set[str]:
    tokens: set[str] = set()
    values = [
        ocr.product_name,
        ocr.product_code,
        ocr.cas_number,
        ocr.manufacturer_name,
        ocr.description,
    ]
    for value in values:
        if not value:
            continue
        tokens.update(re.findall(r"[a-z0-9]+", value.lower()))
    return {t for t in tokens if len(t) >= 3}


def _score_rag_chunk(chunk: str, ocr: OCRResult, query_tokens: set[str]) -> float:
    chunk_lower = chunk.lower()
    score = 0.0

    for token in query_tokens:
        if token in chunk_lower:
            score += 1.0

    if ocr.cas_number and ocr.cas_number.lower() in chunk_lower:
        score += 4.0

    if ocr.product_code:
        code_norm = _normalize_alnum(ocr.product_code)
        if code_norm and code_norm in _normalize_alnum(chunk_lower):
            score += 3.0

    return score


def _parse_llm_json_strict(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def _validate_sds_candidate_from_content(
    *,
    sds_url: str,
    manufacturer_domain: Optional[str],
    ocr: OCRResult,
    sds_content: str,
    sds_is_pdf: Optional[bool],
) -> dict:
    url_text = (sds_url or "").lower()
    content_text = (sds_content or "").lower()

    domain_match = False
    if manufacturer_domain:
        domain_match = is_same_domain(sds_url, manufacturer_domain)

    matches = []
    reasons = []
    score = 0.0

    if domain_match:
        matches.append("manufacturer_domain")
        score += 2.0
    else:
        reasons.append("SDS URL not on manufacturer domain")

    if sds_is_pdf is True:
        matches.append("pdf_signature")
        score += 2.0
    elif sds_is_pdf is False:
        reasons.append("SDS URL does not look like a PDF")

    if _contains_keyword(url_text, content_text, SDS_KEYWORDS):
        matches.append("sds_keyword")
        score += 1.5

    if _field_match_cas(ocr.cas_number, url_text, content_text):
        matches.append("cas_number")
        score += 2.5
    if _field_match_code(ocr.product_code, url_text, content_text):
        matches.append("product_code")
        score += 2.0
    if _field_match_tokens(ocr.product_name, url_text, content_text):
        matches.append("product_name")
        score += 1.0
    if _field_match_tokens(ocr.manufacturer_name, url_text, content_text):
        matches.append("manufacturer_name")
        score += 1.0
    if _field_match_tokens(ocr.description, url_text, content_text):
        matches.append("description")
        score += 0.5

    if not sds_url or not sds_content:
        status = "not_found"
        reason = "No SDS content to validate."
    elif sds_is_pdf is False:
        status = "fail"
        reason = "SDS URL is not a PDF."
    elif manufacturer_domain and not domain_match:
        status = "fail"
        reason = "SDS URL is not on the manufacturer domain."
    elif score >= 5.0:
        status = "pass"
        reason = "Validation passed based on OCR matches."
    else:
        status = "fail"
        reason = "Limited match between OCR data and SDS file."

    return {
        "status": status,
        "score": round(score, 2),
        "matches": matches,
        "reasons": reasons,
        "reason": reason,
        "is_pdf": sds_is_pdf,
        "domain_match": domain_match,
        "content_snippet": (sds_content or "")[:MAX_CONTENT_SNIPPET_CHARS],
        "model": "heuristic",
    }


def _contains_keyword(url_text: str, content_text: str, keywords: tuple[str, ...]) -> bool:
    for keyword in keywords:
        if keyword in url_text or keyword in content_text:
            return True
    return False


def _normalize_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _field_match_cas(value: Optional[str], url_text: str, content_text: str) -> bool:
    if not value:
        return False
    cas_value = value.lower()
    if cas_value in url_text or cas_value in content_text:
        return True
    cas_digits = re.sub(r"[^0-9]", "", cas_value)
    if cas_digits:
        hay = re.sub(r"[^0-9]", "", url_text + " " + content_text)
        return cas_digits in hay
    return False


def _field_match_code(value: Optional[str], url_text: str, content_text: str) -> bool:
    if not value:
        return False
    normalized = _normalize_alnum(value)
    if not normalized:
        return False
    hay = _normalize_alnum(url_text + " " + content_text)
    return normalized in hay


def _field_match_tokens(value: Optional[str], url_text: str, content_text: str) -> bool:
    if not value:
        return False
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return False
    for token in tokens:
        if token in url_text or token in content_text:
            return True
    return False
