from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Literal, Optional, TypedDict
from urllib.parse import urlparse

import requests
from langgraph.graph import END, START, StateGraph

from app.agents.ocr_agent import OCRAgent, OCRResult

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT_S = 60
PDF_SAMPLE_BYTES = 200_000
MAX_SDS_CONTENT_CHARS = 8000
MAX_CONTENT_SNIPPET_CHARS = 1200
RAG_CHUNK_CHARS = 800
RAG_CHUNK_OVERLAP = 120
RAG_TOP_K = 4

URL_CANDIDATE_RE = re.compile(
    r"(https?://[^\s)]+|www\.[^\s)]+|\b[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)

MCP_SERVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "mcp", "mcp_server.py")
)


def _load_mcp_tools():
    spec = importlib.util.spec_from_file_location("chemcheck_mcp_server", MCP_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MCP server module from {MCP_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_manufacturer_url, module.get_sds_url


get_manufacturer_url, get_sds_url = _load_mcp_tools()

SDS_KEYWORDS = ("sds", "safety data sheet", "safety datasheet")

ValidationStatus = Literal["pass", "fail", "not_found"]
FinalStatus = Literal["found", "not_found"]


class SDSFlowState(TypedDict, total=False):
    image_path: str
    ocr: OCRResult
    manufacturer: Optional[str]
    manufacturer_url: Optional[str]
    manufacturer_domain: Optional[str]
    sds_url: Optional[str]
    sds_content: Optional[str]
    sds_is_pdf: Optional[bool]
    sds_query: Optional[str]
    sds_attempts: list[dict]
    validation_status: ValidationStatus
    validation: dict
    retry_count: int
    flagged_urls: list[str]
    final_status: FinalStatus
    status: str
    confidence: str
    message: str


def build_sds_graph():
    chemcheck = StateGraph(SDSFlowState)
    chemcheck.add_node("ocr", _ocr_node)
    chemcheck.add_node("resolve_manufacturer", _resolve_manufacturer_node)
    chemcheck.add_node("fetch_sds", _fetch_sds_node)
    chemcheck.add_node("validation", _validation_node)
    chemcheck.add_node("retry", _retry_node)
    chemcheck.add_node("finalize", _finalize_node)

    chemcheck.add_edge(START, "ocr")
    chemcheck.add_edge("ocr", "resolve_manufacturer")
    chemcheck.add_conditional_edges("resolve_manufacturer", _route_after_manufacturer)
    chemcheck.add_edge("fetch_sds", "validation")
    chemcheck.add_conditional_edges("validation", _route_after_validation)
    chemcheck.add_edge("retry", "fetch_sds")
    chemcheck.add_edge("finalize", END)
    return chemcheck.compile()


def run_sds_flow(image_path: str) -> dict:
    app = build_sds_graph()
    state = app.invoke({"image_path": image_path})
    return _format_output(state)


def run_sds_flow_from_ocr(ocr: OCRResult, manufacturer_url: Optional[str] = None) -> dict:
    app = build_sds_graph()
    state = app.invoke({"ocr": ocr, "manufacturer_url": manufacturer_url})
    return _format_output(state)


def _format_output(state: SDSFlowState) -> dict:
    validation = state.get("validation")
    if not validation and state.get("validation_status"):
        validation = {"status": state.get("validation_status")}
    return {
        "status": state.get("status") or state.get("final_status"),
        "message": state.get("message"),
        "confidence": state.get("confidence"),
        "manufacturer_url": state.get("manufacturer_url"),
        "sds_url": state.get("sds_url"),
        "sds_query": state.get("sds_query"),
        "sds_attempts": state.get("sds_attempts", []),
        "validation": validation,
    }


def _ocr_node(state: SDSFlowState) -> dict:
    if state.get("ocr"):
        ocr = state["ocr"]
    else:
        image_path = state["image_path"]
        agent = OCRAgent()
        ocr = agent.extract_details(image_path)

    manufacturer_url = state.get("manufacturer_url") or _extract_url_from_text(ocr.full_text)
    return {
        "ocr": ocr,
        "manufacturer": ocr.manufacturer_name,
        "manufacturer_url": manufacturer_url,
    }


def _resolve_manufacturer_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state.get("manufacturer_url")
    base = {"retry_count": 0, "flagged_urls": []}

    if manufacturer_url:
        normalized = _normalize_url(manufacturer_url)
        return {
            **base,
            "manufacturer_url": normalized,
            "manufacturer_domain": _extract_domain(normalized),
        }

    manufacturer_name = state.get("manufacturer") or ocr.manufacturer_name
    if not manufacturer_name:
        return {**base, "manufacturer_url": None, "manufacturer_domain": None}

    result = get_manufacturer_url(manufacturer_name)
    url = result.get("manufacturer_url")
    if not url:
        return {**base, "manufacturer_url": None, "manufacturer_domain": None}

    normalized = _normalize_url(url)
    return {
        **base,
        "manufacturer_url": normalized,
        "manufacturer_domain": result.get("best_domain") or _extract_domain(normalized),
    }


def _fetch_sds_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state.get("manufacturer_url")
    if not manufacturer_url:
        return {"sds_url": None, "sds_content": None}

    attempts = list(state.get("sds_attempts", []))
    flagged = set(state.get("flagged_urls", []))

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
        if sds_url and sds_url in flagged:
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


def _validation_node(state: SDSFlowState) -> dict:
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
        manufacturer_domain = _extract_domain(_normalize_url(manufacturer_url))

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


def _retry_node(state: SDSFlowState) -> dict:
    flagged = list(state.get("flagged_urls", []))
    if state.get("sds_url"):
        flagged.append(state["sds_url"])

    return {
        "flagged_urls": flagged,
        "retry_count": state.get("retry_count", 0) + 1,
        "sds_url": None,
        "sds_content": None,
        "sds_is_pdf": None,
    }


def _finalize_node(state: SDSFlowState) -> dict:
    validation_status = state.get("validation_status") or "not_found"
    sds_url = state.get("sds_url")
    is_found = validation_status == "pass" and bool(sds_url)

    status = "found" if is_found else "not_found"
    message = f"SDS found: {sds_url}" if is_found else "SDS not found"

    validation = state.get("validation") or {}
    if validation_status != "pass":
        reason = validation.get("reason")
        if reason:
            message = f"{message} Validation: {reason}"

    confidence = "high" if validation_status == "pass" else "low"
    return {
        "final_status": status,
        "status": status,
        "message": message,
        "confidence": confidence,
    }


def _route_after_manufacturer(state: SDSFlowState) -> Literal["fetch_sds", "finalize"]:
    return "fetch_sds" if state.get("manufacturer_url") else "finalize"


def _route_after_validation(state: SDSFlowState) -> Literal["finalize", "retry"]:
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
        domain_match = _is_same_domain(sds_url, manufacturer_domain)

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


def _fetch_sds_content(url: str) -> dict:
    sample = _fetch_url_sample(url)
    content = (sample.get("text") or "")[:MAX_SDS_CONTENT_CHARS]
    return {
        "content": content,
        "is_pdf": sample.get("is_pdf"),
        "error": sample.get("error"),
    }


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
        domain_match = _is_same_domain(sds_url, manufacturer_domain)

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


def _extract_url_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = URL_CANDIDATE_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).strip()
    candidate = candidate.rstrip(".,);]")
    return candidate


def _normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = f"https://{value}"
    return value


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _is_same_domain(url: str, domain: str) -> bool:
    netloc = _extract_domain(url)
    return netloc == domain or netloc.endswith("." + domain)


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
