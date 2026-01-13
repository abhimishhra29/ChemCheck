from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Optional, TypedDict
from urllib.parse import urlparse

import requests
from langgraph.graph import END, StateGraph

from app.agents.ocr_agent import OCRAgent, OCRResult

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"
DEFAULT_TIMEOUT_S = 60
PDF_SAMPLE_BYTES = 200_000
MAX_OCR_TEXT_CHARS = 1200
MAX_CONTENT_SNIPPET_CHARS = 1200

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


class SDSFlowState(TypedDict, total=False):
    image_path: str
    ocr: OCRResult
    manufacturer_url: Optional[str]
    manufacturer_domain: Optional[str]
    sds_url: Optional[str]
    sds_query: Optional[str]
    sds_attempts: list[dict]
    validation: dict
    status: str
    confidence: str
    message: str
    llm_check: dict


def build_sds_graph():
    graph = StateGraph(SDSFlowState)
    graph.add_node("ocr", _ocr_node)
    graph.add_node("resolve_manufacturer", _resolve_manufacturer_node)
    graph.add_node("fetch_sds", _fetch_sds_node)
    graph.add_node("validate_sds", _validate_sds_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_node("no_manufacturer", _no_manufacturer_node)

    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "resolve_manufacturer")
    graph.add_conditional_edges(
        "resolve_manufacturer",
        _route_manufacturer,
        {"have_manufacturer": "fetch_sds", "missing_manufacturer": "no_manufacturer"},
    )
    graph.add_edge("fetch_sds", "validate_sds")
    graph.add_edge("validate_sds", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("no_manufacturer", END)
    return graph.compile()


def run_sds_flow(image_path: str) -> dict:
    app = build_sds_graph()
    state = app.invoke({"image_path": image_path})
    return {
        "status": state.get("status"),
        "message": state.get("message"),
        "confidence": state.get("confidence"),
        "manufacturer_url": state.get("manufacturer_url"),
        "sds_url": state.get("sds_url"),
        "sds_query": state.get("sds_query"),
        "sds_attempts": state.get("sds_attempts", []),
        "validation": state.get("validation"),
    }


def run_sds_flow_from_ocr(ocr: OCRResult, manufacturer_url: Optional[str] = None) -> dict:
    app = build_sds_graph()
    state = app.invoke({"ocr": ocr, "manufacturer_url": manufacturer_url})
    return {
        "status": state.get("status"),
        "message": state.get("message"),
        "confidence": state.get("confidence"),
        "manufacturer_url": state.get("manufacturer_url"),
        "sds_url": state.get("sds_url"),
        "sds_query": state.get("sds_query"),
        "sds_attempts": state.get("sds_attempts", []),
        "validation": state.get("validation"),
    }

def _ocr_node(state: SDSFlowState) -> dict:
    if state.get("ocr"):
        ocr = state["ocr"]
        manufacturer_url = state.get("manufacturer_url")
        if not manufacturer_url:
            manufacturer_url = _extract_url_from_text(ocr.full_text)
        return {"ocr": ocr, "manufacturer_url": manufacturer_url}

    image_path = state["image_path"]
    agent = OCRAgent()
    ocr = agent.extract_details(image_path)
    manufacturer_url = _extract_url_from_text(ocr.full_text)
    return {"ocr": ocr, "manufacturer_url": manufacturer_url}


def _resolve_manufacturer_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state.get("manufacturer_url")
    if manufacturer_url:
        normalized = _normalize_url(manufacturer_url)
        return {
            "manufacturer_url": normalized,
            "manufacturer_domain": _extract_domain(normalized),
        }

    if not ocr.manufacturer_name:
        return {"manufacturer_url": None, "manufacturer_domain": None}

    result = get_manufacturer_url(ocr.manufacturer_name)
    url = result.get("manufacturer_url")
    if not url:
        return {"manufacturer_url": None, "manufacturer_domain": None}

    normalized = _normalize_url(url)
    return {"manufacturer_url": normalized, "manufacturer_domain": result.get("best_domain")}


def _fetch_sds_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state["manufacturer_url"]
    attempts: list[dict] = []

    search_order = [
        ("cas_number", ocr.cas_number),
        ("product_code", ocr.product_code),
        ("product_name", ocr.product_name),
    ]

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
        attempts.append(
            {
                "field": field_name,
                "value": value,
                "query": result.get("query"),
                "sds_url": result.get("sds_url"),
            }
        )
        if result.get("sds_url"):
            return {
                "sds_url": result.get("sds_url"),
                "sds_query": result.get("query"),
                "manufacturer_domain": result.get("manufacturer_domain"),
                "sds_attempts": attempts,
            }

    last_query = attempts[-1]["query"] if attempts else None
    return {
        "sds_url": None,
        "sds_query": last_query,
        "manufacturer_domain": state.get("manufacturer_domain"),
        "sds_attempts": attempts,
    }


def _validate_sds_node(state: SDSFlowState) -> dict:
    sds_url = state.get("sds_url")
    if not sds_url:
        return {"validation": {"status": "not_found"}}

    manufacturer_domain = state.get("manufacturer_domain")
    if not manufacturer_domain:
        manufacturer_url = state.get("manufacturer_url") or ""
        manufacturer_domain = _extract_domain(_normalize_url(manufacturer_url))

    validation = _validate_sds_candidate(
        sds_url=sds_url,
        manufacturer_domain=manufacturer_domain,
        ocr=state["ocr"],
    )
    return {"validation": validation}


def _finalize_node(state: SDSFlowState) -> dict:
    sds_url = state.get("sds_url")
    validation = state.get("validation") or {}
    validation_status = validation.get("status")

    if validation_status == "fail":
        sds_url = None
    llm_check = _final_check_with_llm(state)

    confidence = llm_check.get("confidence")
    if confidence not in {"high", "low"}:
        confidence = _fallback_confidence(state)

    if sds_url:
        message = f"SDS found: {sds_url}"
        status = "found"
    else:
        message = "SDS not found."
        status = "not_found"

    if validation_status in {"fail", "low_confidence"}:
        reason = validation.get("reason")
        if reason:
            message = f"{message} Validation: {reason}"

    note = llm_check.get("note")
    if note:
        message = f"{message} {note}".strip()

    if validation_status in {"fail", "low_confidence"}:
        confidence = "low"

    if confidence == "low" and "low confidence" not in message.lower():
        message = f"{message} Low confidence."

    return {
        "status": status,
        "confidence": confidence,
        "message": message,
        "llm_check": llm_check,
    }


def _no_manufacturer_node(state: SDSFlowState) -> dict:
    return {
        "status": "could_not_find",
        "confidence": "low",
        "message": "Could not find manufacturer URL.",
    }


def _route_manufacturer(state: SDSFlowState) -> str:
    return "have_manufacturer" if state.get("manufacturer_url") else "missing_manufacturer"


def _final_check_with_llm(state: SDSFlowState) -> dict:
    api_key = os.getenv(MISTRAL_API_KEY_ENV)
    if not api_key:
        return {"confidence": _fallback_confidence(state)}

    ocr = state["ocr"]
    attempts = state.get("sds_attempts", [])
    validation = state.get("validation") or {}

    ocr_text = (ocr.full_text or "").strip()
    ocr_text_snippet = ocr_text[:MAX_OCR_TEXT_CHARS]

    prompt = (
        "You review SDS lookup results for correctness and completeness.\n"
        "Return ONLY JSON with keys: confidence ('high' or 'low'), "
        "note (short, optional).\n\n"
        "Rules:\n"
        "- If manufacturer_url is missing or sds_url is missing, confidence=low.\n"
        "- Use OCR details and validation signals to judge if the SDS matches the product.\n"
        "- If validation status is fail or low_confidence, confidence should be low.\n"
        "- If CAS or product_code was used and the SDS is on the manufacturer domain "
        "and validation is pass, confidence can be high.\n"
        "- If only product_name was used, lean low unless evidence is strong.\n"
    )

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "manufacturer_url": state.get("manufacturer_url"),
                        "manufacturer_domain": state.get("manufacturer_domain"),
                        "sds_url": state.get("sds_url"),
                        "sds_query": state.get("sds_query"),
                        "attempts": attempts,
                        "validation": validation,
                        "ocr": {
                            "product_name": ocr.product_name,
                            "product_code": ocr.product_code,
                            "cas_number": ocr.cas_number,
                            "manufacturer_name": ocr.manufacturer_name,
                            "description": ocr.description,
                            "full_text_snippet": ocr_text_snippet,
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        "temperature": 0,
    }

    resp = requests.post(
        MISTRAL_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=DEFAULT_TIMEOUT_S,
    )
    if not resp.ok:
        return {"confidence": _fallback_confidence(state)}

    content = resp.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        return {"confidence": _fallback_confidence(state)}

    try:
        parsed = _parse_llm_json_strict(content)
    except ValueError:
        return {"confidence": _fallback_confidence(state)}

    return parsed


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


def _validate_sds_candidate(
    sds_url: str, manufacturer_domain: Optional[str], ocr: OCRResult
) -> dict:
    url_text = sds_url.lower()
    domain_match = False
    if manufacturer_domain:
        domain_match = _is_same_domain(sds_url, manufacturer_domain)

    sample = _fetch_url_sample(sds_url)
    content_text = sample.get("text") or ""

    matches = []
    reasons = []
    score = 0.0

    if domain_match:
        matches.append("manufacturer_domain")
        score += 2.0
    else:
        reasons.append("SDS URL not on manufacturer domain")

    if sample.get("is_pdf") is True:
        matches.append("pdf_signature")
        score += 2.0
    elif sample.get("is_pdf") is False:
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

    if not sds_url:
        status = "not_found"
        reason = "No SDS URL to validate."
    elif sample.get("is_pdf") is False:
        status = "fail"
        reason = "SDS URL is not a PDF."
    elif manufacturer_domain and not domain_match:
        status = "fail"
        reason = "SDS URL is not on the manufacturer domain."
    elif score >= 5.0:
        status = "pass"
        reason = "Validation passed based on OCR matches."
    else:
        status = "low_confidence"
        reason = "Limited match between OCR data and SDS file."

    return {
        "status": status,
        "score": round(score, 2),
        "matches": matches,
        "reasons": reasons,
        "reason": reason,
        "is_pdf": sample.get("is_pdf"),
        "domain_match": domain_match,
        "content_snippet": sample.get("snippet"),
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
    text = data.decode("latin-1", errors="ignore").lower()
    snippet = text[:MAX_CONTENT_SNIPPET_CHARS]
    return {"is_pdf": is_pdf, "text": text, "snippet": snippet}


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


def _fallback_confidence(state: SDSFlowState) -> str:
    if not state.get("manufacturer_url") or not state.get("sds_url"):
        return "low"
    ocr = state["ocr"]
    if ocr.cas_number or ocr.product_code:
        return "high"
    return "low"
