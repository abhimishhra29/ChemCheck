from __future__ import annotations

import importlib.util
import os
import re
from typing import Literal, Optional, TypedDict
from urllib.parse import urlparse

from app.agents.ocr_agent import OCRResult

DEFAULT_TIMEOUT_S = 60
PDF_SAMPLE_BYTES = 200_000
MAX_SDS_CONTENT_CHARS = 8000
MAX_CONTENT_SNIPPET_CHARS = 1200

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


def format_output(state: SDSFlowState) -> dict:
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


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = f"https://{value}"
    return value


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_same_domain(url: str, domain: str) -> bool:
    netloc = extract_domain(url)
    return netloc == domain or netloc.endswith("." + domain)
