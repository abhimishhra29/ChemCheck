from __future__ import annotations

import re
from typing import Optional

from app.agents.ocr_agent import OCRAgent
from app.workflow.utils import SDSFlowState

URL_CANDIDATE_RE = re.compile(
    r"(https?://[^\s)]+|www\.[^\s)]+|\b[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)


def ocr_node(state: SDSFlowState) -> dict:
    if state.get("ocr"):
        ocr = state["ocr"]
    else:
        image_path = state["image_path"]
        agent = OCRAgent()
        ocr = agent.extract_details(image_path)

    manufacturer_url = state.get("manufacturer_url") or _extract_url_from_text(
        ocr.full_text
    )
    return {
        "ocr": ocr,
        "manufacturer": ocr.manufacturer_name,
        "manufacturer_url": manufacturer_url,
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
