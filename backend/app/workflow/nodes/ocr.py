from __future__ import annotations

import re
from typing import Optional

from app.agents.ocr_agent import OCRAgent
from app.workflow.utils import SDSFlowState

URL_CANDIDATE_RE = re.compile(
    r"(https?://[^\s)]+|www\.[^\s)]+|\b[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)


async def ocr_node(state: SDSFlowState, ocr_agent: OCRAgent) -> dict:
    """
    Performs OCR on an image to extract product details.

    If OCR results are already present in the state, it uses them. Otherwise,
    it invokes the OCRAgent to extract details from the image specified in the
    state. It also attempts to find a manufacturer URL from the extracted text
    if one isn't already present.

    Args:
        state: The current state of the SDS workflow.
        ocr_agent: An instance of the OCRAgent for performing OCR.

    Returns:
        A dictionary with the OCR results and extracted manufacturer details
        to be merged back into the workflow state.
    """
    if state.get("ocr"):
        ocr = state["ocr"]
    else:
        image_path = state["image_path"]
        ocr = await ocr_agent.extract_details(image_path)

    manufacturer_url = state.get("manufacturer_url") or _extract_url_from_text(
        ocr.full_text
    )
    return {
        "ocr": ocr,
        "manufacturer": ocr.manufacturer_name,
        "manufacturer_url": manufacturer_url,
    }


def _extract_url_from_text(text: str) -> Optional[str]:
    """A simple regex-based function to find the first likely URL in a block of text."""
    if not text:
        return None
    match = URL_CANDIDATE_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).strip()
    # Clean trailing punctuation that might be included by the regex
    candidate = candidate.rstrip(".,);]")
    return candidate
