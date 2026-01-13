from __future__ import annotations

from typing import Literal, TypedDict

from app.agents.ocr_agent import OCRResult


# --- Constants for Workflow ---

DEFAULT_TIMEOUT_S = 60
PDF_SAMPLE_BYTES = 200_000
MAX_SDS_CONTENT_CHARS = 8000
MAX_CONTENT_SNIPPET_CHARS = 1200

# --- Type Definitions for Workflow State ---

ValidationStatus = Literal["pass", "fail", "not_found"]
FinalStatus = Literal["found", "not_found"]


class SDSFlowState(TypedDict, total=False):
    """
    A TypedDict representing the state of the SDS (Safety Data Sheet) workflow.
    This state is passed between nodes in the langgraph.
    """
    # Input
    image_path: str
    ocr: OCRResult
    
    # Manufacturer resolution
    manufacturer: str | None
    manufacturer_url: str | None
    manufacturer_domain: str | None
    
    # SDS search
    sds_url: str | None
    sds_content: str | None
    sds_is_pdf: bool | None
    sds_query: str | None
    sds_attempts: list[dict]
    
    # Validation & Control Flow
    validation_status: ValidationStatus
    validation: dict
    retry_count: int
    flagged_urls: list[str]
    
    # Final Output
    final_status: FinalStatus
    status: str
    confidence: str
    message: str


def format_output(state: SDSFlowState) -> dict:
    """
    Formats the final state of the workflow into a structured API response.
    """
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
