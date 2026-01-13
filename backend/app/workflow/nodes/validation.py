from __future__ import annotations

from typing import Literal

from app.services.validation_service import ValidationService
from app.utils.file_utils import is_same_domain
from app.workflow.utils import SDSFlowState


async def validation_node(
    state: SDSFlowState, validation_service: ValidationService
) -> dict:
    """
    Validates the found SDS against the OCR'd product information.

    This node checks for the presence of an SDS URL and content. If available,
    it runs a series of checks, including a domain match and a sophisticated
    LLM-based validation using the ValidationService.

    Args:
        state: The current state of the SDS workflow.
        validation_service: The service responsible for running validation logic.

    Returns:
        A dictionary with the validation status and detailed results.
    """
    sds_url = state.get("sds_url")
    sds_content = state.get("sds_content")

    if not sds_url or not sds_content:
        return {
            "validation_status": "not_found",
            "validation": {
                "status": "not_found",
                "reason": "Missing SDS URL or content for validation.",
            },
        }

    # First, perform a simple domain check.
    warnings = []
    domain_match = None
    manufacturer_domain = state.get("manufacturer_domain")
    if manufacturer_domain:
        domain_match = is_same_domain(sds_url, manufacturer_domain)
        if domain_match is False:
            warnings.append("SDS URL is not on the manufacturer domain.")

    # If the content is not a PDF, fail fast.
    if state.get("sds_is_pdf") is False:
        validation = {
            "status": "fail",
            "confidence": "high",
            "reason": "Found document is not a PDF.",
            "method": "gate",
        }
    else:
        # If it is a PDF, proceed with expensive LLM validation.
        validation = await validation_service.validate_sds(sds_url, state["ocr"])

    # Attach metadata to the final validation result.
    validation["is_pdf"] = state.get("sds_is_pdf")
    validation["domain_match"] = domain_match
    if warnings:
        validation["warnings"] = warnings

    # Ensure the status is one of the prescribed literals.
    status = validation.get("status")
    if status not in {"pass", "fail", "not_found"}:
        status = "fail"
        validation["status"] = status

    return {"validation_status": status, "validation": validation}


def route_after_validation(state: SDSFlowState) -> Literal["finalize", "retry"]:
    """
    Determines the next step after validation.

    If validation passed, finalize. If it failed but retries are available,
    go to the retry node. Otherwise, finalize with a failure status.
    """
    if state.get("validation_status") == "pass":
        return "finalize"
    if state.get("retry_count", 0) < 2:
        return "retry"
    return "finalize"
