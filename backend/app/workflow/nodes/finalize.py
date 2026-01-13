from __future__ import annotations

from app.workflow.utils import SDSFlowState


def finalize_node(state: SDSFlowState) -> dict:
    """
    Finalizes the workflow by summarizing the results.

    This is the terminal node for the workflow. It determines the final status
    ("found" or "not_found") based on the validation results and constructs a
    conclusive message for the user.

    Args:
        state: The final state of the SDS workflow.

    Returns:
        A dictionary containing the final status, message, and confidence score.
    """
    validation_status = state.get("validation_status") or "not_found"
    sds_url = state.get("sds_url")
    is_found = validation_status == "pass" and bool(sds_url)

    status = "found" if is_found else "not_found"
    message = f"SDS found: {sds_url}" if is_found else "SDS not found"

    validation = state.get("validation") or {}
    if validation_status != "pass":
        reason = validation.get("reason")
        if reason:
            message = f"{message}. Reason: {reason}"

    confidence = "high" if validation_status == "pass" else "low"
    return {
        "final_status": status,
        "status": status,
        "message": message,
        "confidence": confidence,
    }
