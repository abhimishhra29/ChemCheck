from __future__ import annotations

from app.workflow.utils import SDSFlowState


def retry_node(state: SDSFlowState) -> dict:
    """
    Prepares the state for a retry attempt.

    This node is called when SDS validation fails. It increments the retry
    counter and adds the failed SDS URL to a list of "flagged" URLs so that
    it won't be tried again in the next search attempt. It then clears the
    previous SDS-related fields from the state to ensure a clean retry.

    Args:
        state: The current state of the SDS workflow.

    Returns:
        A dictionary with the updated retry state.
    """
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
