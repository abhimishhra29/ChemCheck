from __future__ import annotations

from app.workflow.utils import SDSFlowState


def retry_node(state: SDSFlowState) -> dict:
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
