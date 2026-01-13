from __future__ import annotations

from typing import Literal

from app.services.search_service import SearchService
from app.utils.file_utils import extract_domain, normalize_url
from app.workflow.utils import SDSFlowState


async def resolve_manufacturer_node(
    state: SDSFlowState, search_service: SearchService
) -> dict:
    """
    Resolves the manufacturer's website URL.

    If a URL is already in the state, it normalizes it. Otherwise, it uses the
    manufacturer name from the OCR results to search for the official website
    using the SearchService.

    Args:
        state: The current state of the SDS workflow.
        search_service: An instance of the SearchService for finding URLs.

    Returns:
        A dictionary containing the resolved manufacturer URL and domain.
    """
    manufacturer_url = state.get("manufacturer_url")
    base = {"retry_count": 0, "flagged_urls": []}

    if manufacturer_url:
        normalized = normalize_url(manufacturer_url)
        return {
            **base,
            "manufacturer_url": normalized,
            "manufacturer_domain": extract_domain(normalized),
        }

    manufacturer_name = state.get("manufacturer")
    if not manufacturer_name:
        return {**base, "manufacturer_url": None, "manufacturer_domain": None}

    result = await search_service.get_manufacturer_url(manufacturer_name)
    url = result.get("manufacturer_url")
    if not url:
        return {**base, "manufacturer_url": None, "manufacturer_domain": None}

    normalized = normalize_url(url)
    return {
        **base,
        "manufacturer_url": normalized,
        "manufacturer_domain": result.get("best_domain") or extract_domain(normalized),
    }


def route_after_manufacturer(state: SDSFlowState) -> Literal["fetch_sds", "finalize"]:
    """
    Determines the next step after attempting to resolve the manufacturer.
    If a URL was found, proceed to fetch the SDS. Otherwise, finalize the workflow.
    """
    return "fetch_sds" if state.get("manufacturer_url") else "finalize"
