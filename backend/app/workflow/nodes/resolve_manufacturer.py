from __future__ import annotations

from typing import Literal

from app.workflow.utils import (
    SDSFlowState,
    extract_domain,
    get_manufacturer_url,
    normalize_url,
)


def resolve_manufacturer_node(state: SDSFlowState) -> dict:
    ocr = state["ocr"]
    manufacturer_url = state.get("manufacturer_url")
    base = {"retry_count": 0, "flagged_urls": []}

    if manufacturer_url:
        normalized = normalize_url(manufacturer_url)
        return {
            **base,
            "manufacturer_url": normalized,
            "manufacturer_domain": extract_domain(normalized),
        }

    manufacturer_name = state.get("manufacturer") or ocr.manufacturer_name
    if not manufacturer_name:
        return {**base, "manufacturer_url": None, "manufacturer_domain": None}

    result = get_manufacturer_url(manufacturer_name)
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
    return "fetch_sds" if state.get("manufacturer_url") else "finalize"
