from __future__ import annotations

from typing import Optional

from app.agents.ocr_agent import OCRResult
from app.workflow.chemcheck import build_sds_graph
from app.workflow.utils import format_output


def run_sds_flow(image_path: str) -> dict:
    app = build_sds_graph()
    state = app.invoke({"image_path": image_path})
    return format_output(state)


def run_sds_flow_from_ocr(
    ocr: OCRResult,
    manufacturer_url: Optional[str] = None,
) -> dict:
    app = build_sds_graph()
    state = app.invoke({"ocr": ocr, "manufacturer_url": manufacturer_url})
    return format_output(state)
