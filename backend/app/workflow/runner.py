from __future__ import annotations

from typing import Optional

from app.agents.ocr_agent import OCRResult
from app.workflow.utils import format_output


async def run_sds_flow(app, image_path: str) -> dict:
    state = await app.ainvoke({"image_path": image_path})
    return format_output(state)


async def run_sds_flow_from_ocr(
    app,
    ocr: OCRResult,
    manufacturer_url: Optional[str] = None,
) -> dict:
    state = await app.ainvoke({"ocr": ocr, "manufacturer_url": manufacturer_url})
    return format_output(state)
