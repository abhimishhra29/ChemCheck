"""
Dependency injection helpers for FastAPI endpoints.
"""
from __future__ import annotations

from fastapi import Depends, Request

from app.agents.ocr_agent import OCRAgent
from app.services.identification_service import IdentificationService


def get_ocr_agent(request: Request) -> OCRAgent:
    """Fetch the shared OCRAgent instance from app state."""
    return request.app.state.ocr_agent


def get_sds_graph(request: Request):
    """Fetch the compiled SDS graph from app state."""
    return request.app.state.sds_graph


def get_identification_service(
    ocr_agent: OCRAgent = Depends(get_ocr_agent),
    sds_graph=Depends(get_sds_graph),
) -> IdentificationService:
    """Construct IdentificationService with shared dependencies."""
    return IdentificationService(ocr_agent=ocr_agent, sds_graph=sds_graph)
