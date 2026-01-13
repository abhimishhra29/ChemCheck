from __future__ import annotations

from functools import partial

import httpx
from langgraph.graph import END, START, StateGraph

from app.agents.ocr_agent import OCRAgent
from app.services.search_service import SearchService
from app.services.validation_service import ValidationService
from app.workflow.nodes.fetch_sds import fetch_sds_node
from app.workflow.nodes.finalize import finalize_node
from app.workflow.nodes.ocr import ocr_node
from app.workflow.nodes.resolve_manufacturer import (
    resolve_manufacturer_node,
    route_after_manufacturer,
)
from app.workflow.nodes.retry import retry_node
from app.workflow.nodes.validation import route_after_validation, validation_node
from app.workflow.utils import SDSFlowState


def build_sds_graph(
    ocr_agent: OCRAgent,
    search_service: SearchService,
    validation_service: ValidationService,
    http_client: httpx.AsyncClient,
):
    """
    Builds the compiled langgraph StateGraph for the SDS workflow.

    This function defines the structure of the workflow, including all nodes and
    the edges between them. It uses dependency injection to provide the necessary
    services (like OCR, search, validation, and HTTP clients) to the individual
    nodes.

    Args:
        ocr_agent: The agent responsible for OCR tasks.
        search_service: The service for searching manufacturer and SDS URLs.
        validation_service: The service for validating the SDS document.
        http_client: The HTTP client for downloading files.

    Returns:
        A compiled langgraph application ready to be executed.
    """
    # Create partial functions for each node, injecting its dependencies.
    # This keeps the node functions pure and easy to test.
    bound_ocr_node = partial(ocr_node, ocr_agent=ocr_agent)
    bound_resolve_manufacturer_node = partial(
        resolve_manufacturer_node, search_service=search_service
    )
    bound_fetch_sds_node = partial(
        fetch_sds_node, search_service=search_service, http_client=http_client
    )
    bound_validation_node = partial(validation_node, validation_service=validation_service)

    # Define the workflow graph structure.
    workflow = StateGraph(SDSFlowState)
    workflow.add_node("ocr", bound_ocr_node)
    workflow.add_node("resolve_manufacturer", bound_resolve_manufacturer_node)
    workflow.add_node("fetch_sds", bound_fetch_sds_node)
    workflow.add_node("validation", bound_validation_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("finalize", finalize_node)

    # Define the edges that connect the nodes.
    workflow.add_edge(START, "ocr")
    workflow.add_edge("ocr", "resolve_manufacturer")
    workflow.add_conditional_edges("resolve_manufacturer", route_after_manufacturer)
    workflow.add_edge("fetch_sds", "validation")
    workflow.add_conditional_edges("validation", route_after_validation)
    workflow.add_edge("retry", "fetch_sds")
    workflow.add_edge("finalize", END)

    return workflow.compile()
