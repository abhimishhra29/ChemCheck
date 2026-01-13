from __future__ import annotations

from langgraph.graph import END, START, StateGraph
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


def build_sds_graph():
    chemcheck = StateGraph(SDSFlowState)
    chemcheck.add_node("ocr", ocr_node)
    chemcheck.add_node("resolve_manufacturer", resolve_manufacturer_node)
    chemcheck.add_node("fetch_sds", fetch_sds_node)
    chemcheck.add_node("validation", validation_node)
    chemcheck.add_node("retry", retry_node)
    chemcheck.add_node("finalize", finalize_node)

    chemcheck.add_edge(START, "ocr")
    chemcheck.add_edge("ocr", "resolve_manufacturer")
    chemcheck.add_conditional_edges("resolve_manufacturer", route_after_manufacturer)
    chemcheck.add_edge("fetch_sds", "validation")
    chemcheck.add_conditional_edges("validation", route_after_validation)
    chemcheck.add_edge("retry", "fetch_sds")
    chemcheck.add_edge("finalize", END)
    return chemcheck.compile()
