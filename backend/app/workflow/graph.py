from __future__ import annotations

from app.workflow.chemcheck import build_sds_graph

# Build the graph once when the module is first imported.
# This compiled graph will be shared across the application.
compiled_sds_graph = build_sds_graph()
