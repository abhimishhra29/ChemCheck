from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.agents.ocr_agent import OCRAgent
from app.api.v1.api import api_router
from app.core.config import settings
from app.services.search_service import SearchService
from app.services.validation_service import ValidationService
from app.workflow.chemcheck import build_sds_graph
from app.mcp_server.tavily_client import TavilyClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of shared resources for the application.
    """
    # --- On startup ---
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Initialize all shared services and clients
        ocr_agent = OCRAgent(client=client)
        tavily_client = TavilyClient(client=client)
        search_service = SearchService(tavily_client=tavily_client)
        validation_service = ValidationService(http_client=client)

        # Build the workflow graph, injecting the required services
        sds_graph = build_sds_graph(
            ocr_agent=ocr_agent,
            search_service=search_service,
            validation_service=validation_service,
            http_client=client,
        )

        # Store shared resources in the application state
        app.state.ocr_agent = ocr_agent
        app.state.search_service = search_service
        app.state.validation_service = validation_service
        app.state.sds_graph = sds_graph

        yield

    # --- On shutdown ---
    # The httpx.AsyncClient is automatically closed by the `async with` block.


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.project_name,
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
