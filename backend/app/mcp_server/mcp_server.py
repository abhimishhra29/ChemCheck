from __future__ import annotations

from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# The MCP server acts as a client to the backend's shared services.
# This demonstrates reusability and separation of concerns.
# Note: This introduces a dependency from `app.mcp_server` to `app.services`.
from app.services.search_service import SearchService
from app.mcp_server.tavily_client import TavilyClient

# --- Server and Client Initialization ---

mcp = FastMCP("tavily-manufacturer")

# Instantiate the clients and services required by the tools.
http_client = httpx.AsyncClient(timeout=20.0)
tavily_client = TavilyClient(client=http_client)
search_service = SearchService(tavily_client=tavily_client)


# --- Tool Definitions ---

@mcp.tool()
async def get_manufacturer_url(manufacturer_name: str) -> dict:
    """
    Resolves a manufacturer's official website URL using the backend's SearchService.
    """
    return await search_service.get_manufacturer_url(manufacturer_name)


@mcp.tool()
async def get_sds_url(
    manufacturer_url: str,
    cas_number: Optional[str] = None,
    product_name: Optional[str] = None,
    product_code: Optional[str] = None,
    exclude_urls: Optional[list[str]] = None,
) -> dict:
    """
    Finds a Safety Data Sheet (SDS) URL using the backend's SearchService.
    """
    return await search_service.get_sds_url(
        manufacturer_url=manufacturer_url,
        cas_number=cas_number,
        product_name=product_name,
        product_code=product_code,
        exclude_urls=exclude_urls,
    )


# --- Main Execution ---

if __name__ == "__main__":
    # This allows the MCP server to be run as a standalone process.
    mcp.run()
