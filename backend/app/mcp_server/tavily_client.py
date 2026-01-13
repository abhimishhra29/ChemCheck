"""
Client for interacting with the Tavily Search API.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

# --- Constants ---

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 20


class TavilyClient:
    """An asynchronous client for the Tavily Search API."""

    def __init__(self, client: httpx.AsyncClient, api_key: Optional[str] = None):
        """
        Initializes the TavilyClient.

        Args:
            client: An httpx.AsyncClient for making API requests.
            api_key: The Tavily API key. If not provided, it will be read from
                     the TAVILY_API_KEY_ENV environment variable.

        Raises:
            RuntimeError: If the API key is neither provided nor set as an env var.
        """
        self.client = client
        self.api_key = api_key or os.getenv(TAVILY_API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                "Missing Tavily API key. Set it in your environment via "
                f"{TAVILY_API_KEY_ENV} or pass it to the client constructor."
            )

    async def search(
        self,
        query: str,
        max_results: int = 6,
        search_depth: str = "basic",
    ) -> list[dict]:
        """
        Performs a search using the Tavily API.

        Args:
            query: The search query string.
            max_results: The maximum number of results to return.
            search_depth: The depth of the search ('basic' or 'advanced').

        Returns:
            A list of search result dictionaries.

        Raises:
            RuntimeError: If the API call fails or returns an unexpected shape.
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        }
        resp = await self.client.post(
            TAVILY_SEARCH_URL,
            json=payload,
            timeout=DEFAULT_TIMEOUT_S,
        )

        if not resp.is_success:
            raise RuntimeError(
                f"Tavily search failed: {resp.status_code} {resp.text}"
            )

        data = resp.json()
        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError(
                "Unexpected Tavily response shape: 'results' is not a list."
            )
        return results
