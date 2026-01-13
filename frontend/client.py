"""
Client for interacting with the ChemCheck backend API.
"""
from __future__ import annotations

import os
from typing import IO

import requests

# Read the base URL from an environment variable, with a default for local dev.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
IDENTIFY_ENDPOINT = f"{API_BASE_URL}/api/v1/identify-chemical"
DEFAULT_TIMEOUT = 120  # Seconds


class APIError(Exception):
    """Custom exception for API-related errors."""


def identify_chemical(
    front_image: IO[bytes] | None = None,
    back_image: IO[bytes] | None = None,
) -> dict:
    """
    Calls the backend's /identify-chemical endpoint.

    Args:
        front_image: An open file-like object for the front image.
        back_image: An open file-like object for the back image.

    Returns:
        The JSON response from the API as a dictionary.

    Raises:
        APIError: If the request fails or returns a non-2xx status code.
    """
    if not front_image and not back_image:
        raise ValueError("At least one image must be provided.")

    files = {}
    if front_image:
        # The files dict for `requests` expects (filename, file-like-object, content_type)
        files["front_image"] = (front_image.name, front_image.read(), front_image.type)
    if back_image:
        files["back_image"] = (back_image.name, back_image.read(), back_image.type)

    try:
        response = requests.post(
            IDENTIFY_ENDPOINT,
            files=files,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        # Wrap the requests exception in our custom APIError
        raise APIError(f"API request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise APIError(f"Failed to decode API response: {exc}") from exc
