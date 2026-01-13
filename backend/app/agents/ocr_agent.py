from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, StrictStr, ValidationError

load_dotenv()


# ---------- Data Structures ----------

class OCRPayload(BaseModel):
    """
    A Pydantic model representing the strictly validated data structure expected
    from the LLM's JSON output. It serves as a data validation layer.
    """
    full_text: StrictStr
    product_name: Optional[StrictStr] = None
    product_code: Optional[StrictStr] = None
    cas_number: Optional[StrictStr] = None
    manufacturer_name: Optional[StrictStr] = None
    description: Optional[StrictStr] = None


@dataclass(frozen=True)
class OCRResult:
    """
    A dataclass representing the cleaned, final result of an OCR extraction.
    This object is used throughout the application.
    """
    full_text: str
    product_name: Optional[str] = None
    product_code: Optional[str] = None
    cas_number: Optional[str] = None
    manufacturer_name: Optional[str] = None
    description: Optional[str] = None


# ---------- Agent Definition ----------

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"


class OCRAgent:
    """
    An agent responsible for extracting chemical product information from images
    by calling an external multimodal LLM API (Mistral).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model: str = "mistral-small-latest",
        api_url: str = "https://api.mistral.ai/v1/chat/completions",
    ) -> None:
        """
        Initializes the OCRAgent.

        Args:
            client: An httpx.AsyncClient for making asynchronous API calls.
            model: The name of the Mistral model to use.
            api_url: The URL of the Mistral API endpoint.
        """
        self.client = client
        self.model = model
        self.api_url = api_url

    async def extract_text(self, image_path: str) -> str:
        """
        A convenience method to extract only the full text from an image.

        Args:
            image_path: The local path to the image file.

        Returns:
            The full text extracted from the image.
        """
        return (await self.extract_details(image_path)).full_text

    async def extract_details(self, image_path: str) -> OCRResult:
        """
        Extracts structured details and full text from an image file.

        This method sends the image to the Mistral API and parses the response,
        validating it against a strict schema and cleaning the output.

        Args:
            image_path: The local path to the image file.

        Returns:
            An OCRResult object containing the extracted information.

        Raises:
            RuntimeError: If the API key is missing or the API call fails.
            ValueError: If the image format is unsupported or the API response
                        fails schema validation or is not valid JSON.
        """
        api_key = os.getenv(MISTRAL_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Missing env var {MISTRAL_API_KEY_ENV}. "
                f"Set it in your environment or .env file."
            )

        data_url = self._image_path_to_data_url(image_path)

        prompt = (
            "You are an OCR agent. Extract ALL visible text exactly as shown, "
            "preserving order as best as possible. Then, if possible, identify:\n"
            "- Product Name\n"
            "- product_code: any SKU/catalog/ID/code (often alphanumeric; may include hyphens/slashes).\n"
            "- CAS Number (if available)\n"
            "- Manufacturer Name\n"
            "- Short chemical or product description\n\n"
            "Return ONLY JSON with keys: full_text, product_name, product_code, "
            "cas_number, manufacturer_name, description. Use null when unknown. "
            "Do not fabricate."
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": data_url},
                    ],
                }
            ],
            "temperature": 0,
        }

        resp = await self.client.post(
            self.api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )

        if not resp.is_success:
            raise RuntimeError(f"Mistral OCR request failed: {resp.status_code} {resp.text}")

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {data}") from exc

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected response content type: {type(content)}")

        parsed = self._parse_llm_json_strict(content)

        try:
            validated = OCRPayload.model_validate(parsed)
        except ValidationError as exc:
            raise ValueError(
                f"OCR JSON failed schema validation:\n{exc}\n\nRaw model output:\n{content}"
            ) from exc

        def clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = value.strip()
            return text if text else None

        return OCRResult(
            full_text=validated.full_text.strip(),
            product_name=clean(validated.product_name),
            product_code=clean(validated.product_code),
            cas_number=clean(validated.cas_number),
            manufacturer_name=clean(validated.manufacturer_name),
            description=clean(validated.description),
        )

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> str:
        path = image_path.lower()
        if path.endswith(".png"):
            mime = "image/png"
        elif path.endswith(".jpg") or path.endswith(".jpeg"):
            mime = "image/jpeg"
        else:
            raise ValueError("Only .png, .jpg, .jpeg are supported")

        with open(image_path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode("ascii")

        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_llm_json_strict(text: str) -> dict:
        """
        Strict-ish JSON parser for LLM output:
        - Allows JSON fenced in ```json ... ```
        - Otherwise requires the entire content to be JSON
        - If invalid, raises with raw output included
        """
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model did not return valid JSON: {exc}\n\nRaw model output:\n{text}"
            ) from exc

        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj)}")
        return obj
