from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

import requests
from pydantic import BaseModel, StrictStr, ValidationError
from dotenv import load_dotenv

load_dotenv()


# ---------- Strict schema (guardrails) ----------

class OCRPayload(BaseModel):
    full_text: StrictStr
    product_name: Optional[StrictStr] = None
    product_code: Optional[StrictStr] = None
    cas_number: Optional[StrictStr] = None
    manufacturer_name: Optional[StrictStr] = None
    description: Optional[StrictStr] = None


@dataclass(frozen=True)
class OCRResult:
    full_text: str
    product_name: Optional[str] = None
    product_code: Optional[str] = None
    cas_number: Optional[str] = None
    manufacturer_name: Optional[str] = None
    description: Optional[str] = None


MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"


class OCRAgent:

    def __init__(
        self,
        *,
        model: str = "mistral-small-latest",
        api_url: str = "https://api.mistral.ai/v1/chat/completions",
        timeout_s: int = 60,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.timeout_s = timeout_s

    def extract_text(self, image_path: str) -> str:
        return self.extract_details(image_path).full_text

    def extract_details(self, image_path: str) -> OCRResult:
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

        resp = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_s,
        )

        if not resp.ok:
            raise RuntimeError(f"Mistral OCR request failed: {resp.status_code} {resp.text}")

        # Strictly pull content; if structure differs, fail loudly
        content = resp.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected response content type: {type(content)}")

        parsed = self._parse_llm_json_strict(content)

        try:
            validated = OCRPayload.model_validate(parsed)  # Pydantic v2
        except ValidationError as e:
            raise ValueError(f"OCR JSON failed schema validation:\n{e}\n\nRaw model output:\n{content}")

        # Normalize whitespace only (no guessing / coercion)
        def clean(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = x.strip()
            return s if s else None

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
        p = image_path.lower()
        if p.endswith(".png"):
            mime = "image/png"
        elif p.endswith(".jpg") or p.endswith(".jpeg"):
            mime = "image/jpeg"
        else:
            raise ValueError("Only .png, .jpg, .jpeg are supported")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_llm_json_strict(text: str) -> dict:
        """
        Strict-ish JSON parser for LLM output:
        - Allows JSON fenced in ```json ... ```
        - Otherwise requires the entire content to be JSON
        - If invalid, raises with raw output included
        """
        t = text.strip()

        if t.startswith("```"):
            t = t.strip("`").strip()
            if t.lower().startswith("json"):
                t = t[4:].strip()

        try:
            obj = json.loads(t)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON: {e}\n\nRaw model output:\n{text}")

        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj)}")
        return obj
