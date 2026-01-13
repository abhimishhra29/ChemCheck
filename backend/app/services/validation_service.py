from __future__ import annotations

import json
import os
from typing import Optional

import fitz  # PyMuPDF
import httpx

from app.agents.ocr_agent import OCRResult

# --- Constants ---

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_PDF_BYTES = 8_000_000  # 8 MB limit for downloading PDFs
MAX_TEXT_CHARS = 10_000  # Max characters to extract from the first page
DEFAULT_TIMEOUT_S = 60


class ValidationService:
    """
    A service dedicated to validating a Safety Data Sheet (SDS) document
    against product information extracted from OCR.
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: Optional[str] = None):
        self.http_client = http_client
        self.api_key = api_key or os.getenv(MISTRAL_API_KEY_ENV)

    async def validate_sds(self, sds_url: str, ocr: OCRResult) -> dict:
        """
        Performs validation of an SDS by downloading it and using an LLM.

        Args:
            sds_url: The URL of the SDS PDF to validate.
            ocr: The OCR result containing the product identifiers.

        Returns:
            A dictionary with the validation status and other metadata.
        """
        if not self.api_key:
            return {
                "status": "fail",
                "confidence": "low",
                "reason": "Mistral API is not configured.",
                "method": "mistral",
            }

        if not any([ocr.cas_number, ocr.product_code, ocr.product_name]):
            return {
                "status": "fail",
                "confidence": "low",
                "reason": "No key identifiers (CAS, code, name) from OCR for validation.",
                "method": "none",
            }

        try:
            pdf_bytes = await self._download_pdf(sds_url)
            pdf_text = self._extract_first_page_text(pdf_bytes)
        except Exception as exc:
            return {
                "status": "fail",
                "confidence": "low",
                "reason": f"PDF processing failed: {exc}",
                "method": "mistral",
            }

        if not pdf_text.strip():
            return {
                "status": "fail",
                "confidence": "medium",
                "reason": "No text could be extracted from the PDF's first page.",
                "method": "mistral",
            }

        try:
            validation_result = await self._ask_mistral_to_validate(pdf_text, ocr)
            return validation_result
        except Exception as exc:
            return {
                "status": "fail",
                "confidence": "low",
                "reason": f"Mistral API error during validation: {exc}",
                "method": "mistral",
            }

    async def _download_pdf(self, url: str) -> bytes:
        """Downloads a PDF file from a URL, respecting size limits."""
        async with self.http_client.stream("GET", url, timeout=DEFAULT_TIMEOUT_S) as resp:
            resp.raise_for_status()
            data = bytearray()
            async for chunk in resp.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise ValueError(f"PDF exceeds size limit of {MAX_PDF_BYTES} bytes")
        return bytes(data)

    def _extract_first_page_text(self, pdf_bytes: bytes) -> str:
        """Extracts all text from the first page of a PDF document."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            doc.close()
            return ""

        text = doc.load_page(0).get_text()
        doc.close()

        return text[:MAX_TEXT_CHARS]

    async def _ask_mistral_to_validate(self, pdf_text: str, ocr: OCRResult) -> dict:
        """Uses the Mistral API to validate if the SDS text matches OCR data."""
        looking_for = []
        if ocr.cas_number:
            looking_for.append(f"CAS Number: {ocr.cas_number}")
        if ocr.product_code:
            looking_for.append(f"Product Code: {ocr.product_code}")
        if ocr.product_name:
            looking_for.append(f"Product Name: {ocr.product_name}")
        if ocr.description:
            looking_for.append(f"Description: {ocr.description}")
        looking_for_str = "\n".join(looking_for)

        prompt = f"""You are validating a Safety Data Sheet (SDS) document.

Expected product information:
{looking_for_str}

First page of SDS:
{pdf_text}

Question: Does this SDS match the expected product?

Check if ANY of the expected identifiers (CAS number, product code, or product name) match what's in the SDS text. Be flexible with formatting and minor variations.

Respond ONLY with JSON:
{{
    "is_match": true/false,
    "matched_fields": ["list of what matched, e.g. 'CAS number', 'product code'"],
    "explanation": "brief explanation of what you found"
}}
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        resp = await self.http_client.post(
            MISTRAL_API_URL, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT_S
        )
        resp.raise_for_status()

        result = resp.json()
        content = json.loads(result["choices"][0]["message"]["content"])

        if content.get("is_match"):
            return {
                "status": "pass",
                "confidence": "high",
                "reason": f"Matched: {', '.join(content.get('matched_fields', []))}",
                "method": "mistral",
                "explanation": content.get("explanation"),
            }

        return {
            "status": "fail",
            "confidence": "high",
            "reason": "No matching identifiers found by LLM.",
            "method": "mistral",
            "explanation": content.get("explanation"),
        }
