"""
Pydantic models used for API request and response validation and serialization.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class InputMode(str, Enum):
    """Enumeration for the different image input modes."""
    front_only = "front_only"
    back_only = "back_only"
    front_back = "front_back"


class FileInfo(BaseModel):
    """Represents basic information about an uploaded file."""
    filename: str
    content_type: Optional[str] = None


class OCRData(BaseModel):
    """Represents the structured data extracted from an image via OCR."""
    full_text: str
    product_name: Optional[str] = None
    product_code: Optional[str] = None
    cas_number: Optional[str] = None
    manufacturer_name: Optional[str] = None
    description: Optional[str] = None


class SDSResult(BaseModel):
    """Represents the result of the Safety Data Sheet (SDS) workflow."""
    status: Optional[str] = None
    message: Optional[str] = None
    confidence: Optional[str] = None
    manufacturer_url: Optional[str] = None
    sds_url: Optional[str] = None
    sds_query: Optional[str] = None
    sds_attempts: Optional[list[dict]] = None
    validation: Optional[dict] = None


class IdentifyResponse(BaseModel):
    """The main response model for the /identify-chemical endpoint."""
    message: str
    input_mode: InputMode
    front_image: Optional[FileInfo] = None
    back_image: Optional[FileInfo] = None
    ocr: Optional[OCRData] = None
    sds: Optional[SDSResult] = None
