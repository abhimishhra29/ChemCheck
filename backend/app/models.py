from enum import Enum
from typing import Optional

from pydantic import BaseModel


class InputMode(str, Enum):
    front_only = "front_only"
    back_only = "back_only"
    front_back = "front_back"


class FileInfo(BaseModel):
    filename: str
    content_type: Optional[str] = None


class IdentifyResponse(BaseModel):
    message: str
    input_mode: InputMode
    front_image: Optional[FileInfo] = None
    back_image: Optional[FileInfo] = None
