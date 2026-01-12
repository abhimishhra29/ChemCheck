from typing import Optional

from fastapi import HTTPException, UploadFile

from app.models import FileInfo

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def validate_image(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PNG or JPEG.",
        )


def to_file_info(file: Optional[UploadFile]) -> Optional[FileInfo]:
    if not file:
        return None
    return FileInfo(filename=file.filename, content_type=file.content_type)
