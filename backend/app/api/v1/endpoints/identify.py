from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import get_identification_service
from app.models import IdentifyResponse
from app.services.identification_service import IdentificationService

router = APIRouter()


@router.post("/identify-chemical", response_model=IdentifyResponse)
async def identify_chemical(
    front_image: Optional[UploadFile] = File(default=None),
    back_image: Optional[UploadFile] = File(default=None),
    service: IdentificationService = Depends(get_identification_service),
) -> IdentifyResponse:
    return await service.process(front_image=front_image, back_image=back_image)
