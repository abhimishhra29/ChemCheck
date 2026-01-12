from typing import Optional

from fastapi import HTTPException, UploadFile

from app.models import IdentifyResponse, InputMode
from app.utils.file_utils import to_file_info, validate_image


class IdentificationService:
    async def process(
        self,
        front_image: Optional[UploadFile],
        back_image: Optional[UploadFile],
    ) -> IdentifyResponse:
        if not front_image and not back_image:
            raise HTTPException(status_code=400, detail="Provide front_image or back_image.")

        if front_image:
            validate_image(front_image)
        if back_image:
            validate_image(back_image)

        input_mode = self._get_input_mode(front_image, back_image)

        return IdentifyResponse(
            message="Files received",
            input_mode=input_mode,
            front_image=to_file_info(front_image),
            back_image=to_file_info(back_image),
        )

    @staticmethod
    def _get_input_mode(
        front_image: Optional[UploadFile],
        back_image: Optional[UploadFile],
    ) -> InputMode:
        if front_image and back_image:
            return InputMode.front_back
        if front_image:
            return InputMode.front_only
        return InputMode.back_only


def get_identification_service() -> IdentificationService:
    return IdentificationService()
