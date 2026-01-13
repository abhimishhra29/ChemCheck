from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from fastapi import HTTPException, UploadFile

from app.agents.ocr_agent import OCRAgent, OCRResult
from app.models import IdentifyResponse, InputMode, OCRData, SDSResult
from app.utils.file_utils import to_file_info, validate_image
from app.utils.ocr_utils import merge_ocr_results
from app.workflow.utils import format_output


class IdentificationService:
    def __init__(self, *, ocr_agent: OCRAgent, sds_graph: Any):
        self.ocr_agent = ocr_agent
        self.sds_graph = sds_graph

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

        ocr_result = await self._run_ocr(front_image=front_image, back_image=back_image)
        sds_state = await self.sds_graph.ainvoke({"ocr": ocr_result})
        sds_result = format_output(sds_state)

        return IdentifyResponse(
            message="Processed",
            input_mode=input_mode,
            front_image=to_file_info(front_image),
            back_image=to_file_info(back_image),
            ocr=_to_ocr_data(ocr_result),
            sds=SDSResult(**sds_result),
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

    async def _run_ocr(
        self,
        *,
        front_image: Optional[UploadFile],
        back_image: Optional[UploadFile],
    ) -> OCRResult:
        agent = self.ocr_agent
        front_result = None
        back_result = None

        with TemporaryDirectory() as tmpdir:
            if front_image:
                front_path = await _save_upload(front_image, tmpdir, "front")
                front_result = await agent.extract_details(front_path)
            if back_image:
                back_path = await _save_upload(back_image, tmpdir, "back")
                back_result = await agent.extract_details(back_path)

        if front_result and back_result:
            return merge_ocr_results(front_result, back_result)
        return front_result or back_result  # type: ignore[return-value]



async def _save_upload(file: UploadFile, tmpdir: str, stem: str) -> str:
    suffix = Path(file.filename or "").suffix or ".jpg"
    path = Path(tmpdir) / f"{stem}{suffix}"
    content = await file.read()
    path.write_bytes(content)
    return str(path)


def _to_ocr_data(result: OCRResult) -> OCRData:
    return OCRData(
        full_text=result.full_text,
        product_name=result.product_name,
        product_code=result.product_code,
        cas_number=result.cas_number,
        manufacturer_name=result.manufacturer_name,
        description=result.description,
    )


