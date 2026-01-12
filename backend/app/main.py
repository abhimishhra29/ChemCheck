from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.project_name)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
