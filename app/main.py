from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.routing import APIRoute

from app.api.main import api_router
from app.core.config import settings
from app.core.logger import Logger

logger = Logger().get_logger()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Vector Operation ready — serving embedding gallery data")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Read-only API exposing the Qdrant face-embedding gallery and "
        "drift telemetry produced by the Airflow vector-operation DAG. "
        "Consumed by edge-sync."
    ),
    version="0.1.0",
    openapi_url=f"{settings.API_PREFIX_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.API_PREFIX_STR)
