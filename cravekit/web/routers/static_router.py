from pathlib import Path

from fastapi import APIRouter
from starlette.staticfiles import StaticFiles

static_router = APIRouter(prefix='/')
static_router.mount('/', StaticFiles(directory=Path(__file__).parent.joinpath('static'), html=True),
                    name="static")
