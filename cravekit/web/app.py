import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cravekit import __version__
from cravekit.web.routers.api_router import api_router
from cravekit.web.routers.static_router import static_router
from cravekit.web.deps import config

app = FastAPI(title='RAZREZ Web API', version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)
app.include_router(static_router)

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)

