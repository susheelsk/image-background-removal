import base64
import io
import os
import time
from json import JSONDecodeError
from typing import Optional

import psutil
import requests
from PIL import Image
from fastapi import Header, Depends, Form, File, Request, APIRouter
from fastapi.openapi.models import Response
from pydantic import ValidationError
from starlette.responses import JSONResponse

from cravekit.web.handlers.response import handle_response, Authenticate
from cravekit.web.responses.api import error_dict
from cravekit.web.schemas.request import Parameters
from cravekit.web.deps import config, start_time, queue

api_router = APIRouter(prefix='/api', tags=['api'])


# noinspection PyBroadException
@api_router.post('/removebg')
async def removebg(
        request: Request,
        image_file: Optional[bytes] = File(None),
        auth: bool = Depends(Authenticate),
        content_type: str = Header(""),
        image_file_b64: Optional[str] = Form(None),
        image_url: Optional[str] = Form(None),
        bg_image_file: Optional[bytes] = File(None),
        size: Optional[str] = Form("preview"),
        type: Optional[str] = Form("auto"),
        format: Optional[str] = Form("auto"),
        roi: str = Form("0% 0% 100% 100%"),
        crop: bool = Form(False),
        crop_margin: Optional[str] = Form("0px"),
        scale: Optional[str] = Form("original"),
        position: Optional[str] = Form("original"),
        channels: Optional[str] = Form("rgba"),
        add_shadow: bool = Form(False),  # Not supported at the moment
        semitransparency: bool = Form(False),  # Not supported at the moment
        bg_color: Optional[str] = Form("")
):
    if auth is False:
        return JSONResponse(content=error_dict("Missing API Key"), status_code=401)
    if content_type not in ["application/x-www-form-urlencoded",
                            "application/json"] and "multipart/form-data" not in content_type:
        return JSONResponse(content=error_dict("Invalid request content type"), status_code=400)

    image = None
    bg = None
    parameters = None
    if content_type == "application/x-www-form-urlencoded" or "multipart/form-data" in content_type:
        if image_file_b64 is None and image_url is None and image_file is None:
            return JSONResponse(content=error_dict("File not found"), status_code=400)

        if image_file_b64:
            if len(image_file_b64) == 0:
                return JSONResponse(content=error_dict("Empty image"), status_code=400)
            try:
                image = Image.open(io.BytesIO(base64.b64decode(image_file_b64)))
            except BaseException:
                return JSONResponse(content=error_dict("Error decode image!"), status_code=400)
        elif image_url:
            try:
                image = Image.open(io.BytesIO(requests.get(image_url).content))
            except BaseException:
                return JSONResponse(content=error_dict("Error download image!"), status_code=400)
        elif image_file:
            if len(image_file) == 0:
                return JSONResponse(content=error_dict("Empty image"), status_code=400)
            image = Image.open(io.BytesIO(image_file))

        if bg_image_file:
            if len(bg_image_file) == 0:
                return JSONResponse(content=error_dict("Empty image"), status_code=400)
            bg = Image.open(io.BytesIO(bg_image_file))
        print(scale)
        try:
            parameters = Parameters(
                image_file_b64=image_file_b64,
                image_url=image_url,
                size=size,
                type=type,
                format=format,
                roi=roi,
                crop=crop,
                crop_margin=crop_margin,
                scale=scale,
                position=position,
                channels=channels,
                add_shadow=add_shadow,
                semitransparency=semitransparency,
                bg_color=bg_color,
            )
        except ValidationError as e:
            return Response(content=e.json(), status_code=400, media_type='application/json')

    else:
        payload = None
        try:
            payload = await request.json()
        except JSONDecodeError:
            return JSONResponse(content=error_dict("Empty json"), status_code=400)
        try:
            parameters = Parameters(**payload)
        except ValidationError as e:
            return Response(content=e.json(), status_code=400, media_type='application/json')
        if parameters.image_file_b64 is None and parameters.image_url is None:
            return JSONResponse(content=error_dict("File not found"), status_code=400)

        if parameters.image_file_b64:
            if len(parameters.image_file_b64) == 0:
                return JSONResponse(content=error_dict("Empty image"), status_code=400)
            try:
                image = Image.open(io.BytesIO(base64.b64decode(parameters.image_file_b64)))
            except BaseException:
                return JSONResponse(content=error_dict("Error decode image!"), status_code=400)
        elif parameters.image_url:
            try:
                image = Image.open(io.BytesIO(requests.get(parameters.image_url).content))
            except BaseException:
                return JSONResponse(content=error_dict("Error download image!"), status_code=400)
        if image is None:
            return JSONResponse(content=error_dict("Error download image!"), status_code=400)

    job_id = queue.job_create([parameters.dict(), image, bg, False])

    while queue.job_status(job_id) != "finished":
        time.sleep(1)

    result = queue.job_result(job_id)
    return handle_response(result, image)


@api_router.get("/account")
def account():
    """
    Stub for compatibility with remove.bg api libraries
    """
    return JSONResponse(content={"data": {"attributes": {
        "credits": {"total": 99999, "subscription": 99999, "payg": 99999, "enterprise": 99999},
        "api": {"free_calls": 99999, "sizes": "all"}}}}, status_code=200)


@api_router.get("/status")
def status(auth: bool = Depends(Authenticate)):
    """
    Returns the current server status.
    """
    if not auth:
        return JSONResponse(content=error_dict("Authentication failed"), status_code=401)

    this = psutil.Process(os.getpid())
    data = {
        "status": {
            "program": {
                "used_model": config.ml.segmentation_network,
                "prep_method": config.ml.preprocessing_method,
                "post_method": config.ml.postprocessing_method,
                "use_auth": config.auth.auth
            },
            "server": {
                "process_name": this.name(),
                "cpu_percent": this.cpu_percent(),
                "used_memory": this.memory_info(),
                "uptime": int(time.time() - start_time)
            }
        }
    }
    resp = JSONResponse(content=data, status_code=200)
    resp.headers["X-Credits-Charged"] = "0"
    return resp
