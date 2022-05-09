from typing import Union

from fastapi import Header
from fastapi.responses import Response, JSONResponse
from carvekit.web.deps import config


def Authenticate(x_api_key: Union[str, None] = Header(None)) -> Union[bool, str]:
    if x_api_key in config.auth.allowed_tokens:
        return "allowed"
    elif x_api_key == config.auth.admin_token:
        return "admin"
    elif config.auth.auth is False:
        return "allowed"
    else:
        return False


def handle_response(response, original_image) -> Response:
    """
    Response handler from TaskQueue
    :param response: TaskQueue response
    :param original_image: Original PIL image
    :return: Complete flask response
    """
    response_object = None
    if isinstance(response, dict):
        if response["type"] == "jpg":
            response_object = Response(content=response["data"][0].read(), media_type='image/jpeg')
        elif response["type"] == "png":
            response_object = Response(content=response["data"][0].read(), media_type='image/png')
        elif response["type"] == "zip":
            response_object = Response(content=response["data"][0], media_type='application/zip')
            response_object.headers['Content-Disposition'] = 'attachment; filename=\'no-bg.zip\''

        # Add headers to output result
        response_object.headers["X-Credits-Charged"] = '0'
        response_object.headers["X-Type"] = "other"  # TODO Make support for this
        response_object.headers["X-Max-Width"] = str(original_image.size[0])
        response_object.headers["X-Max-Height"] = str(original_image.size[1])
        response_object.headers["X-Ratelimit-Limit"] = '500'  # TODO Make ratelimit support
        response_object.headers["X-Ratelimit-Remaining"] = '500'
        response_object.headers["X-Ratelimit-Reset"] = '1'
        response_object.headers["X-Width"] = str(response["data"][1][0])
        response_object.headers["X-Height"] = str(response["data"][1][1])

    else:
        response = JSONResponse(content=response[0])
        response.headers["X-Credits-Charged"] = "0"

    return response_object
