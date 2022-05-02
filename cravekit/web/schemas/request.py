import re
from typing import Optional

from pydantic import BaseModel, validator


class Parameters(BaseModel):
    image_file_b64: Optional[str] = ""
    image_url: Optional[str] = ""
    size: Optional[str] = "preview"
    type: Optional[str] = "auto"
    format: Optional[str] = "auto"
    roi: str = "0% 0% 100% 100%"
    crop: bool = False
    crop_margin: Optional[str] = "0px"
    scale: Optional[str] = "original"
    position: Optional[str] = "original"
    channels: Optional[str] = "rgba"
    add_shadow: str = "false"  # Not supported at the moment
    semitransparency: str = "false"  # Not supported at the moment
    bg_color: Optional[str] = ""
    bg_image_url: Optional[str] = ""

    @validator('size')
    def size_is_proper(self, value):
        if value not in ['preview', 'full', 'auto']:
            raise ValueError('Size must be on of: preview, full, auto')
        return value

    @validator('type')
    def type_is_proper(self, value):
        if value not in ['auto', 'product', 'person', 'car']:
            raise ValueError('Size must be one of: auto, person, car, product')
        return value

    @validator('format')
    def format_is_proper(self, value):
        if value not in ['auto', 'jpg', 'png', 'zip']:
            raise ValueError('Size must be one of: auto, jpg, png, zip')
        return value

    @validator('crop_margin')
    def crop_margin_validator(self, value):
        if not re.match(r'[0-9]+(px|%)$', value):
            raise ValueError('crop_margin paramter is not valid')  # TODO: Add support of several values
        if '%' in value and (int(value[:-1]) < 0 or int(value[:-1]) > 100):
            raise ValueError('crop_margin mast be in range between 0% and 100%')
        return value

    @validator('scale')
    def scale_validator(self, value):
        if value != 'original' and (
                not re.match(r'[0-9]+%$', value) or not int(value[:-1]) <= 100 or not int(value[:-1]) >= 10):
            raise ValueError('scale must be original or in between of 10% and 100%')

        if value == 'original':
            return 100

        return int(value[:-1])

    @validator('position')
    def position_validator(self, value, values):
        if len(value.split(' ')) > 2:
            raise ValueError(
                'Position must be a value from 0 to 100 '
                'for both vertical and horizontal axises or for both axises respectively')

        if value == 'original':
            return 'original'
        elif len(value.split(' ')) == 1:
            return [int(value[:-1]), int(value[:-1])]
        else:
            return [int(value.split(' ')[0][:-1]), int(value.split(' ')[1][:-1])]

    @validator('channels')
    def channels_validator(self, value):
        if value not in ['rgba', 'alpha']:
            raise ValueError('Channels must be in: rgba, alpha')

        return value

    @validator('bg_color')
    def bg_color_validator(self, value):
        if not re.match(r'(#{0,1}[0-9a-f]{3}){0,2}$', value):
            raise ValueError('bg_color is not in hex')
        if len(value) and value[0] != '#':
            value = '#' + value
        return value