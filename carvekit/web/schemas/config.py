import secrets
from typing import List
from typing_extensions import Literal

import torch.cuda
from pydantic import BaseModel, validator


class AuthConfig(BaseModel):
    """Config for web api token authentication """
    auth: bool = True
    """Enables Token Authentication for API"""
    admin_token: str = secrets.token_hex(32)
    """Admin Token"""
    allowed_tokens: List[str] = [secrets.token_hex(32)]
    """All allowed tokens"""


class MLConfig(BaseModel):
    """Config for ml part of framework"""
    segmentation_network: Literal["u2net", "deeplabv3", "basnet"] = "u2net"
    """Segmentation Network"""
    preprocessing_method: Literal["none", "stub"] = "none"
    """Pre-processing Method"""
    postprocessing_method: Literal["fba", "none"] = "fba"
    """Post-Processing Network"""
    device: str = "cpu"
    """Processing device"""
    batch_size_seg: int = 5
    """Batch size for segmentation network"""
    batch_size_matting: int = 1
    """Batch size for matting network"""
    seg_mask_size: int = 320
    """The size of the input image for the segmentation neural network."""
    matting_mask_size: int = 2048
    """The size of the input image for the matting neural network."""

    @validator('seg_mask_size')
    def seg_mask_size_validator(cls, value: int, values):
        if value > 0:
            return value
        else:
            raise ValueError("Incorrect seg_mask_size!")

    @validator('matting_mask_size')
    def matting_mask_size_validator(cls, value: int, values):
        if value > 0:
            return value
        else:
            raise ValueError("Incorrect matting_mask_size!")

    @validator('batch_size_seg')
    def batch_size_seg_validator(cls, value: int, values):
        if value > 0:
            return value
        else:
            raise ValueError("Incorrect batch size!")

    @validator('batch_size_matting')
    def batch_size_matting_validator(cls, value: int,values):
        if value > 0:
            return value
        else:
            raise ValueError("Incorrect batch size!")

    @validator('device')
    def device_validator(cls, value):
        if torch.cuda.is_available() is False and "cuda" in value:
            raise ValueError("GPU is not available, but specified as processing device!")
        if 'cuda' not in value and "cpu" != value:
            raise ValueError("Unknown processing device! It should be cpu or cuda!")
        return value


class WebAPIConfig(BaseModel):
    """FastAPI app config"""
    port: int = 5000
    """Web API port"""
    host: str = "0.0.0.0"
    """Web API host"""
    ml: MLConfig = MLConfig()
    """Config for ml part of framework"""
    auth: AuthConfig = AuthConfig()
    """Config for web api token authentication """
