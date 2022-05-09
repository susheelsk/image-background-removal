from typing import List

from pydantic import BaseModel
from typing import Literal, Optional


class AuthConfig(BaseModel):
    """Config for web api token authentication """
    auth: bool = True
    """Enables Token Authentication for API"""
    admin_token: str = "admin"
    """Admin Token"""
    allowed_tokens: List[str] = ["test"]
    """All allowed tokens"""


class MLConfig(BaseModel):
    """Config for ml part of framework"""
    segmentation_network: str = Literal["u2net", "deeplabv3", "basnet"]  # u2net
    """Segmentation Network"""
    preprocessing_method: Optional[str] = Literal["None"]  # TODO add methods
    """Pre-processing Method"""
    postprocessing_method: Optional[str] = Literal["fba"]  # TODO add methods
    """Post-Processing Network"""


class WebAPIConfig(BaseModel):
    """FastAPI app config"""
    port: int = 5000
    """Web API port"""
    host: str = "0.0.0.0"
    """Web API host"""
    ml: MLConfig
    """Config for ml part of framework"""
    auth: AuthConfig
    """Config for web api token authentication """
