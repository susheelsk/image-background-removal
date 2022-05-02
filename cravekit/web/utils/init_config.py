import secrets
from os import getenv
from typing import Optional

from loguru import logger

from cravekit.web.schemas.config import WebAPIConfig, MLConfig, AuthConfig


def init_config(config: Optional[WebAPIConfig] = None) -> WebAPIConfig:
    if config is None:
        ml = MLConfig(
            segmentation_network="u2net",
            preprocessing_method=None,
            postprocessing_method='fba'
        )
        auth = AuthConfig(
            auth=True,
            admin_token=secrets.token_hex(64),
            allowed_tokens=[]

        )
        config = WebAPIConfig(ml=ml,
                              auth=auth)

    elif getenv('RAZREZ_PORT', None) is not None:
        config = WebAPIConfig(
            **{
                "port": int(getenv('RAZREZ_PORT')),
                "host": getenv('RAZREZ_HOST'),
                "ml": MLConfig(
                    segmentation_network=getenv('RAZREZ_SEGMENTATION_NETWORK'),
                    preprocessing_method=getenv('RAZREZ_PREPROCESSING_METHOD'),
                    postprocessing_method=getenv('RAZREZ_POSTPROCESSING_METHOD')
                ),
                'auth': AuthConfig(
                    auth=bool(getenv('RAZREZ_AUTH_ENABLE', False)),
                    admin_token=getenv('RAZREZ_ADMIN_TOKEN', "").split(','),
                    allowed_tokens=getenv('RAZREZ_ALLOWED_TOKENS', "").split(',')

                )
            }

        )
    logger.info(f'Admin token for Web API is {config.auth.admin_token}')
    logger.debug(f"Running Web API with this config: {config.json()}")
    return config
