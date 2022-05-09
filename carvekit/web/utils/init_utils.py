from os import getenv
from typing import Union

from loguru import logger

from carvekit.web.schemas.config import WebAPIConfig, MLConfig, AuthConfig
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.basnet import BASNET

from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator


def init_config() -> WebAPIConfig:
    default_config = WebAPIConfig()
    config = WebAPIConfig(
        **dict(port=int(getenv('CARVEKIT_PORT', default_config.port)),
               host=getenv('CARVEKIT_HOST', default_config.host),
               ml=MLConfig(
                   segmentation_network=getenv('CARVEKIT_SEGMENTATION_NETWORK', default_config.ml.segmentation_network),
                   preprocessing_method=getenv('CARVEKIT_PREPROCESSING_METHOD', default_config.ml.preprocessing_method),
                   postprocessing_method=getenv('CARVEKIT_POSTPROCESSING_METHOD',
                                                default_config.ml.postprocessing_method),
                   device=getenv('CARVEKIT_DEVICE', default_config.ml.device),
                   batch_size_seg=int(getenv('CARVEKIT_BATCH_SIZE_SEG', default_config.ml.batch_size_seg)),
                   batch_size_matting=int(getenv('CARVEKIT_BATCH_SIZE_MATTING', default_config.ml.batch_size_matting)),
                   seg_mask_size=int(getenv('CARVEKIT_SEG_MASK_SIZE', default_config.ml.seg_mask_size)),
                   matting_mask_size=int(getenv('CARVEKIT_MATTING_MASK_SIZE', default_config.ml.matting_mask_size))
               ), auth=AuthConfig(
                auth=bool(int(getenv('CARVEKIT_AUTH_ENABLE', default_config.auth.auth))),
                admin_token=getenv('CARVEKIT_ADMIN_TOKEN', default_config.auth.admin_token),
                allowed_tokens=default_config.auth.allowed_tokens
                if getenv('CARVEKIT_ALLOWED_TOKENS') is None else getenv('CARVEKIT_ALLOWED_TOKENS').split(',')

            ))

    )

    logger.info(f'Admin token for Web API is {config.auth.admin_token}')
    logger.debug(f"Running Web API with this config: {config.json()}")
    return config


def init_interface(config: Union[WebAPIConfig, MLConfig]) -> Interface:
    if isinstance(config, WebAPIConfig):
        config = config.ml
    if config.segmentation_network == "u2net":
        seg_net = U2NET(device=config.device,
                        batch_size=config.batch_size_seg,
                        input_image_size=config.seg_mask_size)
    elif config.segmentation_network == "deeplabv3":
        seg_net = DeepLabV3(device=config.device,
                            batch_size=config.batch_size_seg,
                            input_image_size=config.seg_mask_size)
    elif config.segmentation_network == "basnet":
        seg_net = BASNET(device=config.device,
                         batch_size=config.batch_size_seg,
                         input_tensor_size=config.seg_mask_size)
    else:
        seg_net = U2NET(device=config.device,
                        batch_size=config.batch_size_seg,
                        input_image_size=config.seg_mask_size)

    if config.preprocessing_method == "stub":
        preprocessing = PreprocessingStub()
    elif config.preprocessing_method == "none":
        preprocessing = None
    else:
        preprocessing = None

    if config.postprocessing_method == "fba":
        fba = FBAMatting(device=config.device,
                         batch_size=config.batch_size_matting,
                         input_tensor_size=config.matting_mask_size)
        trimap_generator = TrimapGenerator()
        postprocessing = MattingMethod(device=config.device,
                                       matting_module=fba,
                                       trimap_generator=trimap_generator)

    elif config.postprocessing_method == "none":
        postprocessing = None
    else:
        postprocessing = None

    interface = Interface(pre_pipe=preprocessing,
                          post_pipe=postprocessing,
                          seg_pipe=seg_net,
                          device=config.device)
    return interface
