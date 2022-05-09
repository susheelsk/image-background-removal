"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import torch
from PIL import Image

from carvekit.ml.wrap.basnet import BASNET


def test_init():
    BASNET(input_tensor_size=[320, 320], load_pretrained=True)
    BASNET(load_pretrained=False)


def test_preprocessing(basnet_model, converted_pil_image, black_image_pil):
    basnet_model = basnet_model()
    assert isinstance(basnet_model.data_preprocessing(converted_pil_image), torch.FloatTensor) is True
    assert isinstance(basnet_model.data_preprocessing(black_image_pil), torch.FloatTensor) is True


def test_postprocessing(basnet_model, converted_pil_image, black_image_pil):
    basnet_model = basnet_model()
    assert isinstance(basnet_model.data_postprocessing(torch.ones((1, 320, 320), dtype=torch.float64),
                                                       converted_pil_image), Image.Image)


def test_seg(basnet_model, image_pil, image_str, image_path, black_image_pil):
    basnet_model = basnet_model()
    basnet_model([image_pil])
    basnet_model([image_pil, image_str, image_path, black_image_pil])
