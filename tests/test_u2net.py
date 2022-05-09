"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import pytest
import torch
from PIL import Image

from carvekit.ml.wrap.u2net import U2NET


def test_init():
    U2NET(layers_cfg="full", input_image_size=[320, 320], load_pretrained=True)
    U2NET(layers_cfg='full', load_pretrained=False)
    U2NET(layers_cfg={
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    })
    with pytest.raises(ValueError):
        U2NET(layers_cfg="nan")
    with pytest.raises(ValueError):
        U2NET(layers_cfg=[])


def test_preprocessing(u2net_model, converted_pil_image, black_image_pil):
    u2net_model = u2net_model()
    assert isinstance(u2net_model.data_preprocessing(converted_pil_image), torch.FloatTensor) is True
    assert isinstance(u2net_model.data_preprocessing(black_image_pil), torch.FloatTensor) is True


def test_postprocessing(u2net_model, converted_pil_image, black_image_pil):
    u2net_model = u2net_model()
    assert isinstance(u2net_model.data_postprocessing(torch.ones((1, 320, 320), dtype=torch.float64),
                                                      converted_pil_image), Image.Image)


def test_seg(u2net_model, image_pil, image_str, image_path, black_image_pil):
    u2net_model = u2net_model()
    u2net_model([image_pil])
    u2net_model([image_pil, image_str, image_path, black_image_pil])
