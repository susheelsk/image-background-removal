"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import torch
from PIL import Image

from carvekit.ml.wrap.deeplab_v3 import DeepLabV3


def test_init():
    DeepLabV3(load_pretrained=True)
    DeepLabV3(load_pretrained=False).to('cpu')
    DeepLabV3(input_image_size=[128, 256])



def test_preprocessing(deeplabv3_model, converted_pil_image, black_image_pil):
    deeplabv3_model = deeplabv3_model()
    assert isinstance(deeplabv3_model.data_preprocessing(converted_pil_image), torch.FloatTensor) is True
    assert isinstance(deeplabv3_model.data_preprocessing(black_image_pil), torch.FloatTensor) is True


def test_postprocessing(deeplabv3_model, converted_pil_image, black_image_pil):
    deeplabv3_model = deeplabv3_model()
    assert isinstance(deeplabv3_model.data_postprocessing(torch.ones((320, 320), dtype=torch.float64),
                                                          converted_pil_image), Image.Image)


def test_seg(deeplabv3_model, image_pil, image_str, image_path, black_image_pil):
    deeplabv3_model = deeplabv3_model()
    deeplabv3_model([image_pil])
    deeplabv3_model([image_pil, image_str, image_path, black_image_pil])
