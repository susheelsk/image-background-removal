"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import pytest
import torch
from PIL import Image

from carvekit.ml.wrap.fba_matting import FBAMatting


def test_init():
    FBAMatting(load_pretrained=True)
    FBAMatting(load_pretrained=False)
    FBAMatting(input_tensor_size=[128, 256])


def test_preprocessing(fba_model, converted_pil_image, black_image_pil, image_mask):
    fba_model = fba_model()
    assert isinstance(fba_model.data_preprocessing(converted_pil_image)[0], torch.FloatTensor) is True
    assert isinstance(fba_model.data_preprocessing(black_image_pil)[0], torch.FloatTensor) is True
    assert isinstance(fba_model.data_preprocessing(image_mask)[0], torch.FloatTensor) is True
    with pytest.raises(ValueError):
        assert isinstance(fba_model.data_preprocessing(Image.new("P", (512, 512)))[0], torch.FloatTensor) is True
    fba_model = FBAMatting(device='cuda' if torch.cuda.is_available() else 'cpu',
                           input_tensor_size=1024,
                           batch_size=1,
                           load_pretrained=True)
    assert isinstance(fba_model.data_preprocessing(converted_pil_image)[0], torch.FloatTensor) is True
    assert isinstance(fba_model.data_preprocessing(black_image_pil)[0], torch.FloatTensor) is True
    assert isinstance(fba_model.data_preprocessing(image_mask)[0], torch.FloatTensor) is True
    with pytest.raises(ValueError):
        assert isinstance(fba_model.data_preprocessing(Image.new("P", (512, 512)))[0], torch.FloatTensor) is True


def test_postprocessing(fba_model, converted_pil_image, black_image_pil):
    fba_model = fba_model()
    assert isinstance(fba_model.data_postprocessing(torch.ones((7, 320, 320), dtype=torch.float64),
                                                    black_image_pil.convert("L")), Image.Image)
    with pytest.raises(ValueError):
        assert isinstance(fba_model.data_postprocessing(torch.ones((7, 320, 320), dtype=torch.float64),
                                                        black_image_pil.convert("RGBA")), Image.Image)


def test_seg(fba_model, image_pil, image_str, image_path, black_image_pil, image_trimap):
    fba_model = fba_model()
    fba_model([image_pil], [image_trimap])
    fba_model([image_pil, image_str, image_path],
              [image_trimap, image_trimap, image_trimap])
    fba_model([Image.new('RGB', (512, 512)),
               Image.new('RGB', (512, 512))], [Image.new('L', (512, 512)),
                                               Image.new('L', (512, 512))])
    with pytest.raises(ValueError):
        fba_model([image_pil], [image_trimap, image_trimap])
