"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import uuid
from pathlib import Path

import PIL.Image
import pytest
import torch
from PIL import Image
from carvekit.utils.image_utils import load_image, convert_image, is_image_valid, \
    to_tensor, transparency_paste, add_margin


def test_load_image(image_path, image_pil, image_str):
    assert isinstance(load_image(image_path), Image.Image) is True
    assert isinstance(load_image(image_pil), Image.Image) is True
    assert isinstance(load_image(image_str), Image.Image) is True

    with pytest.raises(ValueError):
        load_image(23)


def test_is_image_valid(image_path, image_pil, image_str):
    assert is_image_valid(image_path) is True
    assert is_image_valid(image_path.with_suffix('.JPG')) is True

    with pytest.raises(ValueError):
        is_image_valid(Path(uuid.uuid1().hex).with_suffix('.jpg'))
    with pytest.raises(ValueError):
        is_image_valid(Path(__file__).parent)
    with pytest.raises(ValueError):
        is_image_valid(image_path.with_suffix('.mp3'))
    with pytest.raises(ValueError):
        is_image_valid(image_path.with_suffix('.MP3'))
    with pytest.raises(ValueError):
        is_image_valid(23)

    assert is_image_valid(image_pil) is True
    assert is_image_valid(Image.new('RGB', (512, 512))) is True
    assert is_image_valid(Image.new('L', (512, 512))) is True
    assert is_image_valid(Image.new('RGBA', (512, 512))) is True

    with pytest.raises(ValueError):
        is_image_valid(Image.new('P', (512, 512)))
    with pytest.raises(ValueError):
        is_image_valid(Image.new('RGB', (32, 10)))


def test_convert_image(image_pil):
    with pytest.raises(ValueError):
        convert_image(Image.new('L', (10, 10)))
    assert convert_image(image_pil.convert('RGBA')).mode == "RGB"


def test_to_tensor(image_pil):
    assert isinstance(to_tensor(image_pil), torch.Tensor)


def test_transparency_paste():
    assert isinstance(transparency_paste(PIL.Image.new("RGBA", (1024, 1024)),
                                         PIL.Image.new("RGBA", (1024, 1024))), PIL.Image.Image)
    assert isinstance(transparency_paste(PIL.Image.new("RGBA", (512, 512)),
                                         PIL.Image.new("RGBA", (512, 512))), PIL.Image.Image)


def test_add_margin():
    assert isinstance(add_margin(PIL.Image.new("RGB", (512, 512)),
                                 10, 10, 10, 10, (10, 10, 10, 10)), PIL.Image.Image) is True
