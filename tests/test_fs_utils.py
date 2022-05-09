"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pytest
from carvekit.utils.fs_utils import save_file
from pathlib import Path
import PIL.Image
import os


def test_save_file():
    save_file(Path("output.png"), Path("input.png"), PIL.Image.new("RGB", (512, 512)))
    os.remove(Path("output.png"))
    save_file(Path(__file__).parent.joinpath("data"), Path("input.png"), PIL.Image.new("RGB", (512, 512)))
    os.remove(Path(__file__).parent.joinpath("data").joinpath('input.png'))
    save_file(Path("output.jpg"), Path("input.jpg"), PIL.Image.new("RGB", (512, 512)))
    os.remove(Path("output.png"))
    with pytest.raises(ValueError):
        save_file(Path("NotExistedPath"), Path("input.png"), PIL.Image.new("RGB", (512, 512)))
    save_file(output=None, input_path=Path("input.png"), image=PIL.Image.new("RGB", (512, 512)))
    os.remove(Path("input_bg_removed.png"))