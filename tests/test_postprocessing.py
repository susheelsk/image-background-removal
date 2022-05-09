"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pytest
from carvekit.pipelines.postprocessing import MattingMethod


def test_init(fba_model, trimap_instance):
    fba_model = fba_model()
    trimap_instance = trimap_instance()
    MattingMethod(fba_model, trimap_instance, "cpu")
    MattingMethod(fba_model, trimap_instance, device="cuda")


def test_seg(matting_method_instance, image_str, image_path, image_pil):
    matting_method_instance = matting_method_instance()
    matting_method_instance(images=[image_str, image_path], masks=[image_pil, image_path])
    with pytest.raises(ValueError):
        matting_method_instance(images=[image_str], masks=[image_pil, image_path])
