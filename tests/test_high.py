"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

from carvekit.api.high import HiInterface


def test_init():
    HiInterface(batch_size_seg=1, batch_size_matting=4,
                device='cpu',
                seg_mask_size=160, matting_mask_size=1024)
    HiInterface(batch_size_seg=0, batch_size_matting=0,
                device='cpu',
                seg_mask_size=0, matting_mask_size=0)
