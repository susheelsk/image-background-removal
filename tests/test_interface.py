"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import warnings

import torch

from carvekit.api.interface import Interface


def test_init(available_models):
    models, pre_pipes, post_pipes = available_models
    devices = ["cpu", "cuda"]
    for model in models:
        mdl = model()
        for pre_pipe in pre_pipes:
            pre = pre_pipe() if pre_pipe is not None else pre_pipe
            for post_pipe in post_pipes:
                post = post_pipe() if post_pipe is not None else post_pipe
                for device in devices:
                    if device == "cuda" and torch.cuda.is_available() is False:
                        warnings.warn('Cuda GPU is not available! Testing on cuda skipped!')
                        continue
                    inf = Interface(seg_pipe=mdl, post_pipe=post, pre_pipe=pre, device=device)
                    del inf
                del post
            del pre
        del mdl


def test_seg(image_pil, image_str, image_path, available_models):
    models, pre_pipes, post_pipes = available_models
    for model in models:
        mdl = model()
        for pre_pipe in pre_pipes:
            pre = pre_pipe() if pre_pipe is not None else pre_pipe
            for post_pipe in post_pipes:
                post = post_pipe() if post_pipe is not None else post_pipe
                interface = Interface(seg_pipe=mdl, post_pipe=post, pre_pipe=pre,
                                      device='cuda' if torch.cuda.is_available() else 'cpu')
                interface([image_pil, image_str, image_path])
                del post, interface
            del pre
        del mdl
