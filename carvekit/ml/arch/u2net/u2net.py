"""
Modified by Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
Source url: https://github.com/xuebinqin/U-2-Net
License: Apache License 2.0
"""
from typing import Union

import torch
import torch.nn as nn

import math

__all__ = ['U2NETArchitecture']


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NETArchitecture(nn.Module):
    def __init__(self, cfg_type: Union[dict, str] = "full", out_ch: int = 1):
        super(U2NETArchitecture, self).__init__()
        if isinstance(cfg_type, str):
            if cfg_type == "full":
                layers_cfgs = {
                    # cfgs for building RSUs and sides
                    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
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
                }
            else:
                raise ValueError("Unknown U^2-Net architecture conf. name")
        elif isinstance(cfg_type, dict):
            layers_cfgs = cfg_type
        else:
            raise ValueError("Unknown U^2-Net architecture conf. type")
        self.out_ch = out_ch
        self._make_layers(layers_cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))
