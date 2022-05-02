"""
Name: transforms
Description: The sub module of the neural network.
Source url: https://github.com/MarcoForte/FBA_Matting
License: MIT License
License:
    Copyright (c) 2020 Marco Forte

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import numpy as np
import torch
import cv2


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def trimap_transform(trimap):
    h, w = trimap.shape[0], trimap.shape[1]

    clicks = np.zeros((h, w, 6))
    for k in range(2):
        if(np.count_nonzero(trimap[:, :, k]) > 0):
            dt_mask = -dt(1 - trimap[:, :, k])**2
            L = 320
            clicks[:, :, 3*k] = np.exp(dt_mask / (2 * ((0.02 * L)**2)))
            clicks[:, :, 3*k+1] = np.exp(dt_mask / (2 * ((0.08 * L)**2)))
            clicks[:, :, 3*k+2] = np.exp(dt_mask / (2 * ((0.16 * L)**2)))

    return clicks


# For RGB !
group_norm_std = [0.229, 0.224, 0.225]
group_norm_mean = [0.485, 0.456, 0.406]


def groupnorm_normalise_image(img, format='nhwc'):
    '''
        Accept rgb in range 0,1
    '''
    if(format == 'nhwc'):
        for i in range(3):
            img[..., i] = (img[..., i] - group_norm_mean[i]) / group_norm_std[i]
    else:
        for i in range(3):
            img[..., i, :, :] = (img[..., i, :, :] - group_norm_mean[i]) / group_norm_std[i]

    return img


def groupnorm_denormalise_image(img, format='nhwc'):
    '''
        Accept rgb, normalised, return in range 0,1
    '''
    if(format == 'nhwc'):
        for i in range(3):
            img[:, :, :, i] = img[:, :, :, i] * group_norm_std[i] + group_norm_mean[i]
    else:
        img1 = torch.zeros_like(img).cuda()
        for i in range(3):
            img1[:, i, :, :] = img[:, i, :, :] * group_norm_std[i] + group_norm_mean[i]
        return img1
    return img
