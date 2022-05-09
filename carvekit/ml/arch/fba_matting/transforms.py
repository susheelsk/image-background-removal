"""
Modified by Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
Source url: https://github.com/MarcoForte/FBA_Matting
License: MIT License
"""
import cv2
import numpy as np

group_norm_std = [0.229, 0.224, 0.225]
group_norm_mean = [0.485, 0.456, 0.406]


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def trimap_transform(trimap):
    h, w = trimap.shape[0], trimap.shape[1]

    clicks = np.zeros((h, w, 6))
    for k in range(2):
        if (np.count_nonzero(trimap[:, :, k]) > 0):
            dt_mask = -dt(1 - trimap[:, :, k]) ** 2
            L = 320
            clicks[:, :, 3 * k] = np.exp(dt_mask / (2 * ((0.02 * L) ** 2)))
            clicks[:, :, 3 * k + 1] = np.exp(dt_mask / (2 * ((0.08 * L) ** 2)))
            clicks[:, :, 3 * k + 2] = np.exp(dt_mask / (2 * ((0.16 * L) ** 2)))

    return clicks


def groupnorm_normalise_image(img, format='nhwc'):
    '''
        Accept rgb in range 0,1
    '''
    if (format == 'nhwc'):
        for i in range(3):
            img[..., i] = (img[..., i] - group_norm_mean[i]) / group_norm_std[i]
    else:
        for i in range(3):
            img[..., i, :, :] = (img[..., i, :, :] - group_norm_mean[i]) / group_norm_std[i]

    return img
