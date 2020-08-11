"""
Name: Post-processing class file
Description: This file contains post-processing classes.
Version: [release][3.3]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import time
import os
from PIL import Image
from PIL import ImageFilter
from libs.strings import POSTPROCESS_METHODS
from libs.networks import models_dir

logger = logging.getLogger(__name__)


def method_detect(method: str):
    """Detects which method to use and returns its object"""
    if method in POSTPROCESS_METHODS:
        if method == "rtb-bnb":
            return RemovingTooTransparentBordersHardAndBlurringHardBorders()
        elif method == "rtb-bnb2":
            return RemovingTooTransparentBordersHardAndBlurringHardBordersTwo()
        elif method == "fba":
            return FBAMatting()
        else:
            return None
    else:
        return False


class RemovingTooTransparentBordersHardAndBlurringHardBordersTwo:
    """
    This is the class for the image post-processing algorithm.
    This algorithm improves the boundaries of the image obtained from the neural network.
    It is based on the principle of removing too transparent pixels
    and smoothing the borders after removing too transparent pixels.
    """

    def __init__(self):
        import cv2
        import skimage
        import numpy as np
        self.cv2 = cv2
        self.skimage = skimage
        self.np = np

        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def __extact_alpha_channel__(image):
        """
        Extracts alpha channel from RGBA image
        :param image: RGBA pil image
        :return: RGB Pil image
        """
        # Extract just the alpha channel
        alpha = image.split()[-1]
        # Create a new image with an opaque black background
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        # Copy the alpha channel to the new image using itself as the mask
        bg.paste(alpha, mask=alpha)
        return bg.convert("RGB")

    def __blur_edges__(self, imaged):
        """
        Blurs the edges of the image
        :param imaged: RGBA Pil image
        :return: RGBA PIL  image
        """
        image = self.np.array(imaged)
        image = self.cv2.cvtColor(image, self.cv2.COLOR_RGBA2BGRA)
        # extract alpha channel
        a = image[:, :, 3]
        # blur alpha channel
        ab = self.cv2.GaussianBlur(a, (0, 0), sigmaX=2, sigmaY=2, borderType=self.cv2.BORDER_DEFAULT)
        # stretch so that 255 -> 255 and 127.5 -> 0
        aa = self.skimage.exposure.rescale_intensity(ab, in_range=(140, 255), out_range=(0, 255))
        # replace alpha channel in input with new alpha channel
        out = image.copy()
        out[:, :, 3] = aa
        image = self.cv2.cvtColor(out, self.cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(image)

    def __remove_too_transparent_borders__(self, mask, tranp_val=31):
        """
        Marks all pixels in the mask with a transparency greater than $tranp_val as opaque.
        Pixels with transparency less than $tranp_val, as fully transparent
        :param tranp_val: Integer value.
        :return: Processed mask
        """
        mask = self.np.array(mask.convert("L"))
        mask = self.np.where(mask > tranp_val, mask, 0)
        mask = self.np.where(mask <= tranp_val, mask, 255)
        return Image.fromarray(mask)

    def run(self, model, image, orig_image):
        """
        Runs an image post-processing algorithm to improve background removal quality.
        :param model: The class of the neural network used to remove the background.
        :param image: Image without background
        :param orig_image: Source image
        """
        mask = self.__remove_too_transparent_borders__(self.__extact_alpha_channel__(image))
        empty = Image.new("RGBA", orig_image.size)
        image = Image.composite(orig_image, empty, mask)
        image = self.__blur_edges__(image)

        image = model.process_image(image)

        mask = self.__remove_too_transparent_borders__(self.__extact_alpha_channel__(image))
        empty = Image.new("RGBA", orig_image.size)
        image = Image.composite(orig_image, empty, mask)
        image = self.__blur_edges__(image)
        return image


class RemovingTooTransparentBordersHardAndBlurringHardBorders:
    """
    This is the class for the image post-processing algorithm.
    This algorithm improves the boundaries of the image obtained from the neural network.
    It is based on the principle of removing too transparent pixels
    and smoothing the borders after removing too transparent pixels.
    The algorithm performs this procedure twice.
    For the first time, the algorithm processes the image from the neural network,
    then sends the processed image back to the neural network, and then processes it again and returns it to the user.
     This method gives the best result in combination with u2net without any preprocessing methods.
    """

    def __init__(self):
        import cv2
        import skimage
        import numpy as np
        self.cv2 = cv2
        self.skimage = skimage
        self.np = np

        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def __extact_alpha_channel__(image):
        """
        Extracts alpha channel from RGBA image
        :param image: RGBA pil image
        :return: RGB Pil image
        """
        # Extract just the alpha channel
        alpha = image.split()[-1]
        # Create a new image with an opaque black background
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        # Copy the alpha channel to the new image using itself as the mask
        bg.paste(alpha, mask=alpha)
        return bg.convert("RGB")

    def __blur_edges__(self, imaged):
        """
        Blurs the edges of the image
        :param imaged: RGBA Pil image
        :return: RGBA PIL  image
        """
        image = self.np.array(imaged)
        image = self.cv2.cvtColor(image, self.cv2.COLOR_RGBA2BGRA)
        # extract alpha channel
        a = image[:, :, 3]
        # blur alpha channel
        ab = self.cv2.GaussianBlur(a, (0, 0), sigmaX=2, sigmaY=2, borderType=self.cv2.BORDER_DEFAULT)
        # stretch so that 255 -> 255 and 127.5 -> 0
        # noinspection PyUnresolvedReferences
        aa = self.skimage.exposure.rescale_intensity(ab, in_range=(140, 255), out_range=(0, 255))
        # replace alpha channel in input with new alpha channel
        out = image.copy()
        out[:, :, 3] = aa
        image = self.cv2.cvtColor(out, self.cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(image)

    def __remove_too_transparent_borders__(self, mask, tranp_val=31):
        """
        Marks all pixels in the mask with a transparency greater than tranp_val as opaque.
        Pixels with transparency less than tranp_val, as fully transparent
        :param tranp_val: Integer value.
        :return: Processed mask
        """
        mask = self.np.array(mask.convert("L"))
        mask = self.np.where(mask > tranp_val, mask, 0)
        mask = self.np.where(mask <= tranp_val, mask, 255)
        return Image.fromarray(mask)

    def run(self, _, image, orig_image):
        """
        Runs an image post-processing algorithm to improve background removal quality.
        :param _: The class of the neural network used to remove the background.
        :param image: Image without background
        :param orig_image: Source image
        """
        mask = self.__remove_too_transparent_borders__(self.__extact_alpha_channel__(image))
        empty = Image.new("RGBA", orig_image.size)
        image = Image.composite(orig_image, empty, mask)
        image = self.__blur_edges__(image)
        return image


class FBAMatting:
    """
    This is the class for the image post-processing algorithm.
    This algorithm improves the borders of the image when removing the background from images with hair, etc. using
    [FBA Matting](https://github.com/MarcoForte/FBA_Matting) neural network.
    """

    def __init__(self):
        import cv2
        import skimage
        import numpy as np
        from libs.trimap_module import trimap

        self.trimap = trimap
        self.cv2 = cv2
        self.skimage = skimage
        self.np = np

        self.__fba__ = FBAMattingNeural()

    def __remove_too_transparent_borders__(self, mask, tranp_val=231):
        """
        Marks all pixels in the mask with a transparency greater than tranp_val as opaque.
        Pixels with transparency less than tranp_val, as fully transparent
        :param tranp_val: Integer value.
        :return: Processed mask
        """
        mask = self.np.where(mask > tranp_val, mask, 0)
        mask = self.np.where(mask <= tranp_val, mask, 255)
        return mask

    @staticmethod
    def __extact_alpha_channel__(image):
        """
        Extracts alpha channel from RGBA image
        :param image: RGBA pil image
        :return: L Pil image
        """
        # Extract just the alpha channel
        alpha = image.split()[-1]
        # Create a new image with an opaque black background
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        # Copy the alpha channel to the new image using itself as the mask
        bg.paste(alpha, mask=alpha)
        return bg.convert("L")

    def __png2trimap__(self, png, model_name):
        """
        Calculates trimap from png image.
        :param png: PIL RGBA Image
        :return: PIL trimap
        :model_name: Model name (deeplabv3 or u2net or etc.)
        """
        mask = self.np.array(self.__extact_alpha_channel__(png))
        if model_name == "deeplabv3":
            trimap = self.trimap(mask, "", 50, 2, erosion=12)
        else:
            mask = self.__remove_too_transparent_borders__(mask)
            trimap = self.trimap(mask, "", 50, 2, erosion=1)
        if not isinstance(trimap, bool):
            return Image.fromarray(trimap)
        else:
            return False

    def run(self, model, image, orig_image):
        """
        Runs an image post-processing algorithm to improve background removal quality.
        :param model: The class of the neural network used to remove the background.
        :param image: Image without background
        :param orig_image: Source image
        """
        w, h = image.size
        if w > 1024 or h > 1024:
            image.thumbnail((1024, 1024))
        trimap = self.__png2trimap__(image, model.model_name)
        trimap = trimap.resize((w, h))
        if not isinstance(trimap, bool):  # If something is wrong with trimap, skip processing
            mask = self.__fba__.process_image(orig_image, trimap)
            image = self.__process_mask__(orig_image, mask)
            return image
        else:
            return image

    @staticmethod
    def __apply_mask__(image, mask):
        """
        Applies a mask to an image.
        :param image: PIL Image
        :param mask: L PIL Image
        :return: PIL Image
        """
        empty = Image.new("RGBA", image.size)
        image = Image.composite(image, empty, mask)
        return image

    def __process_mask__(self, orig_image, mask):
        """
        Applies a mask, removes the gray stroke near the borders of the image without a background (Needs refinement).
        :param orig_image: Original PIL image
        :param mask: Mask PIL Image
        :return: Finished image
        """
        image = self.__apply_mask__(orig_image, mask)

        # Image Border Improvement Algorithm
        mask = 255 - self.np.array(mask)  # Invert mask
        mask_unsh = Image.fromarray(mask).filter(ImageFilter.UnsharpMask(5, 120, 3))
        image_unsh = self.__apply_mask__(image, mask_unsh)
        new_mask = self.__extact_alpha_channel__(image_unsh)
        new_mask = Image.fromarray(self.__remove_too_transparent_borders__(255 - self.np.array(new_mask), 0))
        image = self.__apply_mask__(image, new_mask)
        image = self.np.array(image) - self.np.array(
            image_unsh)  # The similarity of the "grain extraction" mode in GIMP

        # image = self.color_correction(image)  # TODO Make RGB color correction around the edges
        image = Image.fromarray(image)
        return image


class FBAMattingNeural:
    """
    FBA Matting Neural Network to improve edges on image.
    """

    class Config:
        """FBA Matting config"""
        encoder = "resnet50_GN_WS"
        decoder = "fba_decoder"
        weights = os.path.join(models_dir,
                               "fba_matting", "fba_matting.pth")

    def __init__(self):
        import cv2
        import numpy as np
        import torch
        from libs.fba.transforms import trimap_transform, groupnorm_normalise_image
        from libs.fba.models import build_model

        self.trimap_transform = trimap_transform
        self.groupnorm_normalise_image = groupnorm_normalise_image
        self.np = np
        self.cv2 = cv2
        self.torch = torch

        logger.debug("Loading FBA Matting neural network")
        self.model = build_model(self.Config)  # Initialize the model
        self.model.eval()

    def __numpy2torch__(self, x):
        """Converts numpy arr to torch tensor"""
        if self.torch.cuda.is_available():
            return self.torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
        else:
            return self.torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cpu()

    def __scale_input(self, x, scale: float, scale_type):
        """ Scales inputs to multiple of 8. """
        h, w = x.shape[:2]
        h1 = int(self.np.ceil(scale * h / 8) * 8)
        w1 = int(self.np.ceil(scale * w / 8) * 8)
        x_scale = self.cv2.resize(x, (w1, h1), interpolation=scale_type)
        return x_scale

    @staticmethod
    def __load_image__(data_input):
        """
        Loads an image file for other processing
        :param data_input: Path to image file or PIL image
        :return: image
        """
        if isinstance(data_input, str):
            try:
                image = Image.open(data_input)
                image = image.convert("RGB")
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data_input)
                return False, False
        else:
            image = data_input.convert("RGB")
        return image

    def process_image(self, image, trimap):
        """
        Predict alpha mask.
        :param image: Path to image or PIL image.
        :param trimap: Path to trimap or PIL image
        :return: alpha mask
        """
        start_time = time.time()  # Time counter
        logger.debug('Start!')
        image = self.__load_image__(image)
        trimap = self.__load_image__(trimap)
        w, h = image.size
        # Reduce image size to increase processing speed.
        if w > 1024 or h > 1024:
            image.thumbnail((1024, 1024))
            trimap.thumbnail((1024, 1024))
        alpha = self.__get_output__(image, trimap)
        alpha = alpha.resize((w, h))
        logger.debug('Finished! Time Spent: {}'.format(str(time.time() - start_time)))
        return alpha

    def __get_output__(self, orig_image, trimap):
        """
        Predict alpha mask
        :param orig_image: PIl original image with background
        :param trimap: Trimap PIL image
        :return: Alpha PIL image
        """
        orig_image = self.np.array(orig_image)
        orig_image = (orig_image / 255.0)[:, :, ::-1]
        trimap = self.np.array(trimap.convert("L"))
        trimap2 = trimap / 255.0
        h, w = trimap2.shape
        trimap = self.np.zeros((h, w, 2))
        trimap[trimap2 == 1, 1] = 1
        trimap[trimap2 == 0, 0] = 1
        h, w = trimap.shape[:2]
        image_scale_np = self.__scale_input(orig_image, 1.0, self.cv2.INTER_LANCZOS4)
        trimap_scale_np = self.__scale_input(trimap, 1.0, self.cv2.INTER_LANCZOS4)
        with self.torch.no_grad():
            image_torch = self.__numpy2torch__(image_scale_np)
            trimap_torch = self.__numpy2torch__(trimap_scale_np)
            trimap_transformed_torch = self.__numpy2torch__(self.trimap_transform(trimap_scale_np))
            image_transformed_torch = self.groupnorm_normalise_image(image_torch.clone(), format='nchw')
            output = self.model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)
            output = self.cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), self.cv2.INTER_LANCZOS4)
        alpha = output[:, :, 0]
        alpha[trimap[:, :, 0] == 1] = 0
        alpha[trimap[:, :, 1] == 1] = 1
        return Image.fromarray(alpha * 255).convert("L")
