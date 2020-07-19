"""
Name: Neural networks file.
Description: This file contains neural network classes.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
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
import os
import time

import numpy as np
from PIL import Image
from skimage import io, transform

from libs import strings

logger = logging.getLogger(__name__)


def model_detect(model_name):
    """Detects which model to use and returns its object"""
    models_names = strings.MODELS_NAMES
    if model_name in models_names:
        if model_name == "xception_model" or model_name == "mobile_net_model":
            return TFSegmentation(model_name)
        elif "u2net" in model_name:
            return U2NET(model_name)
        elif "basnet" == model_name:
            return BasNet(model_name)
        else:
            return False
    else:
        return False


class U2NET:
    """U^2-Net model interface"""

    def __init__(self, name="u2net"):
        import torch
        from torch.autograd import Variable
        from libs.u2net import U2NET as U2NET_DEEP
        from libs.u2net import U2NETP as U2NETP_DEEP
        self.Variable = Variable
        self.torch = torch
        self.U2NET_DEEP = U2NET_DEEP
        self.U2NETP_DEEP = U2NETP_DEEP

        if name == 'u2net':  # Load model
            logger.debug("Loading a U2NET model (176.6 mb) with better quality but slower processing.")
            net = self.U2NET_DEEP()
        elif name == 'u2netp':
            logger.debug("Loading a U2NETp model (4 mb) with lower quality but fast processing.")
            net = self.U2NETP_DEEP()
        else:
            raise Exception("Unknown u2net model!")
        try:
            project_dir = os.path.join(os.path.dirname(__file__), "..")
            if self.torch.cuda.is_available():
                net.load_state_dict(self.torch.load(os.path.join(project_dir, "models", name, name + ".pth")))
                net.cuda()
            else:
                net.load_state_dict(self.torch.load(os.path.join(project_dir, "models", name, name + ".pth"), map_location="cpu"))

        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, data, preprocessing=None, postprocessing=None):
        """
        Removes background from image and returns PIL RGBA Image.
        :param data: Path to image or PIL image
        :param preprocessing: Image Pre-Processing Algorithm Class (optional)
        :param postprocessing: Image Post-Processing Algorithm Class (optional)
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, org_image = self.__load_image__(data)  # Load image
        if image is False or org_image is False:
            return False
        if preprocessing:  # If an algorithm that preprocesses is specified,
            # then this algorithm should immediately remove the background
            image = preprocessing.run(self, image, org_image)
        else:
            image = self.__get_output__(image, org_image)  # If this is not, then just remove the background
        if postprocessing:  # If a postprocessing algorithm is specified, we send it an image without a background
            image = postprocessing.run(self, image, org_image)
        return image

    def __get_output__(self, image, org_image):
        """
        Returns output from a neural network
        :param image: Prepared Image
        :param org_image: Original pil image
        :return: Image without background
        """
        start_time = time.time()  # Time counter
        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)  # Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)
        # Prepare mask
        logger.debug("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # Apply mask to image
        logger.debug("Apply mask to image")
        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __load_image__(self, data):
        """
        Loads an image file for other processing
        :param data: Path to image file or PIL image
        :return: image tensor, original pil image
        """
        image_size = 320  # Size of the input and output image for the model
        if isinstance(data, str):
            try:
                image = io.imread(data)  # Load image if there is a path
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False, False
            pil_image = Image.fromarray(image)
        else:
            image = np.array(data)  # Convert PIL image to numpy arr
            pil_image = data
        image = transform.resize(image, (image_size, image_size), mode='constant')  # Resize image
        image = self.__ndrarray2tensor__(image)  # Convert image from numpy arr to tensor
        return image, pil_image

    def __ndrarray2tensor__(self, image: np.ndarray):
        """
        Converts a NumPy array to a tensor
        :param image: Image numpy array
        :return: Image tensor
        """
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return self.torch.from_numpy(tmp_img)

    def __normalize__(self, predicted):
        """Normalize the predicted map"""
        ma = self.torch.max(predicted)
        mi = self.torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Prepares mask"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        mask = Image.fromarray(predict_np * 255).convert("L")
        mask = mask.resize(image_size, resample=Image.BILINEAR)
        return mask


class BasNet:
    """BasNet model interface"""

    def __init__(self, name="basnet"):
        import torch
        from torch.autograd import Variable
        from libs.basnet import BASNet as BASNet_DEEP

        self.Variable = Variable
        self.torch = torch
        self.BASNet_DEEP = BASNet_DEEP

        if name == 'basnet':  # Load model
            logger.debug("Loading a BASNet model.")
            net = self.BASNet_DEEP(3, 1)
        else:
            raise Exception("Unknown BASNet model")
        try:
            if self.torch.cuda.is_available():
                net.load_state_dict(self.torch.load(os.path.join("models", name, name + '.pth')))
                net.cuda()
            else:
                net.load_state_dict(self.torch.load(os.path.join("models", name, name + '.pth'), map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, data, preprocessing=None, postprocessing=None):
        """
        Removes background from image and returns PIL RGBA Image.
        :param data: Path to image or PIL image
        :param preprocessing: Image Pre-Processing Algorithm Class (optional)
        :param postprocessing: Image Post-Processing Algorithm Class (optional)
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, orig_image = self.__load_image__(data)  # Load image
        if image is False or orig_image is False:
            return False
        if preprocessing:  # If an algorithm that preprocesses is specified,
            # then this algorithm should immediately remove the background
            image = preprocessing.run(self, image, orig_image)
        else:
            image = self.__get_output__(image, orig_image)  # If this is not, then just remove the background
        if postprocessing:  # If a postprocessing algorithm is specified, we send it an image without a background
            image = postprocessing.run(self, image, orig_image)
        return image

    def __get_output__(self, image, org_image):
        """
        Returns output from a neural network
        :param image: Prepared Image
        :param org_image: Original pil image
        :return: Image without background
        """
        start_time = time.time()  # Time counter
        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        mask, d2, d3, d4, d5, d6, d7, d8 = self.__net__(image)  # Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)
        # Prepare mask
        logger.debug("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # Apply mask to image
        logger.debug("Apply mask to image")
        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __load_image__(self, data):
        """
        Loads an image file for other processing
        :param data: Path to image file or PIL image
        :return: image tensor, Original Pil Image
        """
        image_size = 256  # Size of the input and output image for the model
        if isinstance(data, str):
            try:
                image = io.imread(data)  # Load image if there is a path
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False, False
            pil_image = Image.fromarray(image)
        else:
            image = np.array(data)  # Convert PIL image to numpy arr
            pil_image = data
        image = transform.resize(image, (image_size, image_size), mode='constant')  # Resize image
        image = self.__ndrarray2tensor__(image)  # Convert image from numpy arr to tensor
        return image, pil_image

    def __ndrarray2tensor__(self, image: np.ndarray):
        """
        Converts a NumPy array to a tensor
        :param image: Image numpy array
        :return: Image tensor
        """
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return self.torch.from_numpy(tmp_img)

    def __normalize__(self, predicted):
        """Normalize the predicted map"""
        ma = self.torch.max(predicted)
        mi = self.torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Prepares mask"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        mask = Image.fromarray(predict_np * 255).convert("L")
        mask = mask.resize(image_size, resample=Image.BILINEAR)
        return mask


class TFSegmentation(object):
    """Class to load Deeplabv3 model and run inference."""
    def __init__(self, model_type):
        """Creates and loads pretrained deeplab model."""
        import scipy.ndimage as ndi
        import tensorflow as tf
        self.tf = tf
        self.ndi = ndi

        # Environment init
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        # Start load process
        self.graph = self.tf.Graph()
        try:
            graph_def = self.tf.compat.v1.GraphDef.FromString(open(os.path.join("models", model_type, "model",
                                                                                "frozen_inference_graph.pb"),
                                                                   "rb").read())
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        logger.warning("Loading a DeepLab model ({})! "
                       "This is an outdated model with poorer image quality and processing time."
                       "Better use the U2NET model instead of this one!".format(model_type))
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            self.tf.import_graph_def(graph_def, name='')
        self.sess = self.tf.compat.v1.Session(graph=self.graph)

    @staticmethod
    def __load_image__(data):
        """
        Loads an image file for other processing
        :param data: Path to image file or PIL image
        :return: Pil Image, Pil Image
        """
        if isinstance(data, str):
            try:
                image = Image.open(data)  # Load image if there is a path
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False
        else:
            image = data
        return image, image

    def process_image(self, data, preprocessing=None, postprocessing=None):
        """
        Removes background from image and returns PIL RGBA Image.
        :param data: Path to image or PIL image
        :param preprocessing: Image Pre-Processing Algorithm Class (optional)
        :param postprocessing: Image Post-Processing Algorithm Class (optional)
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, org_image = self.__load_image__(data)  # Load image
        if image is False or org_image is False:
            return False
        if preprocessing:  # If an algorithm that preprocesses is specified,
            # then this algorithm should immediately remove the background
            image = preprocessing.run(self, image, org_image)
        else:
            image = self.__get_output__(image, org_image)  # If this is not, then just remove the background
        if postprocessing:  # If a postprocessing algorithm is specified, we send it an image without a background
            image = postprocessing.run(self, image, org_image)
        return image

    def __get_output__(self, image, _=None):
        """
        Returns output from a neural network
        :param image: Prepared Image
        :param _: Not used argument for compatibility with pre-processing module
        :return: Image without background
        """
        start_time = time.time()  # Time counter
        seg_map = self.__predict__(image)
        logger.debug('Finished mask creation')
        image = image.convert('RGB')
        logger.debug("Mask overlay completed")
        image = self.__draw_segment__(image, seg_map)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __predict__(self, image):
        """Image processing."""
        # Get image size
        width, height = image.size
        # Calculate scale value
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # Resize image
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        # Send image to model
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        # Get model output
        seg_map = batch_seg_map[0]
        # Get new image size and original image size
        width, height = resized_image.size
        width2, height2 = image.size
        # Calculate scale
        scale_w = width2 / width
        scale_h = height2 / height
        # Zoom numpy array for original image
        seg_map = self.ndi.zoom(seg_map, (scale_h, scale_w))
        return seg_map

    @staticmethod
    def __draw_segment__(image, alpha_channel):
        """Postprocessing. Returns complete image."""
        # Get image size
        width, height = image.size
        # Create empty numpy array
        dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
        # Create alpha layer from model output
        for x in range(width):
            for y in range(height):
                color = alpha_channel[y, x]
                (r, g, b) = image.getpixel((x, y))
                if color == 0:
                    dummy_img[y, x, 3] = 0
                else:
                    dummy_img[y, x] = [r, g, b, 255]
        # Restore image object from numpy array
        img = Image.fromarray(dummy_img)
        return img
