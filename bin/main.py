# Background remove tool.
# Module Version: 2.0.6 [Public]
# Modified by Anodev. (https://github.com/OPHoperHPO)
# Original source code: https://github.com/susheelsk/image-background-removal

# Imports
import os
import sys
import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import scipy.ndimage as ndi


# Define functions and classes
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        # Environment init
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        # Start load process
        self.graph = tf.Graph()
        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
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
        seg_map = ndi.zoom(seg_map, (scale_h, scale_w))
        output_image = image.convert('RGB')
        return output_image, seg_map


def draw_segment(base_img, mat_img, filename_d):
    """Postprocessing. Saves complete image."""
    # Get image size
    width, height = base_img.size
    # Create empty numpy array
    dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
    # Create alpha layer from model output
    for x in range(width):
        for y in range(height):
            color = mat_img[y, x]
            (r, g, b) = base_img.getpixel((x, y))
            if color == 0:
                dummy_img[y, x, 3] = 0
            else:
                dummy_img[y, x] = [r, g, b, 255]
    # Restore image object from numpy array
    img = Image.fromarray(dummy_img)
    # Remove file extension
    filename_d = os.path.splitext(filename_d)[0]
    # Save image
    img.save(output_dir + '/' + filename_d + '.png')


def run_visualization(filepath, filename_r):
    """Inferences DeepLab model and visualizes result."""
    try:
        jpeg_str = open(filepath, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + filepath)
        return
    resized_im, seg_map = MODEL.run(orignal_im)
    draw_segment(resized_im, seg_map, filename_r)


if __name__ == "__main__":
    # Parse arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Check parameters
    if input_dir is None or output_dir is None:
        print("Bad parameters. Please specify input dir path and output dir path")
        exit(1)
    # Init model
    modelType = "mobile_net_model"
    if len(sys.argv) > 3 and sys.argv[3] == "1":
        modelType = "xception_model"
    MODEL = DeepLabModel(modelType)

    # Start process
    files = os.listdir(input_dir)
    for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='|image|'):
        filename = file
        file = input_dir + '/' + file
        run_visualization(file, filename)
