import os
import glob
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime

import bg_removal_utility

# Function that takes in an image and and returns a new image with background removed
def convert_image(image):

    print("Converting single image\n")
    start = datetime.datetime.now()

    graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open("mobile_net_model/frozen_inference_graph.pb", "rb").read()) 

    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')
    
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session(graph = graph)

    if sess is None:
        raise RuntimeError("Session Graph is empty")

    resized_im, seg_map = bg_removal_utility.run(sess,image)
    converted_img = bg_removal_utility.drawSegment(resized_im, seg_map)

    print("Image converted\n")
    end = datetime.datetime.now()
    diff = end - start
    print("Time taken to convert image : " + str(diff))

    return converted_img

# Function that takes in the path of the original image, and saves the new image in the specified directory, as well as return it
def convert_image_from_dir(source_dir, destination_dir):

    if(source_dir is None):
        raise RuntimeError("Please specify source folder directory!")
    
    if(destination_dir is None):
        raise RuntimeError("Please specify destination folder directory!")

    jpeg_str = open(source_dir, "rb").read()
    original_img = Image.open(BytesIO(jpeg_str))

    graph = tf.Graph()
    graph_def = None
    graph_def = tf.GraphDef.FromString(open("mobile_net_model/frozen_inference_graph.pb", "rb").read()) 

    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')
    
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session(graph = graph)

    if sess is None:
        raise RuntimeError("Session Graph is empty")

    if(destination_dir is None):
        resized_im, seg_map = bg_removal_utility.run(sess,original_img)
        converted_img = bg_removal_utility.drawSegment(resized_im, seg_map)
    else:
        resized_im, seg_map = bg_removal_utility.run(sess,original_img)
        converted_img = bg_removal_utility.drawSegment(resized_im, seg_map)
        converted_img.save(destination_dir)
    
    print("Done converting image")

    return converted_img

# Converts all images in a directory and saves them in a specified directory, as well as a list of the new images
def convert_all_in_dir(source_dir, destination_dir):

    if(source_dir is None):
        raise RuntimeError("Please specify source folder directory!")

    start = datetime.datetime.now()

    images = getAllImages(source_dir)
    converted_images = []

    graph = tf.Graph()
    graph_def = None
    graph_def = tf.GraphDef.FromString(open("mobile_net_model/frozen_inference_graph.pb", "rb").read()) 

    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')
    
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session(graph = graph)

    if sess is None:
        raise RuntimeError("Session Graph is empty")

    i = 0

    if(destination_dir is not None):
        
        for original_img in images:
                path = os.path.join(destination_dir, 'converted_image')
                path = path + str(i) + ".png"
                resized_im, seg_map = bg_removal_utility.run(sess,original_img)
                converted_img = bg_removal_utility.drawSegment(resized_im, seg_map)
                converted_img.save(path)
                converted_images.append(converted_img)
                print("Index : "+ str(i) + ", Original Image : " + path + ", New Image :  " + path + "\n")
                i += 1
                
    else:
        for original_img in images:
                resized_im, seg_map = bg_removal_utility.run(sess,original_img)
                converted_img = bg_removal_utility.drawSegment(resized_im, seg_map)
                converted_images.append(converted_img)
                print("Index : "+ str(i) + " Converted\n")
                i += 1
    
    print("All images converted!\n")
    end = datetime.datetime.now()
    diff = end - start
    print("Time taken to convert image : " + str(diff))

    return converted_images

# Gets all images in a directory and returns it as a list
def getAllImages(source_dir):
    images = []

    for img_path in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_path)
        jpeg_str = open(img_path, "rb").read()
        original_img = Image.open(BytesIO(jpeg_str))
        images.append(original_img)

    print("Number of images in " + source_dir + " : " + str(len(images)) + "\n")

    return images