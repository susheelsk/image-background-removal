import decimal
from flask import Flask
from flask import request
from tempfile import mkdtemp
from werkzeug import serving
import os
import requests
import ssl
from werkzeug.utils import secure_filename
from flask import jsonify
import random
import string
import json
from uuid import uuid4
import sys
import random

from flask import send_file
from text_detection import detect_text
import traceback
import cv2

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


def run(image, fast=True):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    batch_seg_map = sess.run(
        sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME),
        feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

    seg_map = batch_seg_map[0]

    return resized_image, seg_map


def drawSegment(baseImg, matImg, outputFilePath):
    width, height = baseImg.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = matImg[y,x]
            (r,g,b) = baseImg.getpixel((x,y))
            if color == 0:
                dummyImg[y,x,3] = 0
            else :
                dummyImg[y,x] = [r,g,b,255]
    img = Image.fromarray(dummyImg)
    img.save(outputFilePath)


# define a predict function as an endpoint 
@app.route("/process", methods=["POST"])
def process():
    
    input_path = generate_random_filename("jpg")
    output_path = generate_random_filename("png")

    try:
        url = request.json["url"]
        
   
        download(url, input_path)

        jpeg_str = open(input_path, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
  
        resized_im, seg_map = run(orignal_im, True)

        drawSegment(resized_im, seg_map, output_path)

        callback = send_file(output_path, mimetype='image/png')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory
    global model, net
    
    upload_directory = 'upload'
    create_directory(upload_directory)

    model_name = "frozen_east_text_detection.pb"
    model_url = "https://storage.gra5.cloud.ovh.net/v1/AUTH_18b62333a540498882ff446ab602528b/pretrained-models/" + model_name

    get_model_bin(model_url , model_name)

    net = cv2.dnn.readNet(model_name)

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

