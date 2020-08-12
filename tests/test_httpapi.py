"""
Name: tests
Description: This file contains the test code
Version: [release][3.2]
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
import base64
import random
import subprocess
import time
import unittest
from pathlib import Path
import requests
from PIL import Image
import sys


def conv_path(path: str):
    return str(Path(path).absolute())

# TODO rewrite this code!!!!!!

small_img_path = conv_path("docs/imgs/input/7.jpg")
big_img_path = conv_path("docs/imgs/input/5.jpg")

image = Image.open(conv_path("docs/imgs/input/4.jpg"))
small_image = image.copy()
small_image.thumbnail((120, 120))
small_image.save(small_img_path)
big_image = image.copy()
big_image = big_image.resize((3280, 4000))
big_image.save(big_img_path)
del big_image, small_image, image


def run_api():
    proc = subprocess.Popen([sys.executable, "http_api.py"])
    time.sleep(20)


def send_request(data=None, headers=None, files=None, is_json=False):
    if files:
        response = requests.post(
            'http://localhost:5000/api/removebg',
            files=files,
            data=data,
            headers=headers, timeout=120,
        )
    elif is_json is False:
        response = requests.post(
            'http://localhost:5000/api/removebg',
            data=data,
            headers=headers, timeout=120)
    else:
        response = requests.post(
            'http://localhost:5000/api/removebg',
            json=data,
            headers=headers, timeout=120)
    if response.status_code == requests.codes.ok:
        return response
    else:
        return response


def handle(test, case_text, response, expected_code):
    print(case_text)
    if response.status_code != expected_code:
        test.fail(("HTTP API ERROR!!!!!!!\n"
                   "Test Case: {}"
                   "STATUS CODE: {}\n"
                   "TEXT ERROR: {}".format(case_text, response.status_code, response.text)))
        exit(1)
    else:
        return True


def base64_encode_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def test_api(test):
    run_api()
    # Test README.md default case (Multipart)
    response = send_request(data={'size': 'auto'},
                            headers={'X-Api-Key': 'test'},
                            files={'image_file': open(conv_path("docs/imgs/input/4.jpg"), 'rb')},
                            is_json=False)
    handle(test, "Default README.md Multipart case", response, 200)

    # Test README.md url_encoded default case
    response = send_request(data={'size': 'auto',
                                  'image_url':
                                      "https://github.com/OPHoperHPO/image-background-remove-tool/raw/master/docs"
                                      "/imgs/input/4.jpg"},
                            headers={'X-Api-Key': 'test'},
                            files=None,
                            is_json=False)
    handle(test, "Default README.md url_encoded case", response, 200)

    # Test README.md JSON default case
    response = send_request(data={'size': 'auto',
                                  'image_url':
                                      "https://github.com/OPHoperHPO/image-background-remove-tool/raw/master/docs"
                                      "/imgs/input/4.jpg"},
                            headers={'X-Api-Key': 'test'},
                            files=None,
                            is_json=True)
    handle(test, "Default README.md JSON case", response, 200)
    # Test all cases
    all_cases(test)
    return True


def all_cases(test):
    for is_json in [True, False]:
        for input_file in [small_img_path, big_img_path]:
            for input_type in ["image_file", "image_file_b64", "image_url"]:
                for size in ["preview", "full", "auto", "medium", "hd", "4k", "small", "regular"]:
                    for type in ["auto", "person", "product", "car"]:
                        for out_file_format in ["auto", "png", "jpg", "zip"]:
                            for roi in gen_random_roi():
                                for crop in ["true", "false"]:
                                    for crop_margin in gen_random_crop_margin():
                                        for scale in ["10%", "100%", "65%", "original"]:
                                            for position in ["original", "center", "0%", "100%", "50%"]:
                                                for channel in ["rgba", "alpha"]:
                                                    for bg_color in ["81d4fa", "red", "white", "green", ""]:
                                                        if bg_color == "":
                                                            for bg_image_url in [
                                                                "https://github.com/OPHoperHPO/image-background"
                                                                "-remove-tool/raw/master/docs/imgs/input/2.jpg",
                                                                "https://github.com/OPHoperHPO/image-background"
                                                                "-remove-tool/raw/master/docs/imgs/input/3.jpg"]:

                                                                if input_type == "image_url":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": "",
                                                                            "image_url": "https://github.com"
                                                                                         "/OPHoperHPO/image"
                                                                                         "-background-remove-tool/raw"
                                                                                         "/master/docs "
                                                                                         "/imgs/input/4.jpg",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": bg_image_url
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files=None,
                                                                                            is_json=is_json)
                                                                    handle(test, str(setting), response, 200)
                                                                if input_type == "image_file_b64":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": base64_encode_image(
                                                                                input_file),
                                                                            "image_url": "",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": bg_image_url
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files=None,
                                                                                            is_json=is_json)
                                                                    handle(test, str(setting), response, 200)
                                                                if input_type == "image_file":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": "",
                                                                            "image_url": "",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": bg_image_url
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files={'image_file': open(
                                                                                                input_file, 'rb')},
                                                                                            is_json=False)
                                                                    handle(test, str(setting), response,
                                                                           200)

                                                            for bg_image_file in [conv_path("docs/imgs/input/3.jpg"),
                                                                                  conv_path("docs/imgs/input/2.jpg")]:
                                                                if input_type == "image_url":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": "",
                                                                            "image_url": "https://github.com"
                                                                                         "/OPHoperHPO/image"
                                                                                         "-background-remove-tool/raw"
                                                                                         "/master/docs "
                                                                                         "/imgs/input/4.jpg",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": ""
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files={'image_file': open(
                                                                                                input_file, 'rb'),
                                                                                                'bg_image_file': open(
                                                                                                    bg_image_file, 'rb')
                                                                                            },
                                                                                            is_json=False)
                                                                    handle(test, str(setting), response,
                                                                           200)

                                                                if input_type == "image_file_b64":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": base64_encode_image(
                                                                                input_file),
                                                                            "image_url": "",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": ""
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files={'image_file': open(
                                                                                                input_file, 'rb'),
                                                                                                'bg_image_file': open(
                                                                                                    bg_image_file, 'rb')
                                                                                            },
                                                                                            is_json=False)
                                                                    handle(test, str(setting), response,
                                                                           200)

                                                                if input_type == "image_file":
                                                                    setting = \
                                                                        {
                                                                            "image_file_b64": "",
                                                                            "image_url": "",
                                                                            "size": size,
                                                                            "type": type,
                                                                            "format": out_file_format,
                                                                            "roi": roi,
                                                                            "crop": crop,
                                                                            "crop_margin": crop_margin,
                                                                            "scale": scale,
                                                                            "position": position,
                                                                            "channels": channel,
                                                                            "add_shadow": "false",
                                                                            # Not supported at the moment
                                                                            "semitransparency": "false",
                                                                            # Not supported at the moment
                                                                            "bg_color": bg_color,
                                                                            "bg_image_url": ""
                                                                        }
                                                                    response = send_request(data=setting,
                                                                                            headers={
                                                                                                'X-Api-Key': 'test'},
                                                                                            files={'image_file': open(
                                                                                                input_file, 'rb'),
                                                                                                'bg_image_file': open(
                                                                                                    bg_image_file, 'rb')
                                                                                            },
                                                                                            is_json=False)
                                                                    handle(test, str(setting), response,
                                                                           200)
                                                        else:
                                                            if input_type == "image_url":
                                                                setting = \
                                                                    {
                                                                        "image_file_b64": "",
                                                                        "image_url": "https://github.com"
                                                                                     "/OPHoperHPO/image"
                                                                                     "-background-remove-tool/raw"
                                                                                     "/master/docs "
                                                                                     "/imgs/input/4.jpg",
                                                                        "size": size,
                                                                        "type": type,
                                                                        "format": out_file_format,
                                                                        "roi": roi,
                                                                        "crop": crop,
                                                                        "crop_margin": crop_margin,
                                                                        "scale": scale,
                                                                        "position": position,
                                                                        "channels": channel,
                                                                        "add_shadow": "false",
                                                                        # Not supported at the moment
                                                                        "semitransparency": "false",
                                                                        # Not supported at the moment
                                                                        "bg_color": bg_color,
                                                                        "bg_image_url": ""
                                                                    }
                                                                response = send_request(data=setting,
                                                                                        headers={
                                                                                            'X-Api-Key': 'test'},
                                                                                        files=None,
                                                                                        is_json=is_json)
                                                                handle(test, str(setting), response, 200)
                                                            if input_type == "image_file_b64":
                                                                setting = \
                                                                    {
                                                                        "image_file_b64": base64_encode_image(
                                                                            input_file),
                                                                        "image_url": "",
                                                                        "size": size,
                                                                        "type": type,
                                                                        "format": out_file_format,
                                                                        "roi": roi,
                                                                        "crop": crop,
                                                                        "crop_margin": crop_margin,
                                                                        "scale": scale,
                                                                        "position": position,
                                                                        "channels": channel,
                                                                        "add_shadow": "false",
                                                                        # Not supported at the moment
                                                                        "semitransparency": "false",
                                                                        # Not supported at the moment
                                                                        "bg_color": bg_color,
                                                                        "bg_image_url": ""
                                                                    }
                                                                response = send_request(data=setting,
                                                                                        headers={
                                                                                            'X-Api-Key': 'test'},
                                                                                        files=None,
                                                                                        is_json=is_json)
                                                                handle(test, str(setting), response, 200)
                                                            if input_type == "image_file":
                                                                setting = \
                                                                    {
                                                                        "image_file_b64": "",
                                                                        "image_url": "",
                                                                        "size": size,
                                                                        "type": type,
                                                                        "format": out_file_format,
                                                                        "roi": roi,
                                                                        "crop": crop,
                                                                        "crop_margin": crop_margin,
                                                                        "scale": scale,
                                                                        "position": position,
                                                                        "channels": channel,
                                                                        "add_shadow": "false",
                                                                        # Not supported at the moment
                                                                        "semitransparency": "false",
                                                                        # Not supported at the moment
                                                                        "bg_color": bg_color,
                                                                        "bg_image_url": ""
                                                                    }
                                                                response = send_request(data=setting,
                                                                                        headers={'X-Api-Key': 'test'},
                                                                                        files={'image_file': open(
                                                                                            input_file, 'rb')},
                                                                                        is_json=False)
                                                                handle(test, str(setting), response,
                                                                       200)


def gen_random_crop_margin():
    per = "{}%"
    pixel = "{}px"
    rois = []
    for i in range(0, 15):
        r = random.choice([per, pixel])
        r = r.format(random.randint(0, 100))
        rois.append(r)
    return rois


def gen_random_roi():
    roi_per = "{}% {}% {}% {}%"
    roi_pixels = "{}px {}px {}px {}px"
    rois = []
    for i in range(0, 15):
        r = random.choice([roi_per, roi_pixels])
        r = r.format(random.randint(0, 100), random.randint(0, 100),
                     random.randint(0, 100), random.randint(0, 100))
        rois.append(r)
    return rois


class GenTest(unittest.TestCase):
    def test_generator(self):
        self.assertEqual(test_api(self), True)


if __name__ == '__main__':
    unittest.main()
