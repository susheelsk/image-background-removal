"""
Name: Background remove tool.
Description: This file contains the CLI interface.
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
import argparse
import os
import tqdm
import logging
from libs.strings import *
from libs.networks import model_detect
import libs.preprocessing as preprocessing
import libs.postprocessing as postprocessing

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def __work_mode__(path: str):
    """Determines the desired mode of operation"""
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def __save_image_file__(img, file_name, output_path, wmode):
    """
    Saves the PIL image to a file
    :param img: PIL image
    :param file_name: File name
    :param output_path: Output path
    :param wmode: Work mode
    """
    # create output directory if it doesn't exist
    folder = os.path.dirname(output_path)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                if str(e) == "cannot write mode RGBA as JPEG":
                    raise OSError("Error! "
                                  "Please indicate the correct extension of the final file, for example: .png")
                else:
                    raise e
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))


def process(input_path, output_path, model_name="u2net",
            preprocessing_method_name="bbd-fastrcnn", postprocessing_method_name="rtb-bnb"):
    """
    Processes the file.
    :param input_path: The path to the image / folder with the images to be processed.
    :param output_path: The path to the save location.
    :param model_name: Model to use.
    :param postprocessing_method_name: Method for image preprocessing
    :param preprocessing_method_name: Method for image post-processing
    """
    if input_path is None or output_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")

    model = model_detect(model_name)  # Load model
    if not model:
        logger.warning("Warning! You specified an invalid model type. "
                       "For image processing, the model with the best processing quality will be used. "
                       "(u2net)")
        model_name = "u2net"  # If the model line is wrong, select the model with better quality.
        model = model_detect(model_name)  # Load model
    preprocessing_method = preprocessing.method_detect(preprocessing_method_name)
    postprocessing_method = postprocessing.method_detect(postprocessing_method_name)
    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path, preprocessing_method, postprocessing_method)
        __save_image_file__(image, os.path.basename(input_path), output_path, wmode)
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path, preprocessing_method, postprocessing_method)
            __save_image_file__(image, file, output_path, wmode)
    else:
        raise Exception("Bad input parameter! Please indicate the correct path to the file or folder.")


def cli():
    """CLI"""
    parser = argparse.ArgumentParser(description=DESCRIPTION, usage=ARGS_HELP)
    parser.add_argument('-i', required=True,
                        help="Path to input file or dir.", action="store", dest="input_path")
    parser.add_argument('-o', required=True,
                        help="Path to output file or dir.", action="store", dest="output_path")
    parser.add_argument('-m', required=False,
                        help="Model name. Can be {} . U2NET is better to use.".format(MODELS_NAMES),
                        action="store", dest="model_name", default="u2net")
    parser.add_argument('-prep', required=False,
                        help="Preprocessing method. Can be {} . `bbd-fastrcnn` is better to use."
                        .format(PREPROCESS_METHODS),
                        action="store", dest="preprocessing_method_name", default="bbd-fastrcnn")
    parser.add_argument('-postp', required=False,
                        help="Postprocessing method. Can be {} ."
                             " `rtb-bnb` is better to use.".format(POSTPROCESS_METHODS),
                        action="store", dest="postprocessing_method_name", default="rtb-bnb")
    args = parser.parse_args()
    # Parse arguments
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    preprocessing_method_name = args.preprocessing_method_name
    postprocessing_method_name = args.postprocessing_method_name

    if model_name == "test":
        print(input_path, output_path, model_name, preprocessing_method_name, postprocessing_method_name)
    else:
        process(input_path, output_path, model_name, preprocessing_method_name, postprocessing_method_name)


if __name__ == "__main__":
    cli()
