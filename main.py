#!/usr/bin/python3
"""
Name: Background removal tool.
Description: This file contains the CLI interface.
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
# Built-in libraries
import gc
import argparse
import logging
from pathlib import Path

# 3rd party libraries
import tqdm

# Libraries of this project
from libs.strings import *
import libs.networks as networks
import libs.preprocessing as preprocessing
import libs.postprocessing as postprocessing

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def __save_image_file__(img, file_path: Path, output_path: Path):
    """
    Saves the PIL image to a file
    :param img: PIL image
    :param file_path: File path object
    :param output_path: Output path object
    """
    if output_path.exists():
        if output_path.is_file():
            img.save(output_path.with_suffix(".png"))
        elif output_path.is_dir():
            img.save(output_path.joinpath(file_path.stem + ".png"))
        else:
            raise ValueError("Something wrong with output path!")
    else:
        if output_path.suffix == '':
            if not output_path.exists():  # create output directory if it doesn't exist
                output_path.mkdir(parents=True, exist_ok=True)
            img.save(output_path.joinpath(file_path.stem + ".png"))
        else:
            if not output_path.parents[0].exists():  # create output directory if it doesn't exist
                output_path.parents[0].mkdir(parents=True, exist_ok=True)
            img.save(output_path.with_suffix(".png"))


def process(input_path, output_path, model_name=MODELS_NAMES[0],
            preprocessing_method_name=PREPROCESS_METHODS[0],
            postprocessing_method_name=POSTPROCESS_METHODS[0], recursive=False):
    """
    Processes the file.
    :param input_path: The path to the image / folder with the images to be processed.
    :param output_path: The path to the save location.
    :param model_name: Model to use.
    :param postprocessing_method_name: Method for image preprocessing
    :param preprocessing_method_name: Method for image post-processing
    :param recursive: Recursive image search in folder
    """
    if input_path is None or output_path is None:
        raise ValueError("Bad parameters! Please specify input path and output path.")

    model = networks.model_detect(model_name)  # Load model

    if not model:
        logger.warning("Warning! You specified an invalid model type. "
                       "For image processing, the model with the best processing quality will be used. "
                       "({})".format(MODELS_NAMES[0]))
        model_name = MODELS_NAMES[0]  # If the model line is wrong, select the model with better quality.
        model = networks.model_detect(model_name)  # Load model

    preprocessing_method = preprocessing.method_detect(preprocessing_method_name)
    postprocessing_method = postprocessing.method_detect(postprocessing_method_name)
    output_path = Path(output_path)

    if isinstance(input_path, str) or isinstance(input_path, Path):
        input_path = Path(input_path)
        if input_path.is_file():
            image = model.process_image(str(input_path.absolute()), preprocessing_method, postprocessing_method)
            __save_image_file__(image, input_path, output_path)
            gc.collect()

        elif input_path.is_dir():
            if not recursive:
                gen_ext = [input_path.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
            else:
                gen_ext = [input_path.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
            files = []
            for gen in gen_ext:
                for f in gen:
                    files.append(f)
            files = set(files)
            for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                image = model.process_image(str(file.absolute()), preprocessing_method, postprocessing_method)
                __save_image_file__(image, file, output_path)
                gc.collect()
        else:
            if input_path.exists():
                raise ValueError("Bad input path parameter! "
                                 "Please indicate the correct path to the file or folder.")
            else:
                raise FileNotFoundError("The input path does not exist!")
    elif isinstance(input_path, list):
        if len(input_path) == 1:
            input_path = Path(input_path[0])

            if input_path.is_file():
                image = model.process_image(str(input_path.absolute()), preprocessing_method, postprocessing_method)
                __save_image_file__(image, input_path, output_path)
                gc.collect()

            elif input_path.is_dir():
                if not recursive:
                    gen_ext = [input_path.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                else:
                    gen_ext = [input_path.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                files = []
                for gen in gen_ext:
                    for f in gen:
                        files.append(f)
                files = set(files)
                for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                    image = model.process_image(str(file.absolute()), preprocessing_method, postprocessing_method)
                    __save_image_file__(image, file, output_path)
                    gc.collect()
            else:
                if input_path.exists():
                    raise ValueError("Bad input path parameter! "
                                     "Please indicate the correct path to the file or folder.")
                else:
                    raise FileNotFoundError("The input path does not exist!")
        else:
            files = []
            for in_p in input_path:
                input_path_p = Path(in_p)
                if input_path_p.is_file():
                    files.append(input_path_p)
                elif input_path_p.is_dir():
                    if not recursive:
                        gen_ext = [input_path_p.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                    else:
                        gen_ext = [input_path_p.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                    for gen in gen_ext:
                        for f in gen:
                            files.append(f)
                else:
                    if not input_path_p.exists():
                        raise FileNotFoundError("The input path does not exist! Path: ", str(input_path_p.absolute()))

            files = set(files)
            for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                image = model.process_image(str(file.absolute()), preprocessing_method, postprocessing_method)
                __save_image_file__(image, file, output_path)
                gc.collect()


def cli():
    """CLI"""
    parser = argparse.ArgumentParser(description=DESCRIPTION, usage=ARGS_HELP)

    parser.add_argument('-i', required=True, nargs="+",
                        help=ARGS["-i"][1], action="store", dest="input_path")
    parser.add_argument('-o', required=True,
                        help=ARGS["-o"][1], action="store", dest="output_path")
    parser.add_argument('-m', required=False,
                        help=ARGS["-m"][1],
                        action="store", dest="model_name", default=MODELS_NAMES[0])
    parser.add_argument('-pre', required=False,
                        help=ARGS["-pre"][1],
                        action="store", dest="preprocessing_method_name", default=PREPROCESS_METHODS[0])
    parser.add_argument('-post', required=False,
                        help=ARGS["-post"][1],
                        action="store", dest="postprocessing_method_name", default=POSTPROCESS_METHODS[0])
    parser.add_argument('--recursive', required=False, default=False,
                        help=ARGS['--recursive'][1], action="store_true", dest="recursive")
    args = parser.parse_args()

    process(args.input_path, args.output_path,
            args.model_name, args.preprocessing_method_name,
            args.postprocessing_method_name, args.recursive)


if __name__ == "__main__":
    cli()
