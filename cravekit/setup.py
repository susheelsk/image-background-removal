"""
Name: Configuration Tool
Description: This file contains the code for the installation tool.
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
import os
import sys
import tarfile
import argparse

# Third party libraries
import gdown

# Libraries of this project
from libs.networks import models_dir

config = {
    "download_url": "https://github.com/OPHoperHPO/image-background-remove-tool/releases/download/3.2/",
    "models": {
        "u2net": {
            "file": "u2net.pth",
            "dir": "u2net"
        },
        "basnet": {
            "file": "basnet.pth",
            "dir": "basnet"
        },
        "u2netp": {
            "file": "u2netp.pth",
            "dir": "u2netp"
        },
        "fba_matting": {
            "file": "fba_matting.pth",
            "dir": "fba_matting"
        },
    }
}


def __setup_model__(model_name, silent):
    model_url = config["download_url"] + config["models"][model_name]["file"]
    model_dir = os.path.join(models_dir, config["models"][model_name]["dir"])
    model_file = os.path.join(model_dir, config["models"][model_name]["file"])
    if not silent:
        print("Create {} dir".format(model_name))
    os.makedirs(model_dir, exist_ok=True)  # Create dir
    if not silent:
        print("Download {} checkpoint file".format(model_name))
    gdown.download(model_url, model_file, quiet=silent)  # Download file

    if "archive" in config["models"][model_name].keys():  # If necessary, unpack the archive with the checkpoint
        if not silent:
            print("Start unpacking ", model_name)
        if model_file.endswith("tar.gz"):
            tar = tarfile.open(model_file, "r:gz")
            tar.extractall(path=model_dir)
            tar.close()
            if "archive_folder" in config["models"][model_name].keys():
                os.rename(os.path.join(model_dir, config["models"][model_name]["archive_folder"]),
                          os.path.join(model_dir, "model"))
            os.remove(model_file)
            if not silent:
                print("Unpacking {} archive finished".format(model_name))


def setup(model_name, silent=False):
    """
    Downloads the required model
    :param model_name: Model name
    :param silent: Determines if silent mode will be enabled
    """
    if model_name == "all":
        for model in config["models"]:
            __setup_model__(model, silent)
    else:
        __setup_model__(model_name, silent)


def __cli__():
    """
    Console interface
    """
    parser = argparse.ArgumentParser(description="Setup tool")

    parser.add_argument('--silent', required=False, default=False,
                        help="Install the selected model in 'silent' mode", action="store_true", dest="silent")
    parser.add_argument('--model', required=False,
                        help="Model to install", action="store", dest="model", default="all")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("\033[0;32mChoose which model you want to install: \033[0m"
              "\033[1;36m\nall\n{}\033[0m".format('\n'.join(config["models"].keys())))
        print("\033[0;35mHint: Specify 'all' to install all models.\033[0m\n")
        model_name = input("\033[0;33mEnter model name: \033[0m")
        if model_name not in config["models"] and model_name != "all":
            raise ValueError("Please indicate the correct model for installation")
        setup(model_name)  # Start installation
        print("\033[0;32mInstallation finished! :)\033[0m")
    else:
        if args.model not in config["models"] and args.model != "all":
            raise ValueError("Please indicate the correct model for installation")
        if args.silent:
            setup(args.model, True)
        else:
            setup(args.model)


if __name__ == "__main__":
    __cli__()
