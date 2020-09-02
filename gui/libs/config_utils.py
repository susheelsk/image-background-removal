"""
Name: Config utilities file
Description: This file contains configuration utilities for the QT GUI.
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
import os
from pathlib import Path
import sys
import json


class Config:
    """Config class"""

    def __init__(self, path="config.json"):
        try:
            self.path = Path(path)
            __config_file__ = self.path.open("r")
            self.c = json.load(__config_file__)
            __config_file__.close()
        except FileNotFoundError as e:
            print("Config file not found!")
            raise e

    def save(self):
        """Saves config to file"""
        try:
            __config_file__ = self.path.open("w")
            __config_file__.write(json.dumps(self.c,
                                             ensure_ascii=False,
                                             indent=2))
            __config_file__.close()
        except OSError:
            raise Exception("Config write error!")


class Utils:
    """Utils object"""

    @staticmethod
    def restart():
        """Restarts the current program."""
        python = sys.executable
        os.execl(python, python, *sys.argv)


def param2text(config_param):
    """Converts a parameter from config.json to text"""
    if config_param in config_texts:
        return config_texts[config_param]
    return config_param


config_texts = {
    "['tool']['model']": "Tool: Segmentation Model",
    "['tool']['preprocessing_method']": "Tool: Image Pre-processing Method",
    "['tool']['postprocessing_method']": "Tool: Image Post-processing Method",
    "['tool']['use_gpu']": "Tool: Use a GPU to speed up processing?"
}
