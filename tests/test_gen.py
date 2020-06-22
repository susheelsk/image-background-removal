"""
Name: tests
Description: This file contains the test code
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
import unittest
from libs import strings
from main import process
import os


def gen():
    for model_name in strings.MODELS_NAMES:
        for preprocess_method_name in strings.PREPROCESS_METHODS:
            for postprocess_method_name in strings.POSTPROCESS_METHODS:
                if not os.path.exists("docs/imgs/examples/{}/{}/{}".format(model_name,
                                                                           preprocess_method_name, postprocess_method_name)):
                    os.makedirs("docs/imgs/examples/{}/{}/{}".format(model_name,
                                                                     preprocess_method_name, postprocess_method_name),
                                exist_ok=True)
                print(model_name, preprocess_method_name, postprocess_method_name)
                process("docs/imgs/input/", "docs/imgs/examples/{}/{}/{}".format(model_name,
                                                                     preprocess_method_name, postprocess_method_name),
                        model_name, preprocess_method_name, postprocess_method_name)
    return True


class GenTest(unittest.TestCase):
    def test_generator(self):
        self.assertEqual(gen(), True)


if __name__ == '__main__':
    unittest.main()
