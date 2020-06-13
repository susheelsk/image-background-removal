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
from libs import networks
import unittest


class ModelsCase(unittest.TestCase):
    def test_models(self):
        for model_name in networks.strings.MODELS_NAMES:
            try:
                model = networks.model_detect(model_name)
            except FileNotFoundError:
                model = None
            if model is False:
                self.fail("Unknown model in MODELS_NAME !!!")


if __name__ == '__main__':
    unittest.main()
