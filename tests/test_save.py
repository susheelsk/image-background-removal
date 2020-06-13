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
from main import __save_image_file__
import os
import shutil
import unittest
import random
from PIL import Image


def new_name():
    filename = str(random.randint(0, 1202)) + ".jpg"
    return filename


def save():
    path = "tests/tests_temp/save_test/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    __save_image_file__(Image.new("RGBA", (256, 256), color=0), new_name(), path, "dir")  # Dir mode
    __save_image_file__(Image.new("RGBA", (256, 256), color=0), new_name(), path, "file")  # File name empty base name
    a = None
    f = new_name()
    try:
        __save_image_file__(Image.new("RGBA", (256, 256), color=0), f, path + f, "file")  # Extension Exception
    except OSError:
        a = True
    if a:
        a = False
        try:
            __save_image_file__(Image.new("RGBA", (256, 256), color=0), f, path + f, "dir")  # Not dir error
        except OSError as e:
            a = True
        if a:
            __save_image_file__(Image.new("RGBA", (256, 256), color=0), f, path + f + '.png',
                                "file")  # filename png test
        else:
            return False
    else:
        return False
    shutil.rmtree(path)
    return True


class SaveTest(unittest.TestCase):
    def test_save(self):
        self.assertEqual(save(), True)


if __name__ == '__main__':
    unittest.main()
