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

import subprocess
import sys
from pathlib import Path
import multiprocessing
import unittest
from libs import strings
from main import process


def run(test, i, o, m, prep, postp):
    try:
        process(i, o, m, prep, postp)
    except BaseException as e:
        test.fail("TESTING FAILED!\n"
                  "PARAMS:\n"
                  "model_name: {}\n"
                  "input_path: {}\n"
                  "output_path: {}\n"
                  "preprocessing_method: {}\n"
                  "postprocessing_method: {}\n"
                  "Error: {}\n".format(m, i, o, prep, postp, str(e)))
        exit(1)
    exit(0)


def cli_call(input, out, model, prep="bla", post="bla", python="python3"):
    sub = subprocess.Popen("{} main.py -i {} -o {} -m {} -pre {} -post {}".format(python, input, out, model, prep,
                                                                                  post), shell=True,
                           stdout=subprocess.PIPE)
    return sub


def gen(test):
    input_path = Path("docs/imgs/input/")
    for model_name in strings.MODELS_NAMES:
        for preprocess_method_name in strings.PREPROCESS_METHODS:
            for postprocess_method_name in strings.POSTPROCESS_METHODS:
                print(model_name, preprocess_method_name, postprocess_method_name)
                path = Path("docs/imgs/examples/{}/{}/{}".format(model_name,
                                                                 preprocess_method_name,
                                                                 postprocess_method_name))
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                if sys.platform == "linux":
                    try:
                        proc = multiprocessing.Process(target=run,
                                                       args=(test, str(input_path.absolute()), str(path.absolute()),
                                                             model_name, preprocess_method_name,
                                                             postprocess_method_name,))
                        proc.start()
                        proc.join()
                        if proc.exitcode == 1:
                            return False
                    except BaseException as e:
                        print(e)
                        raise e
                if sys.platform == "win32":
                    try:
                        sub = cli_call(str(input_path.absolute()), str(path.absolute()), model_name,
                                       preprocess_method_name, postprocess_method_name, python=sys.executable)
                        test.fail(sub.communicate()[0].decode("UTF-8"))
                        if sub.returncode == 1:
                            test.fail("TESTING FAILED!\n"
                                      "PARAMS:\n"
                                      "model_name: {}\n"
                                      "input_path: {}\n"
                                      "output_path: {}\n"
                                      "preprocessing_method: {}\n"
                                      "postprocessing_method: {}\n"
                                      "Error: {}\n".format(model_name, str(input_path.absolute()),
                                                           str(path.absolute()), preprocess_method_name,
                                                           postprocess_method_name,
                                                           str(sub.communicate()[0].decode("UTF-8"))))
                            return False
                    except BaseException as e:
                        print(e)
                        raise e
                if sys.platform == "darwin":
                    try:
                        sub = cli_call(str(input_path.absolute()), str(path.absolute()), model_name,
                                       preprocess_method_name, postprocess_method_name)
                        if sub.returncode == 1:
                            test.fail("TESTING FAILED!\n"
                                      "PARAMS:\n"
                                      "model_name: {}\n"
                                      "input_path: {}\n"
                                      "output_path: {}\n"
                                      "preprocessing_method: {}\n"
                                      "postprocessing_method: {}\n"
                                      "Error: {}\n".format(model_name, str(input_path.absolute()),
                                                           str(path.absolute()), preprocess_method_name,
                                                           postprocess_method_name,
                                                           str(sub.communicate()[0].decode("UTF-8"))))
                            return False
                    except BaseException as e:
                        print(e)
                        raise e
    return True


class GenTest(unittest.TestCase):
    def test_generator(self):
        self.assertEqual(gen(self), True)


if __name__ == '__main__':
    unittest.main()
