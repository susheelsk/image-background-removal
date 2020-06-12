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
from main import __work_mode__
import unittest


class WmodeTest(unittest.TestCase):
    def test_wmode(self):
        self.assertEqual(__work_mode__("docs"), "dir")
        self.assertEqual(__work_mode__("docs/"), "dir")
        self.assertEqual(__work_mode__("docs/CREDITS.md"), "file")
        self.assertEqual(__work_mode__("docs/CREDITS.md/"), "no")


if __name__ == '__main__':
    unittest.main()
