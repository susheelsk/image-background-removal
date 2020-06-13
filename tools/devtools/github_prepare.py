"""
Name: github prepare tool
Description: This file contains the github prepare tool code.
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
import os
import subprocess


def main():
    # # Run Tests and create examples
    subprocess.check_call("cd ../../tests && ./run_tests.sh", shell=True)

    # Remove dependencies, create new ones
    os.remove("../../requirements.txt")
    subprocess.check_call("cd ../../ && pipreqs .", shell=True)
    with open("../../requirements.txt", "r") as f:
        data = f.read()
    data = data.split("\n")
    data.remove("skimage==0.0")
    with open("../../requirements.txt", "w") as f:
        f.write('\n'.join(data))
    if os.path.exists("../../tests/requirements.txt"):
        os.remove("../../tests/requirements.txt")



if __name__ == '__main__':
    main()
