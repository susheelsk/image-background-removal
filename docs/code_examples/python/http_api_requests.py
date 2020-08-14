"""
Name: Example for interacting with Flask Api
Description: This file contains an example of interacting with this tool via an HTTP request.
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
# Requires "requests" to be installed
import requests
from pathlib import Path

response = requests.post(
    'http://localhost:5000/api/removebg',
    files={'image_file': Path("images/4.jpg").read_bytes()},
    data={'size': 'auto'},
    headers={'X-Api-Key': 'test'},
)
if response.status_code == 200:
    Path("image_without_bg.png").write_bytes(response.content)
else:
    print("Error:", response.status_code, response.text)