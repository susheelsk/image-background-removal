"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
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