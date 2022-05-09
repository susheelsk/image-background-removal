"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
# Install this library before using this example!
# https://github.com/OPHoperHPO/remove-bg-api
import remove_bg_api
from pathlib import Path

remove_bg_api.API_URL = "http://localhost:5000/api"  # Change the endpoint url
removebg = remove_bg_api.RemoveBg("test")

settings = \
    {  # API settings. See https://www.remove.bg/api for more details.
        "size": "preview",  # ["preview", "full", "auto", "medium", "hd", "4k", "small", "regular"]
        "type": "auto",  # ["auto", "person", "product", "car"]
        "format": "auto",  # ["auto", "png", "jpg", "zip"]
        "roi": "",  # {}% {}% {}% {}% or {}px {}px {}px {}px
        "crop": False,  # True or False
        "crop_margin": "0px",  # {}% or {}px
        "scale": "original",  # "{}%" or "original"
        "position": "original",  # "original" "center", or {}%
        "channels": "rgba",  # "rgba" or "alpha"
        "add_shadow": "false",  # Not supported at the moment
        "semitransparency": "false",  # Not supported at the moment
        "bg_color": "",  # "81d4fa" or "red" or any other color
        "bg_image_url": ""  # URL
    }

removebg.remove_bg_file(str(Path("images/4.jpg").absolute()), raw=False,
                        out_path=str(Path("./4.png").absolute()), data=settings)