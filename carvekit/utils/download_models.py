"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import hashlib
import os
import warnings
from pathlib import Path

import requests
import tqdm

MODELS_URLS = {
    "basnet.pth":
        "https://huggingface.co/anodev/basnet-universal/resolve/870becbdb364fda6d8fdb2c10b072542f8d08701/basnet.pth",
    "deeplab.pth":
        "https://huggingface.co/anodev/deeplabv3-resnet101/resolve/d504005392fc877565afdf58aad0cd524682d2b0/deeplab.pth",
    "fba_matting.pth":
        "https://huggingface.co/anodev/fba/resolve/a5d3457df0fb9c88ea19ed700d409756ca2069d1/fba_matting.pth",
    "u2net.pth":
        "https://huggingface.co/anodev/u2net-universal/resolve/10305d785481cf4b2eee1d447c39cd6e5f43d74b/full_weights"
        ".pth",
}

MODELS_CHECKSUMS = {
    "basnet.pth": "e409cb709f4abca87cb11bd44a9ad3f909044a917977ab65244b4c94dd33"
                  "8b1a37755c4253d7cb54526b7763622a094d7b676d34b5e6886689256754e5a5e6ad",
    "deeplab.pth":
        "9c5a1795bc8baa267200a44b49ac544a1ba2687d210f63777e4bd715387324469a59b072f8a28"
        "9cc471c637b367932177e5b312e8ea6351c1763d9ff44b4857c",
    "fba_matting.pth":
        "890906ec94c1bfd2ad08707a63e4ccb0955d7f5d25e32853950c24c78"
        "4cbad2e59be277999defc3754905d0f15aa75702cdead3cfe669ff72f08811c52971613",
    "u2net.pth":
        "16f8125e2fedd8c85db0e001ee15338b4aa2fda77bab8ba70c25e"
        "bea1533fda5ee70a909b934a9bd495b432cef89d629f00a07858a517742476fa8b346de24f7",

}


def download_model(path: Path) -> Path:
    """ Downloads model from repo.

    Args:
        path (pathlib.Path): Path to file

    Returns:
         Path if exists

    Raises:
        FileNotFoundError: if model checkpoint is not exists in known checkpoints models
        ConnectionError: if the model cannot be loaded from the URL.
    """
    if path.name in MODELS_URLS:
        model_url = MODELS_URLS[path.name]
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(model_url, stream=True)
            if r.status_code == 200:
                with path.absolute().open('wb') as f:
                    r.raw.decode_content = True
                    for chunk in tqdm.tqdm(r, desc="Downloading " + path.name + ' model', colour='blue'):
                        f.write(chunk)
        except BaseException as e:
            if path.exists():
                os.remove(path)
            raise ConnectionError(f"Exception caused when downloading model! "
                                  f"Model name: {path.name}. Exception: {str(e)}")
        return path
    else:
        raise FileNotFoundError("Unknown model!")


def sha512_checksum_calc(file: Path) -> str:
    """
    Calculates the SHA512 hash digest of a file on fs

    Args:
        file: Path to the file

    Returns:
        SHA512 hash digest of a file.
    """
    dd = hashlib.sha512()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            dd.update(chunk)
    return dd.hexdigest()


def check_model(path: Path) -> bool:
    """ Verifies model checksums and existence in the file system

    Args:
        path: Path to the model

    Returns:
        True if all is okay and False if not

    Raises:
        FileNotFoundError: if model checkpoint is not exists in known checkpoints models
    """
    if path.exists():
        if path.name in MODELS_URLS:
            if MODELS_CHECKSUMS[path.name] != sha512_checksum_calc(path):
                warnings.warn(f"Invalid checksum for model {path.name}. Downloading correct model!")
                os.remove(path)
                return False
            return True
        else:
            raise FileNotFoundError("Unknown model!")
    else:
        return False


def check_for_exists(path: Path) -> Path:
    """ Checks for checkpoint path exists

    Args:
        path (pathlib.Path): Path to file

    Returns:
         Path if exists

    Raises:
        FileNotFoundError: if model checkpoint is not exists in known checkpoints models
        ConnectionError: if the model cannot be loaded from the URL.
    """
    if not check_model(path):
        download_model(path)

    return path
