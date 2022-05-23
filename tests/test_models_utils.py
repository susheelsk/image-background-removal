"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import os
import pytest
from pathlib import Path
from carvekit.utils.download_models import check_for_exists, check_model, sha512_checksum_calc, download_model
from carvekit.ml.files.models_loc import u2net_full_pretrained, fba_pretrained, deeplab_pretrained, basnet_pretrained, \
    download_all
from carvekit.utils.models_utils import fix_seed, suppress_warnings


def test_fix_seed():
    fix_seed(seed=42)


def test_suppress_warnings():
    suppress_warnings()


def test_download_all():
    download_all()


def test_download_model():
    hh = Path(__file__).parent.joinpath('data', 'u2net.pth')
    hh.write_text('1234')
    assert download_model(hh) == hh
    os.remove(hh)
    with pytest.raises(FileNotFoundError):
        download_model(Path("NotExistedPath/2.dl"))
    with pytest.raises(FileNotFoundError):
        download_model(Path(__file__).parent.joinpath('data', 'cat.jpg'))


def test_sha512():
    hh = Path(__file__).parent.joinpath('data', 'basnet.pth')
    hh.write_text('1234')
    assert sha512_checksum_calc(hh) == "d404559f602eab6fd602ac7680dacbfaadd13630335e951f097a" \
                                       "f3900e9de176b6db28512f2e000" \
                                       "b9d04fba5133e8b1c6e8df59db3a8ab9d60be4b97cc9e81db"


def test_check_model():
    invalid_hash_file = Path(__file__).parent.joinpath('data', 'basnet.pth')
    invalid_hash_file.write_text('1234')
    assert check_model(invalid_hash_file) is False
    assert check_model(Path(__file__).parent.joinpath('data', 'u2net.pth')) is False
    assert check_model(u2net_full_pretrained()) is True
    assert check_model(Path("NotExistedPath/2.dl")) is False
    with pytest.raises(FileNotFoundError):
        assert check_model(Path(__file__).parent.joinpath('data', 'cat.jpg')) is False


def test_check_for_exists():
    assert isinstance(check_for_exists(u2net_full_pretrained()), Path) is True
    assert isinstance(check_for_exists(fba_pretrained()), Path) is True
    assert isinstance(check_for_exists(deeplab_pretrained()), Path) is True
    assert isinstance(check_for_exists(basnet_pretrained()), Path) is True
    with pytest.raises(FileNotFoundError):
        check_for_exists(Path(__file__).parent.joinpath('data', 'cat.jpg'))
