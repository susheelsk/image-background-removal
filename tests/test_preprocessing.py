"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""


def test_seg(preprocessing_stub_instance, image_str, image_path, image_pil, interface_instance):
    preprocessing_stub_instance = preprocessing_stub_instance()
    interface_instance = interface_instance()
    preprocessing_stub_instance(interface_instance, [image_str, image_path])
    preprocessing_stub_instance(interface_instance, [image_pil, image_path])
