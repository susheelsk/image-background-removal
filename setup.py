"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import os

from setuptools import setup, find_packages

from carvekit import version


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()


setup(
    name='carvekit',
    version=version,
    author="Nikita Selin (Anodev)",
    author_email='farvard34@gmail.com',
    description='Open-Source background removal framework',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    license='Apache License v2.0',
    keywords=[],
    url='https://github.com/OPHoperHPO/image-background-remove-tool',
    packages=find_packages(),
    scripts=[],
    install_requires=[
        'requests~=2.27.1',
        'torch~=1.11.0',
        'Pillow==9.0.1',
        'typing~=3.7.4.3',
        'torchvision~=0.12.0',
        'opencv-python~=4.5.5.64',
        'numpy~=1.22.4',
        'loguru~=0.6.0',
        'uvicorn~=0.17.6',
        'fastapi~=0.78.0',
        'starlette~=0.19.1',
        'pydantic~=1.9.1',
        'click~=8.1.3',
        'tqdm~=4.64.0',
        'setuptools~=62.3.2',
        'aiofiles~=0.8.0',
        'python-multipart~=0.0.5',
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
