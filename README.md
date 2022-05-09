# ✂️ CarveKit ✂️ 

<center> <img src="docs/imgs/logo.png"> </center>

<center>
<img src="https://github.com/OPHoperHPO/image-background-remove-tool/workflows/Test%20release%20version/badge.svg?branch=master"> <a src="https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb">
<img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>
</center>

**********************************************************************
## 📄 Description:  
Automated high-quality background removal framework for an image using neural networks.
**********************************************************************
## 🎆 Features:  
- High Quality
- Batch Processing
- NVIDIA CUDA and CPU processing
- Easy inference
- 100% remove.bg compatible FastAPI HTTP API 
- Removes background from hairs
- Easy integration with your code
**********************************************************************
## ⛱ Try yourself on [Google Colab](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb) 
**********************************************************************
## 🎓 Implemented Neural Networks:
* [U^2-net](https://github.com/NathanUA/U-2-Net)
*  [BASNet](https://github.com/NathanUA/BASNet)
* [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab) 
**********************************************************************
## 🖼️ Image pre-processing and post-processing methods:
### 🔍 Preprocessing methods:
* `None` - No preprocessing methods used.
### ✂ Post-processing methods:
* `No` - No post-processing methods used.
* `fba` (default) - This algorithm improves the borders of the image when removing the background from images with hair, etc. using FBA Matting neural network. This method gives the best result in combination with u2net without any preprocessing methods.
**********************************************************************
## 🧷 Dependencies:  
* **See** `requirements.txt`
**********************************************************************
## 🏷 Setup for CPU processing:
1. Clone this repository
2. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
3. `pip install ./`
**********************************************************************
## 🏷 Setup for GPU processing:  
1. Install `CUDA` and setup `PyTorch` for GPU processing.
2. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
3. `pip install ./`
**********************************************************************
## 🧰 Running the CLI interface:  
 * ```python3 -m carvekit  -i <input_path> -o <output_path> --device <device>```  
 
### Explanation of args:  
````
Usage: carvekit [OPTIONS]

  Performs background removal on specified photos using console interface.

Options:
  -i ./2.jpg                   Path to input file or dir  [required]
  -o ./2.png                   Path to output file or dir
  --pre none                   Preprocessing method
  --post fba                   Postprocessing method.
  --net u2net                  Segmentation Network
  --recursive                  Enables recursive search for images in a folder
  --batch_size 10              Batch Size for list of images to be loaded to
                               RAM

  --batch_size_seg 5           Batch size for list of images to be processed
                               by segmentation network

  --batch_size_mat 1           Batch size for list of images to be processed
                               by matting network

  --seg_mask_size 320          The size of the input image for the
                               segmentation neural network.

  --matting_mask_size 2048     The size of the input image for the matting
                               neural network.

  --device cpu                 Processing Device.
  --help                       Show this message and exit.

````
## 📦 Running the Framework / FastAPI HTTP API server via Docker:
Using the API via docker is a **fast** and non-complex way to have a working API.\
Docker image has default front-end at `/` url and FastAPI backend with docs at `/docs` url. \
**This HTTP API is 100% compatible with remove.bg API clients.** 
>Authentication is **enabled** by default. \
> **Token keys are reset** on every container restart if ENV variables are not set. \
See `docker-compose.<device>.yml` for more information. \
> **You can see your access keys in the docker container logs.**
> 
### 🔨 Building yourself:
1. Install `docker-compose`
2. Run `docker-compose -f docker-compose.cpu.yml up -d`  # For CPU Processing
3. Run `docker-compose -f docker-compose.cuda.yml up -d`  # For GPU Processing


### ☑️ Testing with docker
1. Run `docker-compose -f docker-compose.cpu.yml run carvekit_api pytest`  # For testing on CPU
2. Run `docker-compose -f docker-compose.cuda.yml run carvekit_api pytest`  # For testing on GPU
> You can mount folders from your host machine and use the CLI interface inside the docker container to process files in it. 
## 👪 Credits: [More info](docs/CREDITS.md)

## 📧 __Feedback__
I will be glad to receive feedback about the project and suggestions for integration.

For all questions write: [farvard34@gmail.com](mailto://farvard34@gmail.com)

## 💵 Support
  You can thank us and buy a small cup of coffee ☕
- Ethereum wallet `0x7Ab1B8015020242D2a9bC48F09b2F34b994bc2F8`

