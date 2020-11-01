# 🥧 Image Background Remove Tool 🥧 ![Test](https://github.com/OPHoperHPO/image-background-remove-tool/workflows/Test/badge.svg?branch=master) [![](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/try.ipynb)
![Input](/docs/imgs/compare/readme.jpg)
> The higher resolution images from the picture above can be seen in the docs/imgs/compare/ folder.
**********************************************************************
## 📄 Description:  
The program removes the background from photos using neural networks.  

**********************************************************************
## 🎆 Features:  
* **GUI**
* **Removes background from hair**
* **Significantly improved output image quality**
* **Removes background from image without loss of image resolution**
* **All models support processing both on the video card and on the processor**
* **Added support for new neural networks ([U^2-NET](https://github.com/NathanUA/U-2-Net), [BASNet]((https://github.com/NathanUA/BASNet))) on PyTorch**
* **Updated DeepLabv3 core and moved to PyTorch implementation with ResNet 101 backbone.**
* **The program has a lot of methods for image preprocessing and post-processing, which allows you to configure the quality and speed of image processing for your needs**
* **Added flask http api, fully compatible with `remove.bg` api libraries. Just change the `endpoint url` and voila!** 

**********************************************************************
## ⛱ Try this program yourself on [Google Colab](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/try.ipynb) 

**********************************************************************
## 🎓 Implemented Neural Networks:
* [U^2-net](https://github.com/NathanUA/U-2-Net)
* [BASNet](https://github.com/NathanUA/BASNet)
* [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
> [More info about models.](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/MODELS.md)  

**********************************************************************
## 🖼️ Image pre-processing and post-processing methods:
### 🔍 Preprocessing methods:
* `None` (**default**) - No preprocessing methods used.
* `bbd-fastrcnn` - This image pre-processing technique uses two neural networks ($used_model and Fast RCNN) to first detect the boundaries of objects in a photograph, cut them out, sequentially remove the background from each object in turn and subsequently collect the entire image from separate parts.
* `bbmd-maskrcnn` - This image pre-processing technique uses two neural networks ($used_model and Mask RCNN) to first detect the boundaries and masks of objects in a photograph, cut them out, expand the masks by a certain number of pixels, apply them and remove the background from each object in turn and subsequently collect the entire image from separate parts. **So far it works very poorly!**

### ✂ Post-processing methods:
* `No` - No post-processing methods used.
* `fba` (**default**) - This algorithm improves the borders of the image when removing the background from images with hair, etc. using [FBA Matting](https://github.com/MarcoForte/FBA_Matting) neural network. This method gives the best result in combination with u2net without any preprocessing methods.
* `rtb-bnb` - This algorithm improves the boundaries of the image obtained from the neural network. It is based on the principle of removing too transparent pixels and smoothing the borders after removing too transparent pixels.
* `rtb-bnb2` - This algorithm improves the boundaries of the image obtained from the neural network. It is based on the principle of removing too transparent pixels and smoothing the borders after removing too transparent pixels. The algorithm performs this procedure twice. For the first time, the algorithm processes the image from the neural network, then sends the processed image back to the neural network, and then processes it again and returns it to the user. 
**********************************************************************
## 🧷 Dependencies:  
* **See** `requirements.txt`
* **See** `requirements_http.txt`, if you need http api.
> Note: `mxnet` and `gluoncv` are used for image preprocessing methods and are installed optionally. \
**Also, to speed up image processing by performing all the calculations on the video card, install separately special versions of the dependencies (`torch, mxnet, gluoncv and others`) designed to work with your video card.** 
**********************************************************************
## 🏷 Setup for Windows:  
* Clone this repository  
* Install numpy ```pip3 install numpy```
* Install all the dependencies from **requirements.txt** via ```pip3 install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt```  
* Run ```python3 setup.py```
> _This setup.bat script loads the trained model._  \
> The install script also supports installing models using arguments. For more information, run `python3 setup.py --help`.\
> The program was tested on python version 3.7.3
**********************************************************************
## 🏷 Setup for Linux:  
* Clone repository: ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Install numpy ```pip3 install numpy```  
* Install all the dependencies from **requirements.txt**: ```pip3 install -r requirements.txt```  
* Run ```python3 setup.py``` and select the model you need.
> _This setup.py script loads the pre-trained model._ \
> The install script also supports installing models using arguments. For more information, run `python3 setup.py --help`.\
> The program was tested on python version 3.7.3
**********************************************************************
## 🖼️ GUI screenshot:
![](/docs/imgs/screenshots/gui.png)
**********************************************************************
## 🖵 Running the GUI app:
```python3 gui.py```
**********************************************************************
## 📦 Running the HTTP API server:
### 🧲 With defaults:
```python3 http_api.py```

### 🧲 With custom arguments:
```python3 http_api.py -auth false -port 5000 -host 0.0.0.0 -m u2net -pre None -post fba```

### ⏩ Example usage with curl:
```bash
curl -H 'X-API-Key: test'                                   \
       -F 'image_file=@/home/user/test.jpg'                 \
       -F 'size=auto'                                       \ # oneOf 'preview', 'medium', 'hd', 'auto'
       -f http://localhost:5000/api/removebg -o no-bg.png
```
> Note:  See example scripts in docs/code_examples/python for more information on using the http api.  
## 📦 Running the HTTP API server via docker:
Using the API via docker is a **fast** and non-complex way to have a working API.  \
The docker image uses `u2net` as default and runs without authentication.
### 💻 Using an already built image from DockerHub: 
```bash
docker run -d --restart unless-stopped \
 --name image-background-remove-tool \
 -p 5000:5000 \
 -e HOST='0.0.0.0'   \
 -e PORT='5000'  \
 -e AUTH='false'  \
 -e MODEL='u2net'  \
 -e PREPROCESSING='None'  \
 -e POSTPROCESSING='fba'  \
 -e ADMIN_TOKEN='admin'  \
 -e ALLOWED_TOKENS_PYTHON_ARR='["test"]'  \
 -e IS_DOCKER_CONTAINER='true'  \
docker.io/anodev/image-background-remove-tool:release 
```
### 🔨 Building your own image:
* Build the docker image
```bash
docker build --tag image-background-remove-tool:latest .
```
* Start a container from the image
```bash
docker run -d --restart unless-stopped \
 --name image-background-remove-tool \
 -p 5000:5000 \
 -e HOST='0.0.0.0'   \
 -e PORT='5000'  \
 -e AUTH='false'  \
 -e MODEL='u2net'  \
 -e PREPROCESSING='None'  \
 -e POSTPROCESSING='fba'  \
 -e ADMIN_TOKEN='admin'  \
 -e ALLOWED_TOKENS_PYTHON_ARR='["test"]'  \
 -e IS_DOCKER_CONTAINER='true'  \
image-background-remove-tool:latest
```


**********************************************************************
## 🧰 Running the script:  
 * ```python3 main.py -i <input_path> -o <output_path> -m <model_type> -pre <preprocessing_method> -post <postprocessing_method> --recursive```  
 
### Explanation of args:  
* `-i <input_path>` - Path to input file or dir.
* `-o <output_path>` - Path to output file or dir.
* `-pre <preprocessing_method>` - Preprocessing method. Can be `bbd-fastrcnn` or `bbmd-maskrcnn` or `None`. `None` is better to use.
* `-post <postprocessing_method>` - Postprocessing method. Can be `fba` or `rtb-bnb` or `rtb-bnb2` or `No`. `fba` is better to use.
* `-m <model_type>` - Can be `u2net` or `basnet` or `u2netp` or `deeplabv3`. `u2net` is better to use. 
* `--recursive`  - Enables recursive search for images in a folder \
**DeepLabV3** model designed to remove the background from **PORTRAIT** photos or **PHOTOS WITH ANIMALS!** \
[More info about models.](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/MODELS.md)  
> Note:  See example scripts in docs/code_examples/shell for more information on using the program.  
**********************************************************************

## ⏳ TODO:  
```
1) Check TODOs in code.
2) Implement support for Mask RCNN. (90% done)
3) Add an algorithm for automatic color correction at image borders. (0% done)
```

**********************************************************************
## 👪 Credits: [More info](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/CREDITS.md) 

**********************************************************************
## 💵 Support me:  
  You can thank me for developing any of my projects, provide financial support for developing new projects and buy me a small cup of coffee.☕ \
  Just support me on these platforms:
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/boosty_logo.jpeg) [**Boosty**](https://boosty.to/anodev)
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/donationalerts_logo.png) [**DonationAlerts**](https://www.donationalerts.com/r/anodev_development)
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/paypal_logo.jpg) [**PayPal**](https://paypal.me/anodevru)
**********************************************************************
## 🖼️ Sample Result:
* **More sample images in [docs/imgs/input/](docs/imgs/input) and [docs/imgs/examples/](docs/imgs/examples) folders.**  \
Examples of images from the background are contained in folders in the following format: `{model_name}/{preprocessing_method_name}/{postprocessing_method_name}`
* Input:   
* ![Input](/docs/imgs/input/1.jpg)
* Output(u2net/None/fba):
* ![Output](/docs/imgs/examples/u2net/None/fba/1.png "Output")
* Output(deeplabv3/None/fba):
* ![Output](/docs/imgs/examples/deeplabv3/None/fba/1.png "Output")
*  Output(basnet/None/fba):
* ![Output](/docs/imgs/examples/basnet/None/fba/1.png "Output")
* Output(u2netp/None/fba):
* ![Output](/docs/imgs/examples/u2netp/None/fba/1.png "Output")
**********************************************************************

