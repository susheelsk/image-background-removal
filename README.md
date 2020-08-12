# 🥧 Image Background Remove Tool 🥧 ![Test release version](https://github.com/OPHoperHPO/image-background-remove-tool/workflows/Test%20release%20version/badge.svg?branch=master) [![](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/try.ipynb)
Tool for removing background from image using neural networks 
**********************************************************************
### 📄 Description:  
The program removes the background from photos  
**********************************************************************
### 🎆 Features:  
* **Added support for new neural networks ([U^2-NET](https://github.com/NathanUA/U-2-Net), [BASNet]((https://github.com/NathanUA/BASNet)) on PyTorch**
* **Significantly improved output image quality**
* **Removes background from hair**
* **Added flask http api, fully compatible with `remove.bg` api libraries. Just change the `endpoint url` and voila!**
* **Added GUI by [@Munawwar](https://github.com/Munawwar)** 
* __Tensorflow 2.0 compatible__  
* __All models support processing both on the video card and on the processor__  
* __```tqdm``` progress bar__
* __The program has a lot of methods for image preprocessing and post-processing, which allows you to configure the quality and speed of image processing for your needs__
* __Removes background from image without loss of image resolution__  
*  __The script not only processes a single file, but can also process all images from the input folder and save them in the output folder with the same name__  
*  __Implemented support for the neural network from this [ script](https://github.com/susheelsk/image-background-removal) and improved the result of its work__  
**********************************************************************
### ⛱ Try this program yourself on [Google Colab](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/try.ipynb) 
**********************************************************************
 ### 🎓 Implemented Neural Networks:
* [U^2-net](https://github.com/NathanUA/U-2-Net)
*  [BASNet](https://github.com/NathanUA/BASNet)
* [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab)
 > [More info about models.](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/MODELS.md)  
**********************************************************************
 ### 🖼️ Image pre-processing and post-processing methods:
 #### 🔍 Preprocessing methods:
* `None` (**default**) - No preprocessing methods used.
* `bbd-fastrcnn` - This image pre-processing technique uses two neural networks ($used_model and Fast RCNN) to first detect the boundaries of objects in a photograph, cut them out, sequentially remove the background from each object in turn and subsequently collect the entire image from separate parts.
* `bbmd-maskrcnn` - This image pre-processing technique uses two neural networks ($used_model and Mask RCNN) to first detect the boundaries and masks of objects in a photograph, cut them out, expand the masks by a certain number of pixels, apply them and remove the background from each object in turn and subsequently collect the entire image from separate parts. **So far it works very poorly!**

#### ✂ Post-processing methods:
* `No` - No post-processing methods used.
* `fba` (**default**) - This algorithm improves the borders of the image when removing the background from images with hair, etc. using [FBA Matting](https://github.com/MarcoForte/FBA_Matting) neural network.
* `rtb-bnb` - This algorithm improves the boundaries of the image obtained from the neural network. It is based on the principle of removing too transparent pixels and smoothing the borders after removing too transparent pixels.
* `rtb-bnb2` - This algorithm improves the boundaries of the image obtained from the neural network. It is based on the principle of removing too transparent pixels and smoothing the borders after removing too transparent pixels. The algorithm performs this procedure twice. For the first time, the algorithm processes the image from the neural network, then sends the processed image back to the neural network, and then processes it again and returns it to the user. This method gives the best result in combination with u2net without any preprocessing methods.
**********************************************************************
### 🧷 Dependencies:  
* **See** `requirements.txt`
* **See** `requirements_http.txt`, if you need http api.
> Note:  You can choose what to install PyTorch or TensorFlow, based on which model you want to use. \
PyTorch for `u2net`, `u2netp`  \
TensorFlow for `xception_model`, `mobile_net_model`  \
Mxnet and Gluoncv are used for image preprocessing methods and are installed optionally. \
**Also, to speed up image processing by performing all the calculations on the video card, install separately special versions of the dependencies (`tensorflow, torch, mxnet, gluoncv and others`) designed to work with your video card.** \
**TensorFlow models are not recommended for use, since these models have much worse quality and lower image processing speed, also these models are designed solely to remove the background from portrait photos and photos with animals.**
**********************************************************************
### 🏷 Setup for Windows:  
* Clone this repository  
* Install all the dependencies from **requirements.txt** via ```pip3 install -r requirements.txt```  
* Run ```./setup.bat```
> _This setup.bat script loads the trained model._  \
> The program was tested on python version 3.7.3
**********************************************************************
### 🏷 Setup for Linux:  
* Clone repository: ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```  
* Install all the dependencies from **requirements.txt**: ```pip3 install -r requirements.txt```  
* Run ```./setup.sh``` and select the model you need.
> _This setup.sh script loads the pre-trained model._ \
> The program was tested on python version 3.7.3
**********************************************************************
### 🖼️ GUI screenshot:
![](/docs/imgs/screenshots/gui.png)
**********************************************************************
### 🖵 Running the GUI app:
```python3 gui.py```
**********************************************************************
### 📦 Running the HTTP API server:

#### 🧲 With defaults:
```python3 http_api.py```

#### 🧲 With custom arguments:

```python3 http_api.py -auth false -port 5000 -host 0.0.0.0 -m u2net```


#### ⏩ Example usage with curl:
```bash
curl -H 'X-API-Key: test'                                   \
       -F 'image_file=@/home/user/test.jpg'                 \
       -F 'size=auto'                                       \ # oneOf 'preview', 'medium', 'hd', 'auto'
       -f http://localhost:5000/api/removebg -o no-bg.png
```

### 📦 Running the HTTP API server via docker:
Using the API via docker is a **fast** and non-complex way to have a working API.  \
The docker image uses `u2net` as default and runs without authentication.
#### 🔨 Building your own image:
* Build the docker image
```bash
docker build --tag image-background-removal-tool:latest .
```
* Start a container from the image
```bash
docker run -d --restart unless-stopped \
 --name image-background-removal-tool-http-api \
 -p 5000:5000 \
 -e HOST=0.0.0.0   \
 -e PORT=5000  \
 -e AUTH=false  \
 -e MODEL=u2net  \
 -e PREPROCESSING=None  \
 -e POSTPROCESSING=fba  \
 -e ADMIN_TOKEN=admin  \
 -e ALLOWED_TOKENS_PYTHON_ARR=["test"]  \
 -e IS_DOCKER_CONTAINER=true  \
image-background-removal-tool:latest
```


**********************************************************************
### 🧰 Running the script:  
 * ```python3 main.py -i <input_path> -o <output_path> -m <model_type> -prep <preprocessing_method> -postp <postprocessing_method>```  
 
#### Explanation of args:  
* `-i <input_path>` - path to input file or dir.
* `-o <output_path>` - path to output file or dir.
* `-prep <preprocessing_method>` - Preprocessing method. Can be `bbd-fastrcnn` or `bbmd-maskrcnn` or `None`. `None` is better to use.
* `-postp <postprocessing_method>` - Postprocessing method. Can be `fba` or `rtb-bnb` or `rtb-bnb2` or `No`. `fba` is better to use.
* `-m <model_type>` - can be `u2net` or `basnet` or `u2netp` or `xception_model` or `mobile_net_model`. `u2net` is better to use. \
**DeepLab** models (`xception_model` or `mobile_net_model`) are **outdated** 
and designed to remove the background from **PORTRAIT** photos or **PHOTOS WITH ANIMALS!** \
[More info about models.](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/MODELS.md)  
 > Note:  See example scripts for more information on using the program.  
**********************************************************************
### ⏳ TODO:  
```
1) Check TODOs in code.
2) Implement support for Mask RCNN. (90% done)
3) Add an algorithm for automatic color correction at image borders. (0% done)
```
**********************************************************************
### 👪 Credits: [More info](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/CREDITS.md) 
**********************************************************************
### 💵 Support me:  
  You can thank me for developing any of my projects, provide financial support for developing new projects and buy me a small cup of coffee.☕ \
  Just support me on these platforms:
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/boosty_logo.jpeg) [**Boosty**](https://boosty.to/anodev)
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/donationalerts_logo.png) [**DonationAlerts**](https://www.donationalerts.com/r/anodev_development)
  * ![](https://github.com/OPHoperHPO/OPHoperHPO/raw/master/assets/imgs/paypal_logo.jpg) [**PayPal**](https://paypal.me/anodev)
**********************************************************************
### 🖼️ Sample Result:
* **More sample images in [docs/imgs/input/](docs/imgs/input) and [docs/imgs/examples/](docs/imgs/examples) folders.**  \
Examples of images from the background are contained in folders in the following format: `{model_name}/{preprocessing_method_name}/{postprocessing_method_name}`
* Input:   
* ![Input](/docs/imgs/input/4.jpg)
* Output(u2net/None/fba):
* ![Output](/docs/imgs/examples/u2net/None/fba/4.png "Output")
*  Output(basnet/None/fba):
* ![Output](/docs/imgs/examples/basnet/None/fba/4.png "Output")
* Output(u2netp/None/fba):
* ![Output](/docs/imgs/examples/u2netp/None/fba/4.png "Output")
* Output(xception_model/None/fba):
* ![Output](/docs/imgs/examples/xception_model/None/fba/4.png "Output")
* Output(mobile_net_model/None/fba):
* ![Output](/docs/imgs/examples/mobile_net_model/None/fba/4.png "Output")
**********************************************************************
