# 🥧 Image Background Remove Tool 🥧
A tool to remove a background from a portrait image using Tensorflow 
**********************************************************************
### 📄 Description:
The program removes the background from portrait photos
**********************************************************************
### 🎆 Differences from the [original script](https://github.com/susheelsk/image-background-removal):
* __Tensorflow 2.0 compatible__
* Added comments to the code.
* Added ```tqdm``` progress bar.
* __Removes background from image without loss of image resolution.__
*  __The script now not only processes a single file, but can also process all images from the input folder and save them in the output folder with the same name.__
* __New sample images.__
**********************************************************************
### 🧷 Dependencies:
```	wget ``` **for setup.py!** \
```	tensorflow, pillow, tqdm, numpy, scipy ``` **for main.py!**
**********************************************************************
### 🏷 Setup for Windows:
* Clone this repository
* Install all the dependencies from **requirements.txt** via ```pip3 install -r requirements.txt```
* Run ```./setup.bat``` \
_This setup.bat script loads the trained model._
### 🏷 Setup for Linux:
* Clone repository: ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Install all the dependencies from **requirements.txt**: ```pip3 install -r requirements.txt```
* Run ```./setup.sh``` \
_This setup.sh script loads the trained model._
**********************************************************************
### 🧰 Running the script:
 * ```python3 main.py <input_path> <output_path> <model_type>```
#### Explanation of variables:
 * `<input_path>` - path to input file or dir.
 * `<output_path>` - path to output file or dir.
 * ```<model_type>``` - can be ``` xception_model``` or ``` mobile_net_model```.
The first model has better quality, but it runs much slower than the second.
 > Note:  See sample scripts for more information on using the program.
**********************************************************************
### ⏳ TODO:
```
1) Add a graphical interface. (0% done)
```
### 💵 Support me:

You can thank me for developing this project, provide financial support for the development of new projects and buy me a small cup of coffee.☕\
  Just support me on these platforms: \
  ⭐[**Boosty**⭐](https://boosty.to/anodev) \
  ⭐[**DonationAlerts**⭐](https://www.donationalerts.com/r/anodev_development)
### 😀 Sample Result:
* __More sample images in ``docs/imgs/input/`` and ``docs/imgs/examples/`` folders__
* Input: 
* ![Input](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/input/1.jpg "Input")

* Output: 
* ![Output](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/examples/1.png "Output")
**********************************************************************

