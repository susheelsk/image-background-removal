# Portrait Segmentation using Tensorflow

_This is a github-fork of this [repo](https://github.com/susheelsk/image-background-removal)_

## Improvements

* Add window batch script for setup
* Able to convert images using Python instead of running script
* Convert single image or multiple images efficiently

This script removes the background from an input image. You can read more about segmentation [here](http://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)

## Setup
The script setup.sh downloads the trained model and sets it up so that the seg.py script can understand.

For windows : 
```bat
./setup.sh
```

For Unix:
```bsh
./setup.sh
```

## Usage

### Running the script
Go ahead and use the script as specified below, to execute fast but lower accuracy model:
>	python3 seg.py sample.jpg sample.png 

For better accuracy, albiet a slower approach, go ahead and try :
>	python3 seg.py sample.jpg sample.png 1

### Python

The following functions are supported:

* `convert_image(image)` : Takes in an image and returns a image with background removed.
* `convert_image_from_dir(source_dir, destination_dir)` : Converts an image specified at the source_dir and saves a copy with background removed with destination_dir path. Also returns a copy of the converted image. Pass `None` if saving image is not required. 
* `convert_all_in_dir(source_dir, destination_dir)` : Converts all images in the source_dir folder and saves a copy with background removed in the destination_dir folder. Also returns a list of converted images. Pass `None` if saving images is not required. 
* `getAllImages(source_dir)` : Returns a list of all images in the source_dir folder.

## Dependencies
>	tensorflow, PIL

## Sample Result
Input: 
![alt text](https://github.com/callmesusheel/image-background-removal/raw/master/sample.jpg "Input")

Output: 

![alt text](https://github.com/callmesusheel/image-background-removal/raw/master/sample_bgremoved.png "Output")