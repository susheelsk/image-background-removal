# Portrait Segmentation using Tensorflow

This script removes the background from an input image. You can read more about segmentation [here](http://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)

### Setup
The script setup.sh downloads the trained model and sets it up so that the seg.py script can understand. 
>	./setup.sh

### Running the script
Go ahead and use the script as specified below, to execute fast but lower accuracy model:
>	python3 seg.py sample.jpg sample.png 

For better accuracy, albiet a slower approach, go ahead and try :
>	python3 seg.py sample.jpg sample.png 1

### Dependencies
>	tensorflow, PIL

### Sample Result
Input: 
![alt text](https://github.com/callmesusheel/image-background-removal/raw/master/sample.jpg "Input")

Output: 
![alt text](https://github.com/callmesusheel/image-background-removal/raw/master/sample_bgremoved.png "Output")