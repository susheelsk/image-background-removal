# Removing the background from portrait photos using Tensorflow

This script removes the background from an input image.

### Dependencies
```	tensorflow, pillow, tqdm, numpy, scipy ```

### Setup
* Clone repository ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Run ```./bin/setup.sh``` _This setup.sh script loads the trained model._
### Running the script
 * Put images to the input folder.
 * Use the script as specified below.
```	./run.sh```

### Differences from the [original script](https://github.com/susheelsk/image-background-removal)
* Added comments to the code.
* Added ```tqdm``` progress bar.
* __Removes background from image without loss of image resolution.__
* __The script now processes all images from the input folder and saves them to the output folder with the same name.__

### Sample Result:
Input: 
![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/input/1.jpg "Input")

Output: 
![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/output/1.png "Output")
