# Tool to remove the background from the portrait using Tensorflow
A tool to remove a background from a portrait image using Tensorflow

### Dependencies
```	tensorflow, pillow, tqdm, numpy, scipy ```

### Setup
* Clone repository ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Run ```./bin/setup.sh``` _This setup.sh script loads the trained model._
### Running the script
 * Put images to the input folder.
 * Run ```run.sh``` for Linux or ```run.bat``` for Windows

### Differences from the [original script](https://github.com/susheelsk/image-background-removal)
* Added comments to the code.
* Added ```tqdm``` progress bar.
* __Removes background from image without loss of image resolution.__
* __The script now processes all images from the input folder and saves them to the output folder with the same name.__
* __New sample images.__

### Sample Result:
```More sample images in input and output folders```
Input: 
![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/input/1.jpg "Input")

Output: 
![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/output/1.png "Output")
