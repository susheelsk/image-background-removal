#!/bin/bash
# All files in the input directory will be saved in the output folder under their names with the extension .png.
# When replacing folder paths, ALWAYS add / (or \ in the case of Windows) at the end!
python3 main.py ./docs/imgs/input/ ./docs/imgs/output/ xception_model
