"""
Name: Strings file
Description: This file contains the strings.
Version: [release][3.3]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

NAME = "Image Background Remove Tool"
MODELS_NAMES = ["u2net", "basnet", "u2netp", "xception_model", "mobile_net_model"]
PREPROCESS_METHODS = ["None", "bbmd-maskrcnn", "bbd-fastrcnn"]
POSTPROCESS_METHODS = ["fba", "rtb-bnb", "rtb-bnb2", "No"]
DESCRIPTION = "A tool to remove a background from image using Neural Networks"
LICENSE = "Apache License 2.0"
ARGS = {
    "-i": ["<input_path>", "Path to input file or dir."],
    "-o": ["<output_path>", "Path to output file or dir."],
    "-m": ["<model_type>", "Can be {}. u2net is better to use.\n"
                           "\t\t\t\t  DeepLab models (xception_model or mobile_net_model) are outdated\n"
                           "\t\t\t\t  and designed to remove the background from PORTRAIT photos or PHOTOS WITH "
                           "ANIMALS! "
                           "".format(' or '.join(MODELS_NAMES))],
    "-prep": ["<preprocessing_method>", "Preprocessing method. Can be {} . "
                                        "`{}` is better to use.".format(' or '.join(PREPROCESS_METHODS),
                                                                        PREPROCESS_METHODS[0])],
    "-postp": ["<postprocessing_method>", "Postprocessing method. Can be {} . "
                                          "`{}` is better to use.".format(' or '.join(POSTPROCESS_METHODS),
                                                                          POSTPROCESS_METHODS[0])],
}
ARGS_HELP = """{}
{}
License: {}
Running the script:
python3 main.py {}
Explanation of args:
{}
""".format(NAME, DESCRIPTION, LICENSE, ' '.join([i + " " + ARGS[i][0] for i in ARGS]),
           '\n'.join([i + " " + ARGS[i][0] + ' - ' + ARGS[i][1] for i in ARGS]))
