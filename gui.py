"""
Name: GUI
Description: This file contains the GUI interface.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Authors: Munawwar [https://github.com/Munawwar], Anodev (OPHoperHPO)[https://github.com/OPHoperHPO].
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
import os
import logging
import platform
import subprocess
import threading
import webview
from main import process
from libs.strings import MODELS_NAMES

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def show_error(w, e):
    """
    Shows error
    @param w: Window obj
    @param e: Error text
    """
    w.evaluate_js("""
    var err_label = document.querySelector('.error_label');
    err_label.style.visibility = "";
    err_label.textContent = 'REPLACE';
    """.replace("REPLACE", "An error has occurred! ERROR: " + str(e)))


# noinspection PyMissingOrEmptyDocstring
def worker_thread(win, input_files, model):
    logger.debug('Processing started ...')
    for i, file in enumerate(input_files):
        (file_path, file_name) = os.path.split(file)
        output_file = os.path.join(
            file_path,
            'bg-removed',
            # only support PNG files as of now
            os.path.splitext(file_name)[0] + '.png'
        )
        win.evaluate_js(
            "window.app.fileUploadButton.textContent = 'Processing "
            + str(i + 1) + ' of ' + str(len(input_files)) + " ...'"
        )
        try:
            process(file, output_file, model)
        except BaseException as e:
            show_error(win, e)
    win.evaluate_js("window.app.fileUploadButton.textContent = 'Select photos'")
    logger.debug('Processing complete')
    # noinspection PyUnboundLocalVariable
    open_folder(os.path.join(file_path, 'bg-removed'))


def onWindowStart(win2):
    """
    Window Start Event listener
    @param win2: Window object
    """

    def addModels():
        """
        Adds models to the model selection list.
        """

        def __add_model__(model: str):
            if not model == MODELS_NAMES[0]:
                win2.evaluate_js("""
              function createElementFromHTML(htmlString) {
                      var div = document.createElement('div');
                      div.innerHTML = htmlString.trim();
                    
                      // Change this to div.childNodes to support multiple top-level nodes
                      return div.firstChild; 
                    }
                    var models = document.querySelector('.models');
                    var rbut = createElementFromHTML('<label style="margin-left:16;" class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="MODELD"><input  id="MODELD" class="mdl-radio__button MODELD" type="radio" name="model" value="MODELD" /><span class="mdl-radio__label">MODELD</span></label>')
                    models.appendChild(rbut)
                    componentHandler.upgradeDom()
                    """.replace("MODELD", model))
            else:
                win2.evaluate_js("""
                function createElementFromHTML(htmlString) {
                        var div = document.createElement('div');
                        div.innerHTML = htmlString.trim();

                        // Change this to div.childNodes to support multiple top-level nodes
                        return div.firstChild; 
                      }
                      var models = document.querySelector('.models');
                      var rbut = createElementFromHTML('<label style="margin-left:16;" class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="MODELD"><input  id="MODELD" class="mdl-radio__button MODELD" type="radio" name="model" value="MODELD" checked /><span class="mdl-radio__label">MODELD</span></label>')
                      models.appendChild(rbut)
                      componentHandler.upgradeDom()
                      """.replace("MODELD", model))

        for i in MODELS_NAMES:
            __add_model__(i)

    def openFileDialog():
        """
        Opens a file selection dialog
        """
        file_types = ('Image Files (*.png;*.jpg;*.jpeg)', 'All files (*.*)')

        input_files = win2.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=file_types)
        logger.debug(input_files)

        model = win2.evaluate_js("window.app.getModel()")
        logger.debug('Use model: {}'.format(model))

        if input_files is not None:
            win2.evaluate_js("window.app.fileUploadButton.disabled = true")
            w_th = threading.Thread(target=worker_thread, args=(win2, input_files, model,))
            w_th.start()
            w_th.join()
            win2.evaluate_js("window.app.fileUploadButton.disabled = false")

    # expose a function during the runtime
    win2.expose(openFileDialog)
    if win2.loaded:
        addModels()


def open_folder(path):
    """
    Opens a output folder in file explorer.
    @param path: Path to folder
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def getfile(filename: str):
    """
    Opens html file
    @param filename: filename
    @return: file content
    """
    dir1 = os.path.dirname(__file__)
    path = os.path.join(dir1, filename)
    with open(path) as f:
        content = f.read()
    return content


if __name__ == '__main__':
    html = getfile('gui/index.html')
    window = webview.create_window('Automated BG Removal Tool', html=html)
    webview.start(onWindowStart, window)
