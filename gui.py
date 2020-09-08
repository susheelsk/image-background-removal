#!/usr/bin/python3
"""
Name: Gui for background removal tool.
Description: This file contains a QT based GUI.
Version: [release][3.3]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
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

# Built-in libraries
import time
import gc
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

# Third party libraries
from PyQt5.QtCore import *
from PyQt5.QtQml import *
from PyQt5.QtWidgets import *

# Libraries of this project
import gui.ui as ui
from gui.libs import qrc, config_utils

gui_dir = Path(__file__).parent.joinpath("gui")  # Absolute path to gui folder
config_ctl = config_utils.Config(gui_dir.joinpath("config.json"))  # Init config
config = config_ctl.c

if not config["tool"]["use_gpu"]:  # Enable or disable cuda acceleration
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Libraries of this project
from main import __save_image_file__
import libs.networks as networks
import libs.preprocessing as preprocessing
import libs.postprocessing as postprocessing


class UIcls:
    """UI initializer"""

    def __init__(self, engine):
        self.engine = engine  # Qml engine
        self.settings = ui.SettingsUI(config_ctl, self.engine)  # Configure settings ui

    def open_settings(self):
        """Shows settings ui"""
        self.settings.init(str(gui_dir.joinpath("qml/settings.qml")))

    @staticmethod
    def open_about():
        open_folder("https://github.com/OPHoperHPO/image-background-remove-tool/")


class App(QQmlApplicationEngine):
    """Main Window"""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        # noinspection PyArgumentList
        self.load(QUrl(str(gui_dir.joinpath("qml/main.qml"))))
        self.window = self.rootObjects()[0]  # Find root window
        self.ui = UIcls(self)  # Init ui

        self.photos_queue = multiprocessing.Queue()
        self.worker = Worker(self.config, self.photos_queue)  # Init worker
        self.worker.update_busypage.connect(self.__update_busypage__)

        self.init_ui()  # Initialize gui

    def init_ui(self):
        """Gui initializer"""
        # Configure window
        window = self.window
        window.closing.connect(sys.exit)

        # Configure ui elements
        fd = window.findChild(QObject, "fileDialog")
        fd.fileDialogCallback.connect(self.__fileDialogCallback__)
        logo_label = window.findChild(QObject, "logo_label")  # Find logo label
        row = logo_label.findChild(QObject, "row")  # Find row

        about_button = row.findChild(QObject, "about")  # Find settings button
        about_button.clicked.connect(self.ui.open_about)

        settings_button = row.findChild(QObject, "settings")  # Find settings button
        settings_button.clicked.connect(self.ui.open_settings)
        window.show()

        # Start worker
        self.worker.start()

    def __update_busypage__(self, data: list):
        """Updates the state of the program's busy interface"""
        window = self.window
        busy_page = window.findChild(QObject, "busyPage")
        busy_page.setProperty("visible", data[0])
        processing_label = busy_page.findChild(QObject, "processing_label")
        processing_label.setProperty("text", data[1])

    def __fileDialogCallback__(self, filedialog: QObject):
        """Callback for qt file dialog"""
        file_q_urls = filedialog.property("fileUrls")
        if len(file_q_urls) > 0:
            file_paths = [i.path() for i in file_q_urls]
            self.photos_queue.put(file_paths)


class Worker(QThread):
    update_busypage = pyqtSignal(list)

    def __init__(self, cfg: dict, queue: multiprocessing.Queue):
        super(Worker, self).__init__()
        self.config = cfg
        self.queue = queue

    def run(self):
        """Launches a worker"""
        model = networks.model_detect(self.config["tool"]["model"])  # Load model
        preprocessing_method = preprocessing.method_detect(self.config["tool"]["preprocessing_method"])
        postprocessing_method = postprocessing.method_detect(self.config["tool"]["postprocessing_method"])
        while True:
            file_paths = self.queue.get()  # Get file paths
            if len(file_paths) > 0:
                output_path = Path(file_paths[0]).parent.joinpath("bg_removed")
                # Show busy interface
                self.update_busypage.emit([True, "Processing {} of {}".format(0, len(file_paths))])
                for i, file_path in enumerate(file_paths):
                    self.update_busypage.emit([True, "Processing {} of {}".format(i + 1, len(file_paths))])
                    file_path = Path(file_path)
                    try:
                        image = model.process_image(file_path, preprocessing_method,
                                                    postprocessing_method)
                        __save_image_file__(image, file_path, output_path)
                    except BaseException as e:
                        self.update_busypage.emit([True, "A program error has occurred!\n"
                                                         "Run the gui in the console for more details."])
                        print("GUI WORKER ERROR!: ", e)
                        print("Raising an exception for debug through 15 seconds!\n"
                              "Please open an issue with this log in the project repository on GitHub. \n"
                              "More info here https://github.com/OPHoperHPO/image-background-remove-tool/issues")
                        time.sleep(15)
                        raise e
                self.update_busypage.emit([False, "Processing"])
                open_folder(str(output_path.absolute()))
                del file_paths, output_path
                gc.collect()  # Cleanup memory


def open_folder(path):
    """
    Opens a output folder in file explorer.
    :param path: Path to folder
    """
    if "win" in sys.platform:
        os.startfile(path)
    elif "darwin" in sys.platform:
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)  # Init Gui
        ex = App(config)
        sys.exit(app.exec_())  # Exit when app close
    except KeyboardInterrupt:
        sys.exit(0)
