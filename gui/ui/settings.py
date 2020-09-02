"""
Name: Settings UI File
Description: This file contains the QT based settings UI.
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
import ast

from PyQt5.QtCore import *
from PyQt5.QtQuick import *

from gui.libs.config_utils import param2text


class SettingsUI:
    """SettingsUI window"""

    def __init__(self, config_ctl, engine):
        self.ctl = config_ctl  # Config object
        self.config = config_ctl.c  # Config dict
        self.engine = engine  # Qml Engine
        self.window = None  # configWindow
        self.apply_button = None  # Apply changes button
        self.textlabel = None  # Text label

    def init(self, path):
        """
        Initializes gui
        :param path: path to qml file
        """
        if self.window:
            self.window.show()
        else:
            self.engine.load(QUrl(path))
            for window in self.engine.rootObjects():  # Find window object
                if window.objectName() == "configWindow":
                    self.window = window
                    break
            if self.window:
                self.__init_elements__()  # Init ui
            else:
                raise Exception("Config window not found! Exit!")
            self.window.show()

    def cls(self):
        """Hides a window"""
        self.window.hide()

    def __init_elements__(self):
        """Initializes interface elements"""
        window = self.window

        listview = window.findChild(QObject, "listView")  # Find listview
        listview.itemValChanged.connect(self.item_val_changed)

        self.textlabel = window.findChild(QObject, "textLabel")  # Find textlabel

        self.apply_button = window.findChild(QObject, "applyButton")  # Find Apply Changes Button
        self.apply_button.clicked.connect(self.save_config)

        exit_button = window.findChild(QObject, "exitButton")  # Find Exit button
        exit_button.clicked.connect(self.cls)
        listview.setProperty("textColor", "#000000")
        listview.setProperty("editTextColor", "#005505")

        self.__cfg_sect_reverse__(listview, self.config)  # Start parsing and add config options to the screen

    def __cfg_sect_reverse__(self, listview, value, path="", p=""):
        """Recursive config parser
        :param listview ListView
        :param value Config dict
        """
        if type(value) is dict:
            for sc in value:
                pr = value[sc]
                if path == "":
                    path = str([sc])
                else:
                    path += str([sc])
                if type(pr) is dict:
                    self.__cfg_sect_reverse__(listview, pr, path, path)
                    path = p
                else:
                    val = self.__get_empty_line__()  # Add one line to listView
                    val['pr_name'] = param2text(path) + " = "  # Get normal parameter text from cfg_text.py
                    val['pr_id'] = path
                    if type(pr) is bool:
                        val["ch1_val"] = pr
                        val['ch1_visible'] = True
                        self.__add_line__(listview, val)
                    elif type(pr) is str:
                        val['te1_text'] = str(pr)
                        val['te1_visible'] = True
                        self.__add_line__(listview, val)
                    elif type(pr) is int:
                        val['te1_text'] = str(pr)
                        val['te1_visible'] = True
                        self.__add_line__(listview, val)
                    elif type(pr) is list:
                        val['te1_text'] = str(pr)
                        val['te1_visible'] = True
                        self.__add_line__(listview, val)
                    path = p

    def __add_line__(self, listview, value):
        """Displays one line from the config
        :param listview: ListView QtQuick object
        :param value: Look __get_empty_line__ function for more details.
        """
        QMetaObject.invokeMethod(listview, "append", Q_ARG(QVariant, value))

    def __get_empty_line__(self):
        """Returns an empty dictionary with parameters for interface elements of one configuration line"""
        return {
            "pr_name": "",  # Visible text
            "pr_id": "",  # param id
            "r1_text": "",  # radiobutton
            "r1_val": False,
            "r1_visible": False,
            "r2_text": "",  # radiobutton
            "r2_val": False,
            "r2_visible": False,
            "r3_text": "",  # radiobutton
            "r3_val": False,
            "r3_visible": False,
            "te1_text": "",  # textedit
            "te1_visible": False,
            "ch1_text": "",  # checkbox
            "ch1_val": False,
            "ch1_visible": False
        }

    def save_config(self):
        """Saves configuration to config.json file"""
        self.ctl.save()  # Save config to file
        self.textlabel.setProperty("text", "Changes was saved! \n"  # Display text
                                           " Restart app to apply changes")

    def item_val_changed(self, row: QQuickItem):
        """Configuration Change Listener"""
        property_id = row.property("objectName")  # Get property_id
        if property_id != "":  # Filter app start
            if not self.apply_button.property("enabled"):
                self.apply_button.setProperty("enabled", True)  # Enable apply changes button
            for obj in row.children():
                if obj.property('visible') is True and obj.property('objectName') != property_id:
                    checked = obj.property('checked')
                    text = obj.property('text')
                    type_val = eval("type(self.config{})".format(property_id))
                    if checked is not None:  # If checkbox
                        if type(checked) is bool and type_val is bool:
                            exec("self.config{} = {}".format(property_id, checked))  # Write config var
                    elif text is not None:  # If text input
                        if "[" in text and "]" in text and "," in text and type_val is list:
                            text = check_input(text)
                            if text:
                                exec("self.config{} = list({})".format(property_id, text))
                        elif type_val is str:
                            text = "'" + text + "'"
                            text = check_input(text)
                            if text:
                                exec("self.config{} = str('{}')".format(property_id, text))
                        elif type_val is int:
                            if text is '':
                                text = 0
                            text = check_input(text)
                            if text:
                                exec("self.config{} = int('{}')".format(property_id, text))


def check_input(string: str):
    """Checks a string for unsafe expressions"""
    r = False
    try:
        string = ast.literal_eval(string)
    except BaseException:
        r = True
    if not r:
        return string
    else:
        return None
