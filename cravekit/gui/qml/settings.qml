//Name: Settings QML File
//Description: This file contains the QT based GUI.
//Version: [release][3.3]
//Source url: https://github.com/OPHoperHPO/image-background-remove-tool
//Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
//License: Apache License 2.0
//License:
//   Copyright 2020 OPHoperHPO

//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

import QtQuick 2.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.2
import QtQuick.Controls.Material 2.1

ApplicationWindow{
    title: qsTr('Settings')
    id: configWindow
    objectName: "configWindow"
    width:  640
    height: 480
    visible: true
    Material.theme: Material.Light
    Material.accent: Material.Green

    Text {
        id: text_label
        x: 373
        y: 557
        text: qsTr("Settings")
        objectName: "textLabel"
        anchors.right: parent.right
        anchors.rightMargin: 20
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 13
        renderType: Text.NativeRendering
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize: 24
        color: listView.textColor
    }

    ListView {
        signal itemValChanged(QtObject row_id)
        property string textColor: "#ffffff"
        property string editTextColor: "#13e600"
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.topMargin: 8
        anchors.bottom: apply_button.top
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottomMargin: 0
        id: listView
        objectName: "listView"
        delegate: Item {
            id: item_list
            width: listView.width
            height: row_id.height +5
            Row {
                id: row_id
                objectName: pr_id
                spacing: 10
                width: item_list.width
                Text{
                    id: t1
                    color: listView.textColor
                    objectName: pr_id
                    text: pr_name
                    font.pixelSize: 17
                }

                RadioButton {
                    id: r1
                    text: r1_text
                    checked: r1_val
                    objectName: "r1"
                    font.pixelSize: 17
                    visible: r1_visible
                    onCheckedChanged: listView.itemValChanged(row_id)
                }
                RadioButton {
                    id: r2
                    text: r2_text
                    checked: r2_val
                    objectName: "r2"
                    font.pixelSize: 17
                    visible: r2_visible
                    onCheckedChanged: listView.itemValChanged(row_id)
                }
                RadioButton {
                    id: r3
                    text: r3_text
                    objectName: "r3"
                    checked: r3_val
                    font.pixelSize: 17
                    visible: r3_visible
                    onCheckedChanged: listView.itemValChanged(row_id)
                }
                TextField{
                    id: te1
                    objectName: "te1"
                    font.pixelSize: 17
                    color: listView.editTextColor
                    text: te1_text
                    visible: te1_visible
                    onTextChanged: listView.itemValChanged(row_id)
                }
                CheckBox {
                    id: ch1
                    objectName: "ch1"
                    text: ch1_text
                    checked: ch1_val
                    visible: ch1_visible
                    font.pixelSize: 17
                    onCheckedChanged: listView.itemValChanged(row_id)
                }
            }

        }

        model: mdel
        function append(newElement) {
            mdel.append(newElement)
        }

        ListModel{
            id:mdel
        }
    }

    Button {
        id: apply_button
        y: 544
        text: qsTr("Apply Changes")
        objectName: "applyButton"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 4
        anchors.left: parent.left
        anchors.leftMargin: 4
        font.pointSize: 12
        enabled: false
    }

    Button {
        id: exit_button
        x: -9
        y: 548
        text: qsTr("Go back")
        objectName: "exitButton"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 4
        anchors.left: apply_button.right
        font.pointSize: 12
        anchors.leftMargin: 36
    }


}









/*##^##
Designer {
    D{i:1;anchors_x:373}D{i:12;anchors_x:894}D{i:13;anchors_x:894}
}
##^##*/
