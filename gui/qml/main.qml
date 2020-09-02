//Name: Main GUI QML File
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
import QtQuick 2.12
import QtQuick.Dialogs 1.0
import QtQuick.Window 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Controls 2.15

ApplicationWindow {
    id: mainWindow
    objectName: "mainWindow"
    visible: true
    width: 640
    height: 480
    Material.theme: Material.Light
    Material.accent: Material.Green
    title: qsTr("Automated BG Removal Tool")

    FileDialog {
        signal fileDialogCallback(QtObject file)
        id: fileDialog
        selectMultiple: true
        objectName: "fileDialog"
        title: "Please choose a files"
        folder: shortcuts.home
        nameFilters: ['Image files (*.jpg *.png)', 'All files (*)']
        onAccepted: {fileDialog.fileDialogCallback(fileDialog)}
    }

    Page {
        id: busyPage
        objectName: "busyPage"
        anchors.fill: parent
        visible: false
        onVisibleChanged: {if(busyPage.visible){logo_label.visible=false;}else{logo_label.visible=true;}}
        BusyIndicator {
            id: busyIndicator
            x: 281
            y: 301
            width: 86*mainWindow.width/(640)
            height: width
            anchors.verticalCenterOffset: 0
            anchors.horizontalCenterOffset: 0
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            objectName: "busyIndicator"

        }

        Text {
            id: processing_label
            x: 285
            y: 380
            width: 246
            height: 26
            objectName: "processing_label"
            text: qsTr("Processing...")
            anchors.verticalCenterOffset: 62*busyIndicator.width/(86)
            anchors.horizontalCenterOffset: 0
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 18*mainWindow.width/(640)
        }
    }

    Text {
        objectName: "logo_label"
        id: logo_label
        text: qsTr("Automated BG Removal Tool")
        anchors.bottomMargin: 64
        anchors.topMargin: 0
        wrapMode: Text.NoWrap
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        anchors.fill: parent
        font.pixelSize: 34*mainWindow.width/(640)
        Row {
            objectName: "row"
            id: row
            x: 85
            y: 244
            width: 470
            height: 30
            anchors.verticalCenterOffset: 48*logo_label.font.pixelSize/34
            anchors.horizontalCenterOffset: 0
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            rightPadding: (row.width-(b1.width+b2.width+b3.width+spacing*2))/2
            leftPadding: (row.width-(b1.width+b2.width+b3.width+spacing*2))/2
            spacing: 20

            Button {
                objectName: "settings"
                id: b1
                width: 116*logo_label.font.pixelSize/34
                height: 38*logo_label.font.pixelSize/34
                text: qsTr("Settings")
                Material.background: Material.Green
                font.pixelSize: 14*width/(116)
                Image {
                    id: b1_icon
                    width: 24*b1.width/116
                    height: 24*b1.height/38
                    anchors.left: parent.left
                    anchors.leftMargin: 2
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 3
                    anchors.top: parent.top
                    anchors.topMargin: 4
                    fillMode: Image.PreserveAspectFit
                    source: "icons/settings.png"
                }
            }

            Button {
                objectName: "select_photos"
                id: b2
                width: 156*logo_label.font.pixelSize/34
                height: 38*logo_label.font.pixelSize/34
                text: "Select Photos"
                hoverEnabled: false
                enabled: true
                Material.background: Material.Orange
                font.pixelSize: 14*width/(156)
                onClicked: {fileDialog.open()}
                Image {
                    id: b2_icon
                    width: 24*b2.width/156
                    height: 24*b2.height/38
                    anchors.left: parent.left
                    anchors.leftMargin: 2
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 3
                    anchors.top: parent.top
                    anchors.topMargin: 4
                    fillMode: Image.PreserveAspectFit
                    source: "icons/select_photos.png"
                }

            }

            Button {
                objectName: "about"
                id: b3
                width: 100*logo_label.font.pixelSize/34
                height: 38*logo_label.font.pixelSize/34
                text: qsTr("About")
                Material.background: Material.Green
                font.pixelSize: 14*width/(100)

                Image {
                    id: b3_icon
                    width: 24*b3.width/100
                    height: 24*b3.height/38
                    anchors.left: parent.left
                    anchors.leftMargin: 4
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 3
                    anchors.top: parent.top
                    anchors.topMargin: 4
                    fillMode: Image.PreserveAspectFit
                    source: "icons/about.png"
                }
            }

        }
    }



}
