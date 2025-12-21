# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MaskEditorUI.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_MaskEditorUI(object):
    def setupUi(self, MaskEditorUI):
        if not MaskEditorUI.objectName():
            MaskEditorUI.setObjectName(u"MaskEditorUI")
        MaskEditorUI.resize(1200, 800)
        self.centralwidget = QWidget(MaskEditorUI)
        self.centralwidget.setObjectName(u"centralwidget")
        self.mainLayout = QHBoxLayout(self.centralwidget)
        self.mainLayout.setObjectName(u"mainLayout")
        self.leftPanelLayout = QVBoxLayout()
        self.leftPanelLayout.setObjectName(u"leftPanelLayout")
        self.taskLabel = QLabel(self.centralwidget)
        self.taskLabel.setObjectName(u"taskLabel")

        self.leftPanelLayout.addWidget(self.taskLabel)

        self.taskCombo = QComboBox(self.centralwidget)
        self.taskCombo.addItem("")
        self.taskCombo.addItem("")
        self.taskCombo.setObjectName(u"taskCombo")

        self.leftPanelLayout.addWidget(self.taskCombo)

        self.displayLabel = QLabel(self.centralwidget)
        self.displayLabel.setObjectName(u"displayLabel")

        self.leftPanelLayout.addWidget(self.displayLabel)

        self.displayCombo = QComboBox(self.centralwidget)
        self.displayCombo.addItem("")
        self.displayCombo.addItem("")
        self.displayCombo.setObjectName(u"displayCombo")

        self.leftPanelLayout.addWidget(self.displayCombo)

        self.btnOpenFolder = QPushButton(self.centralwidget)
        self.btnOpenFolder.setObjectName(u"btnOpenFolder")

        self.leftPanelLayout.addWidget(self.btnOpenFolder)

        self.folderStatusLabel = QLabel(self.centralwidget)
        self.folderStatusLabel.setObjectName(u"folderStatusLabel")
        self.folderStatusLabel.setStyleSheet(u"color: gray; font-size: 10px;")
        self.folderStatusLabel.setWordWrap(True)

        self.leftPanelLayout.addWidget(self.folderStatusLabel)

        self.listWidget = QListWidget(self.centralwidget)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setMinimumSize(QSize(200, 0))

        self.leftPanelLayout.addWidget(self.listWidget)

        self.modeGroup = QGroupBox(self.centralwidget)
        self.modeGroup.setObjectName(u"modeGroup")
        self.modeLayout = QVBoxLayout(self.modeGroup)
        self.modeLayout.setObjectName(u"modeLayout")

        self.leftPanelLayout.addWidget(self.modeGroup)

        self.brushSizeLabel = QLabel(self.centralwidget)
        self.brushSizeLabel.setObjectName(u"brushSizeLabel")

        self.leftPanelLayout.addWidget(self.brushSizeLabel)

        self.brushSlider = QSlider(self.centralwidget)
        self.brushSlider.setObjectName(u"brushSlider")
        self.brushSlider.setMinimum(1)
        self.brushSlider.setMaximum(50)
        self.brushSlider.setValue(10)
        self.brushSlider.setOrientation(Qt.Orientation.Horizontal)

        self.leftPanelLayout.addWidget(self.brushSlider)

        self.opacityLabel = QLabel(self.centralwidget)
        self.opacityLabel.setObjectName(u"opacityLabel")

        self.leftPanelLayout.addWidget(self.opacityLabel)

        self.opacitySlider = QSlider(self.centralwidget)
        self.opacitySlider.setObjectName(u"opacitySlider")
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(50)
        self.opacitySlider.setOrientation(Qt.Orientation.Horizontal)

        self.leftPanelLayout.addWidget(self.opacitySlider)

        self.undoRedoLayout = QHBoxLayout()
        self.undoRedoLayout.setObjectName(u"undoRedoLayout")
        self.undoBtn = QPushButton(self.centralwidget)
        self.undoBtn.setObjectName(u"undoBtn")

        self.undoRedoLayout.addWidget(self.undoBtn)

        self.redoBtn = QPushButton(self.centralwidget)
        self.redoBtn.setObjectName(u"redoBtn")

        self.undoRedoLayout.addWidget(self.redoBtn)


        self.leftPanelLayout.addLayout(self.undoRedoLayout)

        self.saveBtn = QPushButton(self.centralwidget)
        self.saveBtn.setObjectName(u"saveBtn")

        self.leftPanelLayout.addWidget(self.saveBtn)

        self.finalBtn = QPushButton(self.centralwidget)
        self.finalBtn.setObjectName(u"finalBtn")

        self.leftPanelLayout.addWidget(self.finalBtn)

        self.unfinalBtn = QPushButton(self.centralwidget)
        self.unfinalBtn.setObjectName(u"unfinalBtn")

        self.leftPanelLayout.addWidget(self.unfinalBtn)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.leftPanelLayout.addItem(self.verticalSpacer)


        self.mainLayout.addLayout(self.leftPanelLayout)

        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setObjectName(u"scrollArea")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(False)
        self.scrollArea.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(400, 372, 100, 30))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.mainLayout.addWidget(self.scrollArea)

        MaskEditorUI.setCentralWidget(self.centralwidget)

        self.retranslateUi(MaskEditorUI)

        QMetaObject.connectSlotsByName(MaskEditorUI)
    # setupUi

    def retranslateUi(self, MaskEditorUI):
        MaskEditorUI.setWindowTitle(QCoreApplication.translate("MaskEditorUI", u"Vessel Annotation Tool", None))
        self.taskLabel.setText(QCoreApplication.translate("MaskEditorUI", u"Task:", None))
        self.taskCombo.setItemText(0, QCoreApplication.translate("MaskEditorUI", u"Artery/Vein", None))
        self.taskCombo.setItemText(1, QCoreApplication.translate("MaskEditorUI", u"Optic Disc", None))

        self.displayLabel.setText(QCoreApplication.translate("MaskEditorUI", u"Display:", None))
        self.displayCombo.setItemText(0, QCoreApplication.translate("MaskEditorUI", u"Original Image", None))
        self.displayCombo.setItemText(1, QCoreApplication.translate("MaskEditorUI", u"Enhanced Image", None))

#if QT_CONFIG(tooltip)
        self.btnOpenFolder.setToolTip(QCoreApplication.translate("MaskEditorUI", u"Open a folder containing mask and image subfolders", None))
#endif // QT_CONFIG(tooltip)
        self.btnOpenFolder.setText(QCoreApplication.translate("MaskEditorUI", u"Open Folder", None))
        self.folderStatusLabel.setText(QCoreApplication.translate("MaskEditorUI", u"No folder loaded", None))
        self.modeGroup.setTitle(QCoreApplication.translate("MaskEditorUI", u"Modes", None))
        self.brushSizeLabel.setText(QCoreApplication.translate("MaskEditorUI", u"Brush size", None))
        self.opacityLabel.setText(QCoreApplication.translate("MaskEditorUI", u"Mask opacity", None))
        self.undoBtn.setText(QCoreApplication.translate("MaskEditorUI", u"Undo", None))
        self.redoBtn.setText(QCoreApplication.translate("MaskEditorUI", u"Redo", None))
        self.saveBtn.setText(QCoreApplication.translate("MaskEditorUI", u"Save", None))
        self.finalBtn.setText(QCoreApplication.translate("MaskEditorUI", u"Finish", None))
        self.unfinalBtn.setText(QCoreApplication.translate("MaskEditorUI", u"Unfinish", None))
    # retranslateUi

