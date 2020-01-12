import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from absl import logging
logging._warn_preinit_stderr = 0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import models

#import warnings
#warnings.filterwarnings('ignore')

class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(600)
        self.setFixedWidth(800)
        self.img = np.zeros((512, 512, 3), np.uint8)
        self.drawing = False
        self.mode = True
        self.ix, self.iy = -1, -1

        self.setupUi()

        self.show()

    def setupUi(self):

        self.header = QLabel(self)
        self.header.setGeometry(QtCore.QRect(0, 0, 801, 71))
        self.header.setStyleSheet("font-size: 20px; font: 75 20pt \"Palatino Linotype\"; color: white;\n"
                                    "background-color: #121; padding-top: 5px; padding-bottom: 5px;")
        self.header.setObjectName("header")
        self.header.setText("MNIST Classification Example")
        self.header.setAlignment(Qt.AlignCenter)


        self.footer = QLabel(self)
        self.footer.setGeometry(QtCore.QRect(0, 560, 801, 41))
        self.footer.setStyleSheet("font-size: 20px; font: 30 10pt \"Palatino Linotype\"; color: white;\n"
                                    "background-color: #121; padding-top: 5px; padding-bottom: 5px;")
        self.footer.setObjectName("footer")
        self.footer.setText("copyright "+u"\u00A9"+ " angshu.btf@gmail.com")
        self.footer.setAlignment(Qt.AlignCenter)


        self.outputLCD = QLabel(self)
        self.outputLCD.move(500,150)
        self.outputLCD.setFixedHeight(300)
        self.outputLCD.setFixedWidth(270)
        self.outputLCD.setStyleSheet("border: 2px solid; background-color: black; color: white; font-size: 240px;")
        self.outputLCD.setObjectName("outputLCD")
        self.outputLCD.setAlignment(Qt.AlignCenter)


        self.viewInputLabel = QLabel(self)
        self.viewInputLabel.setFixedWidth(300)
        self.viewInputLabel.setFixedHeight(300)
        self.viewInputLabel.move(50,150)
        self.viewInputLabel.setStyleSheet("background-color: black; color: white;")


        self.openGLWidget = QPushButton(self)
        self.openGLWidget.setText("Click Here To Start Drawing")
        self.openGLWidget.setFixedWidth(300)
        self.openGLWidget.setFixedHeight(50)
        self.openGLWidget.move(50, 460)
        self.openGLWidget.setObjectName("openGLWidget")
        self.openGLWidget.setStyleSheet("background-color: black; color: white; font-size: 20px; border: 2px solid white;")
        self.openGLWidget.clicked.connect(self.draw)


        self.inputWindowLabel = QLabel(self)
        self.inputWindowLabel.setGeometry(QtCore.QRect(40, 100, 331, 31))
        self.inputWindowLabel.setStyleSheet("font-size: 20px; font-family: \"Palatino Linotype\"; color: white;\n"
                                            "background-color: #121; padding-top: 5px; padding-bottom: 5px;")
        self.inputWindowLabel.setObjectName("inputWindowLabel")
        self.inputWindowLabel.setText("Draw The text to be Classified Here")
        self.inputWindowLabel.setAlignment(Qt.AlignCenter)


        self.predictButton = QPushButton(self)
        self.predictButton.setGeometry(QtCore.QRect(390, 280, 93, 51))
        self.predictButton.setStyleSheet("font-size: 20px; font-family:  \"Palatino Linotype\"; color: white;\n"
                                        "background-color: #121; padding-top: 5px;\n"
                                        "padding-bottom: 5px; border-radius: 10px;")
        self.predictButton.setObjectName("predictButton")
        self.predictButton.setText("Predict")
        self.predictButton.clicked.connect(self.button_click)


        self.predictLabel = QLabel(self)
        self.predictLabel.setGeometry(QtCore.QRect(500, 100, 271, 31))
        self.predictLabel.setStyleSheet("font-size: 20px; font-family: \"Palatino Linotype\";\n"
                                        "color: white; background-color: #121; padding-top: 5px; padding-bottom: 5px;")
        self.predictLabel.setObjectName("predictLabel")
        self.predictLabel.setText("Classified Text")
        self.predictLabel.setAlignment(Qt.AlignCenter)

        self.reset = QPushButton(self)
        self.reset.setFixedHeight(50)
        self.reset.setFixedWidth(270)
        self.reset.move(500,460)
        self.reset.setText("Reset Window")
        self.reset.setStyleSheet("background-color: black; color: white; font-size: 20px; border: 2px solid white;")
        self.reset.clicked.connect(self.resetWindow)

    def resetWindow(self):
        self.viewInputLabel.setPixmap(QPixmap("defaultBlankInput.jpg"))
        self.outputLCD.setText('')

    def button_click(self):
        self.predict_image()

    def draw(self):
        cv2.namedWindow('Draw Here')
        cv2.setMouseCallback('Draw Here', self.draw_circle)

        while True:
            cv2.imshow('Draw Here', self.img)
            # cv2.moveWindow('Draw Here',50,200)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == ord('q'):
                img = cv2.resize(self.img, (300, 300))
                cv2.imwrite('showInput.jpg', img)
                img = cv2.resize(self.img, (28, 28))
                cv2.imwrite('userInput.jpg', img)
                break

        cv2.destroyAllWindows()
        self.img = np.zeros((512, 512, 3), np.uint8)
        pixMap = QPixmap("showInput.jpg")
        self.viewInputLabel.setPixmap(pixMap)

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode == False:
                    cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (255, 255, 255), -1)
                else:
                    cv2.circle(self.img, (x, y), 20, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True:
                a = 1
                # cv2.rectangle(img, (ix,iy), (x,y), (255,255,255), -1)
            else:
                cv2.circle(self.img, (x, y), 20, (255, 255, 255), -1)

    def predict_image(self):
        img = cv2.imread('userInput.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = np.array(img)
        img = img.reshape((1, 28, 28, 1))
        model = models.load_model('mnist_digits_098_002.h5')
        predictValue = np.argmax(model.predict(img))
        self.outputLCD.setText(str(predictValue))

if __name__ == "__main__":
    import sys
    App = QApplication(sys.argv)
    UI_file = Ui_MainWindow()
    sys.exit(App.exec())
