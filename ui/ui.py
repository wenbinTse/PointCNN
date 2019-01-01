import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from opengl_widget import GLWidget

class UI(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1200, 800)

        self.main_layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Vertical, self)
        self.splitter.setStyleSheet("QSplitter:handle{background-color:grey}")

        #################################################################
        #### top layout
        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.top_layout.setAlignment(Qt.AlignLeft)

        self.open_button = QPushButton('打开')
        self.classify_button = QPushButton('分类')
        self.search_button = QPushButton('检索')
        self.button_widget = QWidget()
        self.button_layout = QVBoxLayout(self.button_widget)
        self.button_layout.setAlignment(Qt.AlignTop)
        self.button_layout.addWidget(self.open_button)
        self.button_layout.addWidget(self.classify_button)
        self.button_layout.addWidget(self.search_button)
        self.top_layout.addWidget(self.button_widget)

        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.true_label = QLabel('true label')
        self.true_label.setVisible(False)
        self.result_label = QLabel('result label')
        self.result_label.setVisible(False)
        self.result_layout.addWidget(self.true_label)
        self.result_layout.addWidget(self.result_label)
        self.top_layout.addWidget(self.result_widget, alignment=Qt.AlignTop)
        self.top_widget.setMaximumHeight(400)


        #################################################################

        #################################################################
        #### bottom layout
        self.bottom_container = QWidget()
        self.bottom_layout = QGridLayout(self.bottom_container)

        self.bottom_widget = QScrollArea()
        self.bottom_widget.setWidgetResizable(True)
        self.bottom_widget.setWidget(self.bottom_container)
        #################################################################

        self.splitter.addWidget(self.top_widget)
        self.splitter.addWidget(self.bottom_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 6)

        self.main_layout.addWidget(self.splitter)

