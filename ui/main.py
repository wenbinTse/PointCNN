import sys
import os
current_path = os.path.dirname(__file__)
father_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
sys.path.insert(1, father_path)
import numpy as np
import json

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from opengl_widget import GLWidget
from ui import UI

import application

with open(os.path.join(current_path, 'label_id2label.json')) as f:
    label_id2label = json.load(f)

class Window(UI):
    def __init__(self):
        super().__init__()
        self.open_button.clicked.connect(self.choose_file)
        self.classify_button.clicked.connect(self.classify)
        self.search_button.clicked.connect(self.search)

        self.model = None
        self.model_glwidget = None
        self.label = ''

    @pyqtSlot()
    def choose_file(self):
        model_name, _ = QFileDialog.getOpenFileName(
            self,
            "打开模型文件",
            "",
            "*.npy;")

        if not model_name == '':
            self.model = np.load(model_name)
            self.label = os.path.basename(model_name).split('_')[0]
            print(self.model.shape)

            if self.model_glwidget is None:
                self.model_glwidget = GLWidget(self.model[0])
                self.top_layout.insertWidget(1, self.model_glwidget, alignment=Qt.AlignLeft)
            else:
                self.model_glwidget.update_data(self.model[0])

    @pyqtSlot()
    def classify(self):
        if self.model is None:
            QMessageBox.information(self, '未加载模型', '请打开一个点云模型')
            return

        k_probs, k_labels = application.classify(self.model)

        self.true_label.setText('True label: {}'.format(self.label))
        self.true_label.setVisible(True)

        result_str = 'TOP5:\n'
        for prob, label_id in zip(k_probs, k_labels):
            result_str += '{:<15}{:2f}\n'.format(label_id2label[str(label_id)], prob)
        self.result_label.setText(result_str)
        self.result_label.setVisible(True)

    @pyqtSlot()
    def search(self):
        if self.model is None:
            QMessageBox.information(self, '未加载模型', '请打开一个点云模型')
            return

        models = application.search(self.model, k=18)

        while self.bottom_layout.takeAt(0):
            tmp = self.bottom_layout.takeAt(0)
            tmp_widget = tmp.widget()
            del tmp_widget
            del tmp


        for i, model in enumerate(models):
            self.bottom_layout.addWidget(GLWidget(model), i // 3 + 1, i % 3 + 1)
            print('added')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle('点云模型分类和检索')
    window.show()
    sys.exit(app.exec_())