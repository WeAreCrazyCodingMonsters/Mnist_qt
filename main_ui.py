import os.path
import time
from PySide2.QtGui import QPixmap
from PySide2.QtGui import QFont
from PySide2.QtUiTools import QUiLoader
from os.path import exists, basename
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QComboBox
from PySide2.QtWidgets import QHBoxLayout
from PySide2.QtGui import QColor, QPalette
from main import main_training, main_testing, predict_img
# from script.get_mnist import get_list
from PySide2.QtCore import Qt, QThread, Signal

# 创建调色板对象
palette = QPalette()

# 创建颜色对象
color = QColor(255, 255, 255)  # 白色

# 设置标签的字体颜色
palette.setColor(QPalette.WindowText, color)

# 创建一个QFont对象并设置字体、大小和加粗属性
font_ = QFont('Microsoft YaHei', 10, QFont.Bold)  # 这里使用微软雅黑字体，字号为12，加粗属性为True

class BackendThread_train(QThread):

    def __init__(self, dataset_name, output_folder_path, device, yaml_filename,
                 textBrowser, image_label, able):
        QThread.__init__(self)
        self.dataset_name = dataset_name
        self.output_folder_path = output_folder_path
        self.device = device
        self.yaml_filename = yaml_filename
        self.textBrowser = textBrowser
        self.image_label = image_label
        self.able = able

    def run(self):
        main_training(self.dataset_name, self.output_folder_path, self.device, self.yaml_filename,
                      textBrowser=self.textBrowser)
        pixmap = QPixmap('figs/log.png').scaled(504, 506)
        self.image_label.setPixmap(pixmap)
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        print(ttt + "训练结束!")
        self.textBrowser.append(ttt + "训练结束!")
        self.able(True)


class BackendThread_test(QThread):

    def __init__(self, dataset_name, output_folder_path, device, yaml_filename,
                 textBrowser, able):
        QThread.__init__(self)
        self.dataset_name = dataset_name
        self.output_folder_path = output_folder_path
        self.device = device
        self.yaml_filename = yaml_filename
        self.textBrowser = textBrowser
        self.able = able

    def run(self):
        # ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        # self.textBrowser.append(ttt + '文件列表已生成!')
        main_testing(self.dataset_name, self.output_folder_path, self.device, self.yaml_filename,
                     textBrowser=self.textBrowser)
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        print(ttt + "测试结束!")
        self.textBrowser.append(ttt + "测试结束!")
        self.able(True)


class BackendThread_predict(QThread):

    def __init__(self, img_path, weight_path, textBrowser, image_label):
        QThread.__init__(self)
        self.img_path = img_path
        self.weight_path = weight_path
        self.textBrowser = textBrowser
        self.image_label = image_label

    def run(self):
        pixmap = QPixmap(self.img_path).scaled(600, 600)
        self.image_label.setPixmap(pixmap)

        predict_img(self.img_path, self.weight_path, textBrowser=self.textBrowser)
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        print(ttt + "预测结束!")
        self.textBrowser.append(ttt + "预测结束!")
        # self.able(True)


class Stats:
    def __init__(self):
        self.pt_filename = ""
        self.yaml_filename = ""
        self.fig_filename = ""
        self.output_folder_path = ""
        self.ui = QUiLoader().load('ui/UI.ui')
        # self.ui.setWindowFlag(Qt.FramelessWindowHint)

        self.ui.button_train.clicked.connect(self.train)
        self.ui.button_test.clicked.connect(self.test)
        self.ui.button_predict.clicked.connect(self.predict)
        self.ui.textBrowser.verticalScrollBar().setValue(self.ui.textBrowser.verticalScrollBar().maximum())

        self.page_0()
        self.page_1()
        self.page_2()
        self.page_3()

        self.page1.setVisible(False)
        self.page2.setVisible(False)
        self.page3.setVisible(False)

        layout = self.ui.verticalLayout
        layout.addWidget(self.page0)
        self.page0.setVisible(True)
        layout.addWidget(self.page1)
        layout.addWidget(self.page2)
        layout.addWidget(self.page3)

        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + 'MNIST分类器启动成功。')

    def page_0(self):
        self.page0 = QWidget()
        self.label0 = QLabel(
            "|" + "欢迎使用MNIST分类器!".center(20) + "|" + "\n" + "      |" + "作者: 葛钰峣".center(20) + "|")
        font = QFont("微软雅黑", 12)  # 创建一个12号微软雅黑字体
        self.label0.setPalette(palette)
        self.label0.setFont(font)  # 将QLabel的字体设置为微软雅黑
        layout0 = QHBoxLayout(self.page0)
        layout0.addWidget(self.label0, alignment=Qt.AlignCenter)

    def page_1(self):
        self.page1 = QWidget()

        layout_0 = QVBoxLayout(self.page1)
        self.label0 = QLabel("训练模式")
        font = QFont("微软雅黑", 14)  # 创建一个12号微软雅黑字体
        self.label0.setPalette(palette)
        self.label0.setFont(font)
        layout_0.addWidget(self.label0, alignment=Qt.AlignCenter)

        layout_1 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.label_1 = QLabel("数据集")
        self.label_1.setFont(font_)
        self.label_1.setPalette(palette)
        self.combo_box_1 = QComboBox()
        self.combo_box_1.currentIndexChanged.connect(self.get_dataset)
        self.combo_box_1.setFixedWidth(130)
        self.combo_box_1.addItem("MNIST")
        self.combo_box_1.addItem("FashionMNIST")
        layout_1.addWidget(self.label_1)
        layout_1.addWidget(self.combo_box_1)

        layout_2 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_2_0 = QLabel('权重文件输出路径')
        self.folder_label_2_0.setFont(font_)
        self.folder_label_2_0.setPalette(palette)
        self.folder_label_2 = QLabel()
        self.folder_label_2.setPalette(palette)
        self.folder_button_2 = QPushButton('...')
        self.folder_button_2.setFixedWidth(30)
        self.folder_button_2.clicked.connect(self.choose_output_folder)
        layout_2.addWidget(self.folder_label_2_0)
        layout_2.addWidget(self.folder_button_2)
        layout_2.addWidget(self.folder_label_2)

        layout_3 = QHBoxLayout(alignment=Qt.AlignCenter)
        self.folder_label_3 = QLabel()
        self.train_button = QPushButton('执行训练')
        self.train_button.setFont(font_)
        self.train_button.setCursor(Qt.PointingHandCursor)
        self.train_button.setStyleSheet('''
                QPushButton {
                    color: white;
                    background-color: (39,42,71);
                    border: 1px solid #dcdfe6;
                    padding: 5px;
                    border-radius: 5px;
                }

                QPushButton:hover {
                    background-color: #ecf5ff;
                    color: #409eff;
                }

                QPushButton:pressed, QPushButton:checked {
                    border: 1px solid #3a8ee6;
                    color: #409eff;
                }
                ''')

        self.train_button.setPalette(palette)
        self.train_button.clicked.connect(self.train_exe)
        layout_3.addWidget(self.train_button)

        layout_4 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.label_4 = QLabel("执行硬件")
        self.label_4.setPalette(palette)
        self.label_4.setFont(font_)
        self.combo_box = QComboBox()
        self.combo_box.currentIndexChanged.connect(self.get_device)
        self.combo_box.setFixedWidth(130)
        self.combo_box.addItem("自动选择")
        self.combo_box.addItem("GPU")
        self.combo_box.addItem("CPU")
        layout_4.addWidget(self.label_4)
        layout_4.addWidget(self.combo_box)

        layout_5 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_5_0 = QLabel('配置文件(可不选)')
        self.folder_label_5_0.setPalette(palette)
        self.folder_label_5_0.setFont(font_)
        self.folder_label_5 = QLabel()
        self.folder_label_5.setPalette(palette)
        self.folder_button_5 = QPushButton('...')
        self.folder_button_5.setFixedWidth(30)
        self.folder_button_5.clicked.connect(self.select_yaml)
        layout_5.addWidget(self.folder_label_5_0)
        layout_5.addWidget(self.folder_button_5)
        layout_5.addWidget(self.folder_label_5)

        layout_0.addLayout(layout_1)
        layout_0.addLayout(layout_2)
        layout_0.addLayout(layout_4)
        layout_0.addLayout(layout_5)
        layout_0.addLayout(layout_3)

    def get_dataset(self):
        self.dataset_name = self.combo_box_1.currentText()

    def get_device(self):
        self.device = self.combo_box.currentText()

    def train_exe(self):

        if os.path.exists('figs/log.png'):
            os.remove('figs/log.png')

        if self.dataset_name == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '数据集名称为空,请检查!')
            return
        if self.output_folder_path == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '权重文件输出路径为空,请检查!')
            return
        if self.device == "":
            self.device = '自动选择'
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '异常检查完成! 开始执行训练!')

        self.able_train(False)

        # print(type(self.ui.textBrowser))

        self.backend_thread_train = BackendThread_train(self.dataset_name, self.output_folder_path, self.device,
                                                        self.yaml_filename,
                                                        textBrowser=self.ui.textBrowser,
                                                        image_label=self.ui.image_label, able=self.able_train)
        self.backend_thread_train.start()

    def able_train(self, flag):
        if flag == False:
            self.train_button.setEnabled(False)
            self.combo_box_1.setEnabled(False)
            self.folder_button_2.setEnabled(False)
            self.combo_box.setEnabled(False)
            self.folder_button_5.setEnabled(False)
            self.ui.button_train.setEnabled(False)
            self.ui.button_test.setEnabled(False)
            self.ui.button_predict.setEnabled(False)
        else:
            self.train_button.setEnabled(True)
            self.combo_box_1.setEnabled(True)
            self.folder_button_2.setEnabled(True)
            self.combo_box.setEnabled(True)
            self.folder_button_5.setEnabled(True)
            self.ui.button_train.setEnabled(True)
            self.ui.button_test.setEnabled(True)
            self.ui.button_predict.setEnabled(True)

    def select_yaml(self):
        # 弹出文件选择对话框，获取用户选择的文件路径
        self.yaml_filename, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "yaml files (*.yaml)")
        if self.yaml_filename:
            yaml_name = basename(self.yaml_filename)
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '配置文件为：{}'.format(self.yaml_filename))
            self.folder_label_5.setText(f'配置文件为：{yaml_name}')

    def choose_output_folder(self):
        self.output_folder_path = QFileDialog.getExistingDirectory(None, '选择文件夹')
        if self.output_folder_path:
            folder_name = basename(self.output_folder_path)
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '权重文件输出文件夹为：{}'.format(self.output_folder_path))
            self.folder_label_2.setText(f'已选择文件夹：{folder_name}')

    def select_pt(self):
        # 弹出文件选择对话框，获取用户选择的文件路径
        self.pt_filename, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "pt files (*.pt)")
        if self.pt_filename:
            pt_name = basename(self.pt_filename)
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '配置文件为：{}'.format(self.pt_filename))
            self.folder_label_2_page2.setText(f'配置文件为：{pt_name}')

    def select_pt_page3(self):
        # 弹出文件选择对话框，获取用户选择的文件路径
        self.pt_filename, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "pt files (*.pt)")
        if self.pt_filename:
            pt_name = basename(self.pt_filename)
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '配置文件为：{}'.format(self.pt_filename))
            self.folder_label_4_page3.setText(f'配置文件为：{pt_name}')

    def page_2(self):
        self.page2 = QWidget()

        layout_0_page2 = QVBoxLayout(self.page2)
        self.label0_page2 = QLabel("测试模式")
        font = QFont("微软雅黑", 14)  # 创建一个12号微软雅黑字体
        self.label0_page2.setPalette(palette)
        self.label0_page2.setFont(font)
        layout_0_page2.addWidget(self.label0_page2, alignment=Qt.AlignCenter)

        layout_1_page2 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.label_1_page2 = QLabel("数据集")
        self.label_1_page2.setPalette(palette)
        self.label_1_page2.setFont(font_)
        self.combo_box_dataset_page2 = QComboBox()
        self.combo_box_dataset_page2.currentIndexChanged.connect(self.get_dataset)
        self.combo_box_dataset_page2.setFixedWidth(80)
        self.combo_box_dataset_page2.addItem("MNIST")
        self.combo_box_dataset_page2.addItem("FashionMNIST")
        layout_1_page2.addWidget(self.label_1_page2)
        layout_1_page2.addWidget(self.combo_box_dataset_page2)

        layout_2_page2 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_2_0_page2 = QLabel('权重文件路径')
        self.folder_label_2_0_page2.setPalette(palette)
        self.folder_label_2_0_page2.setFont(font_)
        self.folder_label_2_page2 = QLabel()
        self.folder_label_2_page2.setPalette(palette)
        self.folder_button_2_page2 = QPushButton('...')
        self.folder_button_2_page2.setFixedWidth(30)
        self.folder_button_2_page2.clicked.connect(self.select_pt)
        layout_2_page2.addWidget(self.folder_label_2_0_page2)
        layout_2_page2.addWidget(self.folder_button_2_page2)
        layout_2_page2.addWidget(self.folder_label_2_page2)

        layout_3_page2 = QHBoxLayout(alignment=Qt.AlignCenter)
        self.folder_label_3_page2 = QLabel()
        self.test_button = QPushButton('执行测试')
        self.test_button.setCursor(Qt.PointingHandCursor)
        self.test_button.setStyleSheet('''
            QPushButton {
            color: white;
            background-color: (39,42,71);
            border: 1px solid #dcdfe6;
            padding: 5px;
            border-radius: 5px;
        }

        QPushButton:hover {
            background-color: #ecf5ff;
            color: #409eff;
            cursor: pointer;
        }

        QPushButton:pressed, QPushButton:checked {
            border: 1px solid #3a8ee6;
            color: #409eff;
        }

        #button3 {
            border-radius: 20px;
        }

        ''')

        self.test_button.setPalette(palette)
        self.test_button.clicked.connect(self.test_exe)
        layout_3_page2.addWidget(self.test_button)

        layout_4_page2 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.label_4_page2 = QLabel("执行硬件")
        self.label_4_page2.setPalette(palette)
        self.label_4_page2.setFont(font_)
        self.combo_box_page2 = QComboBox()
        self.combo_box_page2.currentIndexChanged.connect(self.get_device)
        self.combo_box_page2.setFixedWidth(100)
        self.combo_box_page2.addItem("自动选择")
        self.combo_box_page2.addItem("GPU")
        self.combo_box_page2.addItem("CPU")
        layout_4_page2.addWidget(self.label_4_page2)
        layout_4_page2.addWidget(self.combo_box_page2)

        layout_5_page2 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_5_0_page2 = QLabel('配置文件(可不选)')
        self.folder_label_5_0_page2.setPalette(palette)
        self.folder_label_5_0_page2.setFont(font_)
        self.combo_box_page2 = QComboBox()
        self.folder_label_5_page2 = QLabel()
        self.folder_label_5_page2.setPalette(palette)
        self.folder_button_5_page2 = QPushButton('...')
        self.folder_button_5_page2.setFixedWidth(30)
        self.folder_button_5_page2.clicked.connect(self.select_yaml)
        layout_5_page2.addWidget(self.folder_label_5_0_page2)
        layout_5_page2.addWidget(self.folder_button_5_page2)
        layout_5_page2.addWidget(self.folder_label_5_page2)

        layout_0_page2.addLayout(layout_1_page2)
        layout_0_page2.addLayout(layout_2_page2)
        layout_0_page2.addLayout(layout_4_page2)
        layout_0_page2.addLayout(layout_5_page2)
        layout_0_page2.addLayout(layout_3_page2)

    def able_test(self, flag):
        if flag == False:
            self.train_button.setEnabled(False)
            self.combo_box_dataset_page2.setEnabled(False)
            self.folder_button_2.setEnabled(False)
            self.combo_box_page2.setEnabled(False)
            self.folder_button_5.setEnabled(False)
            self.ui.button_train.setEnabled(False)
            self.ui.button_test.setEnabled(False)
            self.ui.button_predict.setEnabled(False)
        else:
            self.train_button.setEnabled(True)
            self.combo_box_dataset_page2.setEnabled(True)
            self.folder_button_2.setEnabled(True)
            self.combo_box_page2.setEnabled(True)
            self.folder_button_5.setEnabled(True)
            self.ui.button_train.setEnabled(True)
            self.ui.button_test.setEnabled(True)
            self.ui.button_predict.setEnabled(True)

    def test_exe(self):
        if self.dataset_name == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '数据集名称空,请检查!')
            return
        if self.pt_filename == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '权重文件路径为空,请检查!')
            return
        if self.device == "":
            self.device = '自动选择'
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '异常检查完成! 开始执行测试!')

        self.able_test(False)

        self.backend_thread_train = BackendThread_test(self.dataset_name, self.pt_filename, self.device,
                                                       self.yaml_filename,
                                                       textBrowser=self.ui.textBrowser, able=self.able_test)
        self.backend_thread_train.start()

    def page_3(self):
        # print("预测")
        self.page3 = QWidget()

        layout_0_page3 = QVBoxLayout(self.page3)
        self.label0_page3 = QLabel("预测模式")
        font = QFont("微软雅黑", 14)  # 创建一个12号微软雅黑字体
        self.label0_page3.setPalette(palette)
        self.label0_page3.setFont(font)
        layout_0_page3.addWidget(self.label0_page3, alignment=Qt.AlignCenter)

        layout_2_page3 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_2_0_page3 = QLabel('预测图像路径')
        self.folder_label_2_0_page3.setPalette(palette)
        self.folder_label_2_0_page3.setFont(font_)
        self.folder_label_2_page3 = QLabel()
        self.folder_label_2_page3.setPalette(palette)
        self.folder_button_2_page3 = QPushButton('...')
        self.folder_button_2_page3.setFixedWidth(30)
        self.folder_button_2_page3.clicked.connect(self.select_fig)
        layout_2_page3.addWidget(self.folder_label_2_0_page3)
        layout_2_page3.addWidget(self.folder_button_2_page3)
        layout_2_page3.addWidget(self.folder_label_2_page3)

        layout_4_page3 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.folder_label_4_0_page3 = QLabel('权重文件路径')
        self.folder_label_4_0_page3.setPalette(palette)
        self.folder_label_4_0_page3.setFont(font_)
        self.folder_label_4_page3 = QLabel()
        self.folder_label_4_page3.setPalette(palette)
        self.folder_button_4_page3 = QPushButton('...')
        self.folder_button_4_page3.setFixedWidth(30)
        self.folder_button_4_page3.clicked.connect(self.select_pt_page3)
        layout_4_page3.addWidget(self.folder_label_4_0_page3)
        layout_4_page3.addWidget(self.folder_button_4_page3)
        layout_4_page3.addWidget(self.folder_label_4_page3)

        layout_3_page3 = QHBoxLayout(alignment=Qt.AlignCenter)
        # self.folder_label_3_page3 = QLabel()
        self.predict_button = QPushButton('执行预测')
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.setStyleSheet('''
            QPushButton {
            color: white;
            background-color: (39,42,71);
            border: 1px solid #dcdfe6;
            padding: 5px;
            border-radius: 5px;
        }
        
        QPushButton:hover {
            background-color: #ecf5ff;
            color: #409eff;
            cursor: pointer;
        }
        
        QPushButton:pressed, QPushButton:checked {
            border: 1px solid #3a8ee6;
            color: #409eff;
        }
        
        #button3 {
            border-radius: 20px;
        }

        ''')

        self.predict_button.setPalette(palette)
        self.predict_button.clicked.connect(self.predict_exe)
        layout_3_page3.addWidget(self.predict_button)

        layout_5_page3 = QHBoxLayout(alignment=Qt.AlignLeft)
        self.label_5_page3 = QLabel("执行硬件")
        self.label_5_page3.setPalette(palette)
        self.label_5_page3.setFont(font_)
        self.combo_box_page3 = QComboBox()
        self.combo_box_page3.currentIndexChanged.connect(self.get_device)
        self.combo_box_page3.setFixedWidth(100)
        self.combo_box_page3.addItem("自动选择")
        self.combo_box_page3.addItem("GPU")
        self.combo_box_page3.addItem("CPU")
        layout_5_page3.addWidget(self.label_5_page3)
        layout_5_page3.addWidget(self.combo_box_page3)

        layout_0_page3.addLayout(layout_2_page3)
        layout_0_page3.addLayout(layout_4_page3)
        layout_0_page3.addLayout(layout_5_page3)
        layout_0_page3.addLayout(layout_3_page3)

    def select_fig(self):
        # 弹出文件选择对话框，获取用户选择的文件路径
        self.fig_filename, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "jpg files (*.jpg)")
        if self.fig_filename:
            fig_name = basename(self.fig_filename)
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '配置文件为：{}'.format(self.fig_filename))
            self.folder_label_2_page3.setText(f'配置文件为：{fig_name}')

    def predict_exe(self):
        print("预测")
        if self.fig_filename == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '图像路径为空,请检查!')
            return
        if self.pt_filename == "":
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            self.ui.textBrowser.append(ttt + '权重文件路径为空,请检查!')
            return
        if self.device == "":
            self.device = '自动选择'
        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '异常检查完成! 开始执行测试!')

        self.backend_thread_predict = BackendThread_predict(self.fig_filename, self.pt_filename, self.ui.textBrowser,
                                                            image_label=self.ui.image_label)
        self.backend_thread_predict.start()

    def train(self):

        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '已选择训练功能')

        self.ui.textBrowser.ensureCursorVisible()  # 游标可用
        cursor = self.ui.textBrowser.textCursor()  # 设置游标
        pos = len(self.ui.textBrowser.toPlainText())  # 获取文本尾部的位置
        cursor.setPosition(pos)  # 游标位置设置为尾部
        self.ui.textBrowser.setTextCursor(cursor)  # 滚动到游标位置

        self.page0.setVisible(False)
        self.page1.setVisible(True)
        self.page2.setVisible(False)
        self.page3.setVisible(False)
        print("train")

    def test(self):

        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '已选择测试功能')

        self.ui.textBrowser.ensureCursorVisible()  # 游标可用
        cursor = self.ui.textBrowser.textCursor()  # 设置游标
        pos = len(self.ui.textBrowser.toPlainText())  # 获取文本尾部的位置
        cursor.setPosition(pos)  # 游标位置设置为尾部
        self.ui.textBrowser.setTextCursor(cursor)  # 滚动到游标位置

        self.page0.setVisible(False)
        self.page1.setVisible(False)
        self.page2.setVisible(True)
        self.page3.setVisible(False)
        print("test")

    def predict(self):

        ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
        self.ui.textBrowser.append(ttt + '已选择预测功能')

        self.ui.textBrowser.ensureCursorVisible()  # 游标可用
        cursor = self.ui.textBrowser.textCursor()  # 设置游标
        pos = len(self.ui.textBrowser.toPlainText())  # 获取文本尾部的位置
        cursor.setPosition(pos)  # 游标位置设置为尾部
        self.ui.textBrowser.setTextCursor(cursor)  # 滚动到游标位置

        self.page0.setVisible(False)
        self.page1.setVisible(False)
        self.page2.setVisible(False)
        self.page3.setVisible(True)
        print("predict")


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
