import sys
import os

import pyqtgraph as pg
import numpy as np

from zro import RemoteObject

REPLY_PORT = 12000


class Visualizer(pg.QtGui.QMainWindow, RemoteObject):
    """docstring for Visualizer"""
    def __init__(self):
        super(Visualizer, self).__init__()
        RemoteObject.__init__(self, rep_port=REPLY_PORT)

        self._setup_ui()

        self.data_file = None
        self.buffer = np.zeros((32,32,3), dtype=np.uint8)

    def set_data_file(self, path):
        self.release_file()
        try:
            self.data_file = open(path, 'rb')
        except IOError as e:
            self.data_file = path

    def release_file(self):
        if self.data_file:
            if not isinstance(self.data_file, str):
                self.data_file.close()
                self.data_file = None

    def _setup_ui(self):
        self.resize(800,600)
        self.cw = pg.QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.layout = pg.QtGui.QVBoxLayout()
        self.cw.setLayout(self.layout)

        self.label_bits = pg.QtGui.QLabel(text="0"*32)
        self.layout.addWidget(self.label_bits)

        self.gw = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.gw)

        self.vb = self.gw.addPlot()
        self.vb.showAxis('top', True)
        self.vb.showAxis('left', False)

        self.img = pg.ImageItem()
        self.vb.addItem(self.img)

        self.update_timer = pg.QtCore.QTimer()
        self.update_timer.timeout.connect(self._update)
        self.update_timer.start(100)

        self.setWindowTitle("Visualizer")
        self.icon = pg.QtGui.QIcon()
        self.icon.addFile("res/visualizer.png", pg.QtCore.QSize(16,16))
        self.setWindowIcon(self.icon)

        with open("res/darkorange_stylesheet.css", 'r') as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)

    def _update(self):
        try:
            self._update_buffer()
        except IOError as e:
            logging.warning("IOError: {}".format(e))
        self._check_rep()

    def _update_buffer(self):
        if not self.data_file:
            return
        if isinstance(self.data_file, str):
            self.set_data_file(self.data_file)
            return
        self.data_file.seek(-4,2)  # most recent data
        data = np.fromfile(self.data_file, dtype=np.uint32, count=1)
        binary = np.binary_repr(data, width=32)
        row = np.array(list(binary)[::-1], dtype=np.uint8)
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[:,-1,2] = row
        self.label_bits.setText(binary)
        self.img.setImage(self.buffer)

    def closeEvent(self, evnt):
        self.release_file()

        


if __name__ == '__main__':

    import sys
    import logging
    logging.basicConfig(level=logging.INFO)

    app = pg.QtGui.QApplication(sys.argv)

    visualizer = Visualizer()
    visualizer.show()

    app.exec_()
