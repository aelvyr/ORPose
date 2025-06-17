from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt
from ui.toolbar import Toolbar
from ui.canvas import Canvas

class Window(QMainWindow):
    """
    Main window of the application.
    """
    def __init__(self, app):
        super().__init__()
        self.toolbar = Toolbar(app, parent=self)
        self.addToolBar(self.toolbar)
        self.canvas = Canvas(app)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("Correct Pose")
        self.show()
