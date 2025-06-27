from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt
from ui.toolbar import Toolbar
from ui.canvas import Canvas

class ProjectWindow(QMainWindow):
    """
    Main window of the application.

    It contains the following attributes:
        - toolbar: Toolbar instance
        - canvas: Canvas instance
    """
    def __init__(self, project):
        super().__init__()
        self.resize(1520, 900)
        self.toolbar = Toolbar(project, parent=self)
        self.addToolBar(self.toolbar)
        self.canvas = Canvas(project)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("Correct Pose")
