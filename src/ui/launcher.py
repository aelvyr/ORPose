from PyQt5.QtWidgets import QInputDialog, QMessageBox, QFormLayout, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QFileDialog
import shutil
from pathlib import Path

class Launcher(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("ORPose Correction Tool")
        self.resize(800, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.addWidget(QLabel("ORPose Correction Tool"))
        self.actions_layout = QFormLayout()
        self.layout.addLayout(self.actions_layout)
        self.video_select_button = QPushButton("Select Videos")
        self.selected_videos = []
        self.video_select_button.clicked.connect(self.handle_video_selection)
        self.create_button = QPushButton("Create new labeling from videos")
        self.create_button.clicked.connect(self.handle_create_labeling)
        self.actions_layout.addRow(self.video_select_button, self.create_button)
        self.labeling_select_button = QPushButton("Select labeling files")
        self.selected_labelings = []
        self.labeling_select_button.clicked.connect(self.handle_labeling_selection)
        self.import_button = QPushButton("Import labeling for correction")
        self.import_button.clicked.connect(self.handle_import_labeling)
        self.actions_layout.addRow(self.labeling_select_button, self.import_button)
        self.project_selector = QComboBox()
        self.projects = self.app.get_project_names()
        self.project_selector.addItems(self.projects)
        self.project_selector.currentIndexChanged.connect(self.handle_project_selection)
        self.open = QPushButton("Open...")
        self.open.clicked.connect(self.handle_open)
        self.actions_layout.addRow(self.project_selector, self.open)

    def handle_video_selection(self):
        selected_videos, ok = QFileDialog.getOpenFileNames(self, "Select Videos", "", "<camera_name>.mp4 (*.mp4)")
        if ok:
            self.selected_videos = selected_videos

    def handle_create_labeling(self):
        if self.selected_videos == []:
            QMessageBox.warning(self, "No videos selected", "Please select at least one video.")
            return
        name, ok = QInputDialog.getText(self, "Create Labeling", "Enter the dataset name:")
        if not ok:
            return
        persons, ok = QInputDialog.getInt(self, "Create Labeling", "Enter the number of persons:")
        if not ok or persons <= 1:
            persons = 1
        (Path("inputs") / name).mkdir(parents=True, exist_ok=True)
        for file in self.selected_videos:
            file = Path(file)
            shutil.copy(file, Path("inputs") / name / file.name)
        for person in range(persons):
            (Path("output_3d") / f"{name}_{person}").mkdir(parents=True, exist_ok=True)
        self.hide()
        self.app.open_project(name)

    def handle_labeling_selection(self):
        selected_labelings, ok = QFileDialog.getOpenFileNames(self, "Select Labelings", "", "Per Person Dataset (*.npz)")
        if ok:
            self.selected_labelings = selected_labelings

    def handle_import_labeling(self):
        if self.selected_videos == []:
            QMessageBox.warning(self, "No videos selected", "Please select at least one video.")
            return
        if self.selected_labelings == []:
            QMessageBox.warning(self, "No labelings selected", "Please select at least one labeling.")
            return
        name, ok = QInputDialog.getText(self, "Import Labeling", "Enter the dataset name:")
        if not ok:
            return
        (Path("inputs") / name).mkdir(parents=True, exist_ok=True)
        for file in self.selected_videos:
            file = Path(file)
            shutil.copy(file, Path("inputs") / name / file.name)
        for person, file in enumerate(self.selected_labelings):
            (Path("output_3d") / f"{name}_{person}").mkdir(parents=True, exist_ok=True)
            shutil.copy(file, Path("output_3d") / f"{name}_{person}" / "hand_poses_2d.npz")
        self.hide()
        self.app.open_project(name)

    def handle_project_selection(self, index):
        self.selected_project_idx = index

    def handle_open(self):
        self.hide()
        self.app.open_project(self.projects[self.selected_project_idx])
