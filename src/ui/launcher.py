from PyQt5.QtWidgets import QInputDialog, QMessageBox, QFormLayout, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QFileDialog

class Launcher(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("ORPose Correction Tool")
        self.resize(800, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(QLabel("ORPose Correction Tool"))
        actions_layout = QFormLayout()
        layout.addLayout(actions_layout)
        self.add_create_section(actions_layout)
        self.add_import_section(actions_layout)
        self.add_open_section(actions_layout)

    def add_create_section(self, actions_layout):
        video_select_button = QPushButton("Select Videos")
        self.selected_videos = []
        video_select_button.clicked.connect(self.handle_video_selection)
        create_button = QPushButton("Create new labeling from videos")
        create_button.clicked.connect(self.handle_create_labeling)
        actions_layout.addRow(video_select_button, create_button)

    def add_import_section(self, actions_layout):
        labeling_select_button = QPushButton("Select labeling files")
        self.selected_labelings = []
        labeling_select_button.clicked.connect(self.handle_labeling_selection)
        import_button = QPushButton("Import labeling for correction")
        import_button.clicked.connect(self.handle_import_labeling)
        actions_layout.addRow(labeling_select_button, import_button)

    def add_open_section(self, actions_layout):
        self.project_selector = QComboBox()
        self.projects = self.app.get_project_names()
        self.project_selector.addItems(self.projects)
        self.selected_project_idx = 0
        self.project_selector.currentIndexChanged.connect(self.handle_project_selection)
        self.open = QPushButton("Open...")
        self.open.clicked.connect(self.handle_open)
        actions_layout.addRow(self.project_selector, self.open)

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
        self.selected_labelings = [None for _ in range(persons)]
        self.app.create_project(name, self.selected_videos, self.selected_labelings)
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
        self.app.create_project(name, self.selected_videos, self.selected_labelings)
        self.hide()
        self.app.open_project(name)

    def handle_project_selection(self, index):
        self.selected_project_idx = index

    def handle_open(self):
        self.hide()
        self.app.open_project(self.projects[self.selected_project_idx])
