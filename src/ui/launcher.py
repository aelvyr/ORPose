from PyQt5.QtWidgets import QInputDialog, QMessageBox, QFormLayout, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QFileDialog

class Launcher(QMainWindow):
    """
    The Launcher Window for ORPose Correction Tool
    """
    def __init__(self, app):
        """
        Initialize the Launcher Window.

        Args:
            app (Application): The Application instance.
        """
        super().__init__()
        self.app = app
        self.setWindowTitle("ORPose Correction Tool")
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

        # ✅ Fotos (images)
        photo_select_button = QPushButton("Select Fotos")
        self.selected_photos = []
        photo_select_button.clicked.connect(self.handle_photo_selection)

        create_photo_button = QPushButton("Create new labeling from Fotos")
        create_photo_button.clicked.connect(self.handle_create_labeling_from_photos)
        actions_layout.addRow(photo_select_button, create_photo_button)

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
        """
        Opens a file dialog to select videos.

        Only intended for the behavior implementation of the Import button.
        """
        selected_videos, ok = QFileDialog.getOpenFileNames(self, "Select Videos", "", "<camera_name>.mp4 (*.mp4)")
        if ok:
            self.selected_videos = selected_videos

    # ✅ New: image selection handler
    def handle_photo_selection(self):
        selected_photos, ok = QFileDialog.getOpenFileNames(
            self,
            "Select Fotos",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if ok:
            self.selected_photos = selected_photos

    def handle_create_labeling(self):
        """
        Creates a new labeling project from videos only.

        Only intended for the behavior implementation of the Create Labeling button.
        """
        if self.selected_videos == []:
            QMessageBox.warning(self, "No videos selected", "Please select at least one video.")
            return
        name, ok = QInputDialog.getText(self, "Create Labeling", "Enter the dataset name:")
        if not ok:
            return
        persons, ok = QInputDialog.getInt(self, "Create Labeling", "Enter the number of persons:", min=1)
        if not ok or persons <= 1:
            persons = 1
        self.selected_labelings = [None for _ in range(persons)]
        self.app.create_project(name, self.selected_videos, self.selected_labelings)
        self.hide()
        self.app.open_project(name)

    def handle_create_labeling_from_photos(self):
        if self.selected_photos == []:
            QMessageBox.warning(self, "No photos selected", "Please select at least one image.")
            return
        name, ok = QInputDialog.getText(self, "Create Labeling from Fotos", "Enter the dataset name:")
        if not ok:
            return
        persons, ok = QInputDialog.getInt(self, "Create Labeling from Fotos", "Enter the number of persons:", min=1)
        if not ok or persons <= 1:
            persons = 1
        self.selected_labelings = [None for _ in range(persons)]
        self.app.create_project_fotos(name, self.selected_photos, self.selected_labelings)
        self.hide()
        self.app.open_project(name)

        
    def handle_labeling_selection(self):
        """
        Opens a file dialog to select labeling files.

        Only intended for the behavior implementation of the Select Labeling button.
        """
        selected_labelings, ok = QFileDialog.getOpenFileNames(self, "Select Labelings", "", "Per Person Dataset (*.npz)")
        if ok:
            self.selected_labelings = selected_labelings

    def handle_import_labeling(self):
        """
        Imports a new labeling project with the selected videos and labeling files.

        Only intended for the behavior implementation of the import button.
        """
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
        """
        Handles the selection of a project.

        Only intended for the behavior implementation of the project selection.
        """
        self.selected_project_idx = index

    def handle_open(self):
        """
        Opens the selected project.

        Only intended for the behavior implementation of the open button.
        """
        self.hide()
        self.app.open_project(self.projects[self.selected_project_idx])
