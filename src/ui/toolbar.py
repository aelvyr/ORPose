from enum import Enum
from PyQt5.QtWidgets import QToolBar, QLabel, QComboBox, QAction, QActionGroup, QSlider, QInputDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class Mode(Enum):
    """
    Enum representing the different modes of the toolbar.
    """
    PLACE = 0
    PAN = 1
    ZOOM = 2

    def toggle_in(self, nav_toolbar):
        """
        Toggles the tools in the backing matplotlib toolbar.
        This is used to ensure consistency between the two toolbars.
        """
        if self == Mode.PAN:
            nav_toolbar.pan()
        elif self == Mode.ZOOM:
            nav_toolbar.zoom()

class Toolbar(QToolBar):
    """
    This is the custom toolbar for the application.
    """

    visibility_icons = {
        True: QIcon("icon/hidden.svg"),
        False: QIcon("icon/shown.svg")
    }

    hand_options = ["left", "right"]

    auto_advance_options = ["-1", "0", "1"]

    def __init__(self, project, parent=None):
        """
        Initializes the toolbar with the given project and parent widget.

        Args:
            project (Project): The project instance.
        """
        super().__init__(parent)

        self.project = project
        self.add_data_options()
        self.addSeparator()
        self.add_media_controls()
        self.addSeparator()
        self.add_canvas_tools()
        self.addSeparator()
        self.add_keypoint_controls()

    def add_data_options(self):
        self.add_label("Data Options")
        self.add_camera_selector()
        self.add_person_selector()
        self.add_person_flip_button()
        self.add_hand_selector()
        self.add_flip_side_button()
        self.add_save_button()

    def add_media_controls(self):
        self.add_label("Media Controls")
        self.add_start_button()
        self.add_prev_button()
        self.add_next_button()
        self.add_end_button()

    def add_canvas_tools(self):
        self.add_label("Canvas Tools")
        self.add_reset_view_button()
        self.tool_group = QActionGroup(self.parentWidget())
        self.tool_group.setExclusive(True)
        self.add_place_button(self.tool_group)
        self.add_pan_button(self.tool_group)
        self.add_zoom_button(self.tool_group)

    def add_keypoint_controls(self):
        self.add_label("Keypoint Controls")
        self.add_previous_keypoint_button()
        self.add_next_keypoint_button()
        self.add_radius_selector()
        self.add_auto_advance_selector()
        self.add_visibility_button()
        self.add_delete_button()

    def add_camera_selector(self):
        self.add_label("Camera:")
        self.camera_selector = QComboBox(self.parentWidget())
        self.camera_selector.addItems(self.project.cameras.data)
        self.camera_selector.currentIndexChanged.connect(self.project.change_camera)
        self.addWidget(self.camera_selector)

    def add_person_selector(self):
        if (self.project.dataset.persons <= 1):
            return
        self.add_label("Person:")
        self.person_selector = QComboBox(self.parentWidget())
        for person in range(self.project.dataset.persons):
            self.person_selector.addItem(str(person))
        self.person_selector.currentIndexChanged.connect(self.project.change_person)
        self.addWidget(self.person_selector)

    def add_person_flip_button(self):
        if (self.project.dataset.persons <= 1):
            return
        self.person_flip_button = QAction(QIcon("icon/flip.svg"), "flips person", self.parentWidget())
        self.person_flip_button.setStatusTip("Flips the data from one person to the other")
        self.person_flip_button.triggered.connect(self.swap_people)
        self.addAction(self.person_flip_button)

    def add_hand_selector(self):
        self.add_label("Hand:")
        self.hand_selector = QComboBox(self.parentWidget())
        self.hand_selector.addItems(self.hand_options)
        self.hand_selector.currentIndexChanged.connect(self.project.change_hand)
        self.addWidget(self.hand_selector)

    def add_flip_side_button(self):
        self.flip_sides_button = QAction(QIcon("icon/flip.svg"), "flips hand side", self.parentWidget())
        self.flip_sides_button.setStatusTip("Flips the data from one side to the other")
        self.flip_sides_button.triggered.connect(self.project.flip_hand_side)
        self.addAction(self.flip_sides_button)

    def add_start_button(self):
        self.start_button = QAction(QIcon("icon/start.svg"), "start", self.parentWidget())
        self.start_button.setStatusTip("Go to first frame")
        self.start_button.triggered.connect(self.project.goto_first_frame)
        self.start_button.setShortcut("Shift+A")
        self.addAction(self.start_button)

    def add_prev_button(self):
        self.prev_button = QAction(QIcon("icon/prev.svg"), "prev", self.parentWidget())
        self.prev_button.setStatusTip(f"Go back {self.project.frame_step} frames")
        self.prev_button.triggered.connect(self.project.prev_frame)
        self.prev_button.setShortcut("A")
        self.addAction(self.prev_button)

    def add_next_button(self):
        self.next_button = QAction(QIcon("icon/next.svg"), "next", self.parentWidget())
        self.next_button.setStatusTip(f"Go forward {self.project.frame_step} frames")
        self.next_button.triggered.connect(self.project.next_frame)
        self.next_button.setShortcut("D")
        self.addAction(self.next_button)

    def add_end_button(self):
        self.end_button = QAction(QIcon("icon/end.svg"), "end", self.parentWidget())
        self.end_button.setStatusTip("Go to last frame")
        self.end_button.triggered.connect(self.project.goto_last_frame)
        self.end_button.setShortcut("Shift+D")
        self.addAction(self.end_button)

    def add_reset_view_button(self):
        self.reset_view_button = QAction(QIcon("icon/home.svg"), "reset view", self.parentWidget())
        self.reset_view_button.setStatusTip("Reset view")
        self.reset_view_button.triggered.connect(lambda: self.project.window.canvas.reset_view())
        self.reset_view_button.setShortcut("Escape")
        self.addAction(self.reset_view_button)

    def add_place_button(self, group):
        self.place_button = QAction(QIcon("icon/place.svg"), "place", self.parentWidget())
        self.place_button.setStatusTip("Place tool")
        self.place_button.setCheckable(True)
        self.place_button.setChecked(True)
        self.mode = Mode.PLACE
        self.place_button.triggered.connect(lambda: self.change_mode(Mode.PLACE))
        self.place_button.setShortcut("1")
        group.addAction(self.place_button)
        self.addAction(self.place_button)

    def add_pan_button(self, group):
        self.pan_button = QAction(QIcon("icon/pan.svg"), "pan", self.parentWidget())
        self.pan_button.setStatusTip("Pan tool")
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(False)
        self.pan_button.triggered.connect(lambda: self.change_mode(Mode.PAN))
        self.pan_button.setShortcut("2")
        group.addAction(self.pan_button)
        self.addAction(self.pan_button)

    def add_zoom_button(self, group):
        self.zoom_button = QAction(QIcon("icon/zoom.svg"), "zoom", self.parentWidget())
        self.zoom_button.setStatusTip("Zoom tool")
        self.zoom_button.setCheckable(True)
        self.zoom_button.setChecked(False)
        self.zoom_button.triggered.connect(lambda: self.change_mode(Mode.ZOOM))
        self.zoom_button.setShortcut("3")
        group.addAction(self.zoom_button)
        self.addAction(self.zoom_button)

    def add_previous_keypoint_button(self):
        self.previous_keypoint_button = QAction(QIcon("icon/prev.svg"), "previous", self.parentWidget())
        self.previous_keypoint_button.setStatusTip("Previous keypoint")
        self.previous_keypoint_button.triggered.connect(self.project.prev_keypoint)
        self.previous_keypoint_button.setShortcut("S")
        self.addAction(self.previous_keypoint_button)

    def add_next_keypoint_button(self):
        self.next_keypoint_button = QAction(QIcon("icon/next.svg"), "next", self.parentWidget())
        self.next_keypoint_button.setStatusTip("Next keypoint")
        self.next_keypoint_button.triggered.connect(self.project.next_keypoint)
        self.next_keypoint_button.setShortcut("W")
        self.addAction(self.next_keypoint_button)

    def add_radius_selector(self):
        self.add_label("Radius:")
        self.radius_selector = QSlider(Qt.Horizontal)
        self.radius_selector.setMinimum(10)
        self.radius_selector.setMaximum(100)  # Represents 0.0 to 10.0 if step is 0.01
        self.radius_selector.setValue(50)
        self.radius_selector.setMinimumWidth(100)
        self.radius_selector.setMaximumWidth(200)
        self.radius_selector.valueChanged.connect(self.project.resize_keypoints)
        self.addWidget(self.radius_selector)

    def add_auto_advance_selector(self):
        self.add_label("Auto Advance:")
        self.auto_advance_selector = QComboBox(self.parentWidget())
        self.auto_advance_selector.addItems(self.auto_advance_options)
        self.auto_advance_selector.setCurrentIndex(self.auto_advance_options.index("0"))
        self.auto_advance_selector.currentIndexChanged.connect(self.set_keypoint_advance)
        self.addWidget(self.auto_advance_selector)

    def add_visibility_button(self):
        self.visibility_button = QAction(self.visibility_icons[self.project.keypoints_hidden], "toggle keypoint visibility", self.parentWidget())
        self.visibility_button.setStatusTip("Toggle keypoint visibility")
        self.visibility_button.triggered.connect(self.project.toggle_keypoint_visibility)
        self.visibility_button.setShortcut("Space")
        self.addAction(self.visibility_button)

    def add_delete_button(self):
        self.delete_keypoint_button = QAction(QIcon("icon/delete.svg"), "delete", self.parentWidget())
        self.delete_keypoint_button.setStatusTip("Delete keypoint")
        self.delete_keypoint_button.triggered.connect(self.project.delete_keypoint)
        self.delete_keypoint_button.setShortcut("R")
        self.addAction(self.delete_keypoint_button)

    def add_save_button(self):
        self.save_button = QAction(QIcon("icon/save.svg"), "save", self.parentWidget())
        self.save_button.setStatusTip("Save keypoints")
        self.save_button.setShortcut("Ctrl+S")
        self.save_button.triggered.connect(self.project.save)
        self.addAction(self.save_button)

    def add_label(self, text):
        label = QLabel(text)
        label.setMargin(5)
        self.addWidget(label)

    def swap_people(self, other):
        other, ok = QInputDialog.getInt(self, "Swap People", "Enter the ID of the person to swap with:", min=0, max=self.project.dataset.persons-1)
        if ok:
            self.project.swap_people(other)

    def change_mode(self, mode):
        """
        Changes the mode of the toolbar to a new mode.

        Args:
            mode (Mode): The new mode to set.
        """
        self.mode.toggle_in(self.project.window.canvas.nav_toolbar)
        self.mode = mode
        self.mode.toggle_in(self.project.window.canvas.nav_toolbar)

    def set_keypoint_advance(self, idx):
        """
        Sets the amount the keypoint advances when a keypoint is placed based on the index into the list of advance levels.
        """
        self.project.keypoint_advance = int(self.auto_advance_options[idx])
