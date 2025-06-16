from enum import Enum
from PyQt5.QtWidgets import QToolBar, QLabel, QComboBox, QAction, QActionGroup
from PyQt5.QtGui import QIcon

class Mode(Enum):
    PLACE = 0
    PAN = 1
    ZOOM = 2

    def toggle_in(self, nav_toolbar):
        if self == Mode.PAN:
            nav_toolbar.pan()
        elif self == Mode.ZOOM:
            nav_toolbar.zoom()

class Toolbar(QToolBar):

    visibility_icons = {
        True: QIcon("icon/hidden.svg"),
        False: QIcon("icon/shown.svg")
    }

    def change_mode(self, mode):
        self.mode.toggle_in(self.app.window.canvas.nav_toolbar)
        self.mode = mode
        self.mode.toggle_in(self.app.window.canvas.nav_toolbar)

    def __init__(self, app, parent=None):
        super().__init__(parent)

        self.app = app

        self.addWidget(QLabel("Camera: "))
        camera_selector = QComboBox(parent)
        camera_selector.addItems(self.app.dataset.poses.keys())
        camera_selector.currentIndexChanged.connect(self.app.change_camera)
        self.addWidget(camera_selector)

        self.addWidget(QLabel(" Hand: "))
        hand_selector = QComboBox(parent)
        hand_selector.addItems(["left", "right"])
        hand_selector.currentIndexChanged.connect(self.app.change_hand)
        self.addWidget(hand_selector)

        flip_sides = QAction(QIcon("icon/flip.svg"), "flips hand side", parent)
        flip_sides.setStatusTip("Flips the data from one side to the other")
        flip_sides.triggered.connect(self.app.flip_hand_side)
        self.addAction(flip_sides)

        self.addSeparator()

        start_button = QAction(QIcon("icon/start.svg"), "start", parent)
        start_button.setStatusTip("Go to first frame")
        start_button.triggered.connect(self.app.goto_first_frame)
        start_button.setShortcut("Shift+W")
        self.addAction(start_button)

        prev_button = QAction(QIcon("icon/prev.svg"), "prev", parent)
        prev_button.setStatusTip(f"Go back {self.app.frame_step} frames")
        prev_button.triggered.connect(self.app.prev_frame)
        prev_button.setShortcut("S")
        self.addAction(prev_button)
        next_button = QAction(QIcon("icon/next.svg"), "next", parent)
        next_button.setStatusTip(f"Advance {self.app.frame_step} frames")
        next_button.triggered.connect(self.app.next_frame)
        next_button.setShortcut("W")
        self.addAction(next_button)

        end_button = QAction(QIcon("icon/end.svg"), "end", parent)
        end_button.setStatusTip(f"Go to last frame")
        end_button.triggered.connect(self.app.goto_last_frame)
        end_button.setShortcut("Shift+S")
        self.addAction(end_button)

        self.addSeparator()

        home_button = QAction(QIcon("icon/home.svg"), "home", parent)
        home_button.setStatusTip("Reset original view")
        home_button.triggered.connect(self.app.home_view)
        home_button.setShortcut("Esc")
        self.addAction(home_button)

        self.tool_group = QActionGroup(parent)
        self.tool_group.setExclusive(True)

        place_button = QAction(QIcon("icon/place.svg"), "place", parent)
        place_button.setStatusTip("Place tool")
        place_button.setCheckable(True)
        place_button.setChecked(True)
        self.mode = Mode.PLACE
        place_button.triggered.connect(lambda: self.change_mode(Mode.PLACE))
        place_button.setShortcut("1")
        self.tool_group.addAction(place_button)
        self.addAction(place_button)

        pan_button = QAction(QIcon("icon/pan.svg"), "pan", parent)
        pan_button.setStatusTip("Pan tool")
        pan_button.setCheckable(True)
        pan_button.triggered.connect(lambda: self.change_mode(Mode.PAN))
        pan_button.setShortcut("2")
        self.tool_group.addAction(pan_button)
        self.addAction(pan_button)

        zoom_button = QAction(QIcon("icon/zoom.svg"), "zoom", parent)
        zoom_button.setStatusTip("Zoom tool")
        zoom_button.setCheckable(True)
        zoom_button.triggered.connect(lambda: self.change_mode(Mode.ZOOM))
        zoom_button.setShortcut("3")
        self.tool_group.addAction(zoom_button)
        self.addAction(zoom_button)

        self.addSeparator()

        self.addWidget(QLabel("Advance Keypoint: "))

        advance_selector = QComboBox(parent)
        advance_selector.addItems(["-1","0","1"])
        advance_selector.setCurrentIndex(1)
        advance_selector.currentIndexChanged.connect(self.app.set_keypoint_advance)
        self.addWidget(advance_selector)

        next_keypoint_button = QAction(QIcon("icon/next.svg"), "next", parent)
        next_keypoint_button.setStatusTip("Next keypoint")
        next_keypoint_button.triggered.connect(self.app.next_keypoint)
        next_keypoint_button.setShortcut("D")
        self.addAction(next_keypoint_button)

        previous_keypoint_button = QAction(QIcon("icon/previous.svg"), "previous", parent)
        previous_keypoint_button.setStatusTip("Previous keypoint")
        previous_keypoint_button.triggered.connect(self.app.prev_keypoint)
        previous_keypoint_button.setShortcut("A")
        self.addAction(previous_keypoint_button)

        self.visibility_button = QAction(self.visibility_icons[self.app.keypoints_hidden], "toggle keypoint visibility", parent)
        self.visibility_button.setStatusTip("Toggle keypoint visibility")
        self.visibility_button.triggered.connect(self.app.toggle_keypoint_visibility)
        self.visibility_button.setShortcut("Space")
        self.addAction(self.visibility_button)

        delete_keypoint_button = QAction(QIcon("icon/delete.svg"), "delete", parent)
        delete_keypoint_button.setStatusTip("Delete keypoint")
        delete_keypoint_button.triggered.connect(self.app.delete_keypoint)
        delete_keypoint_button.setShortcut("R")
        self.addAction(delete_keypoint_button)

        self.addSeparator()

        save_button = QAction(QIcon("icon/save.svg"), "save", parent)
        save_button.setStatusTip("Save poses")
        save_button.triggered.connect(self.app.save)
        save_button.setShortcut("Ctrl+S")
        self.addAction(save_button)
