from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ui.keypoint import Keypoints, Keypoint
from ui.toolbar import Mode

class KeypointPicker(Keypoints):
    """
    This class is responsible for the keypoint picker UI.
    """
    positions = [
        (0,0),
        (-2,1), (-3,3), (-3.5,4), (-4,5),
        (-1,4), (-1.5,5), (-2,6), (-2.5,7),
        (0,4), (0,5.25), (0,6.5), (0,7.75),
        (1,4), (1.25,5), (1.5,6), (1.75,7),
        (2,3.5), (2.5,4.5), (3,5.5), (3.5,6.5)
    ]
    def __init__(self, app, axes):
        """
        Initialize the keypoint picker UI.
        """
        super().__init__(app, axes, self.positions, radius=0.2, pickable=True)
        self.axes.set_xlim(-5,5)
        self.axes.set_ylim(-1,9)
        self.axes.set_aspect('equal')
        self.axes.get_figure().canvas.mpl_connect('pick_event', self.on_pick)
        self.draw()

    def on_pick(self, event):
        """
        This method handles the logic when someone clicks on a keypoint and thus selects it.
        """
        if event.mouseevent.inaxes != self.axes or not isinstance(event.artist, Keypoint):
            return
        self.app.set_current_keypoint(event.artist.index)

class Viewport(Keypoints):
    """
    This class is responsible for rendering the current frame and its associated keypoints.
    """
    def __init__(self, app, axes):
        """
        Initializes the Viewport object.
        """
        super().__init__(app, axes, app.dataset.get_pose(app.current_camera, app.current_hand).get_positions(), pickable=False)
        self.axes.get_figure().canvas.mpl_connect('button_press_event', self.on_click)
        self.img_display = self.axes.imshow(self.app.current_camera.get_current_frame())
        self.render_current_frame()

    def should_keypoint_render(self, keypoint: Keypoint):
        """
        Determines whether a keypoint should be rendered.
        """
        if self.app.keypoints_hidden:
            return False
        return super().should_keypoint_render(keypoint) and self.app.dataset.get_pose(self.app.current_camera, self.app.current_hand).is_keypoint_drawable(keypoint.index)

    def draw(self):
        """
        (Re)Draws the keypoints.
        """
        self.positions = self.app.dataset.get_pose(self.app.current_camera, self.app.current_hand).get_positions()
        super().draw()

    def render_current_frame(self):
        """
        Renders the current frame and (re)draws the keypoints.
        """
        print(f"rendering frame {self.app.current_camera.current_frame_idx}")
        self.axes.set_title(f"Frame: {self.app.current_camera.current_frame_idx}")
        frame = self.app.current_camera.get_current_frame()
        if frame is None:
            print(f"Failed to read frame {self.app.current_camera.current_frame_idx}")
            return
        self.img_display.set_data(frame)
        self.draw()
        self.axes.get_figure().canvas.draw()
        return

    def on_click(self, event):
        """
        Handles mouse click events in the viewport, responsible for placing keypoints.
        """
        if event.inaxes != self.axes or self.app.window.toolbar.mode != Mode.PLACE:
            return
        x, y = event.xdata, event.ydata
        self.app.place_keypoint(x, y)

class Canvas(FigureCanvas):
    """
    This class is the main canvas of the application. It contains the viewport and keypoint picker.

    It contains the following attributes:
        - app: A reference to the main application instance.
        - nav_toolbar: The navigation toolbar from matplotlib which is used in the background for the viewport control tools.
        - viewport: The viewport for displaying the image.
        - keypoint_picker: The keypoint picker for placing keypoints.
    """
    def __init__(self, app, parent=None):
        """
        Initializes the canvas.
        """
        self.app = app
        figure, (ax1, ax2) = plt.subplots(1,2)
        super().__init__(figure)
        self.nav_toolbar = NavigationToolbar(self, parent)
        self.nav_toolbar.hide()
        self.viewport = Viewport(self.app, ax1)
        self.keypoint_picker = KeypointPicker(self.app, ax2)

    def reset_view(self):
        """
        Resets the view of the canvas.
        """
        self.nav_toolbar.home()
