from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from keypoint import Keypoints, Keypoint
from ui.toolbar import Mode

class KeypointPicker(Keypoints):
    def __init__(self, app, axes):
        super().__init__(app, axes, [
            (0,0),
            (-2,1), (-3,3), (-3.5,4), (-4,5),
            (-1,4), (-1.5,5), (-2,6), (-2.5,7),
            (0,4), (0,5.25), (0,6.5), (0,7.75),
            (1,4), (1.25,5), (1.5,6), (1.75,7),
            (2,3.5), (2.5,4.5), (3,5.5), (3.5,6.5)
        ],radius=0.2, pickable=True)
        self.axes.set_xlim(-5,5)
        self.axes.set_ylim(-1,9)
        self.axes.set_aspect('equal')
        self.axes.get_figure().canvas.mpl_connect('pick_event', self.on_pick)
        self.draw()

    def on_pick(self, event):
        if not isinstance(event.artist, Keypoint):
            return
        self.app.set_current_keypoint(event.artist.index)

class Viewport(Keypoints):
    def __init__(self, app, axes):
        super().__init__(app, axes, app.dataset.get_pose(app.current_camera, app.current_hand).get_positions(), render_check=self.render_check)
        self.axes.get_figure().canvas.mpl_connect('button_press_event', self.on_click)
        self.img_display = self.axes.imshow(self.app.current_camera.get_current_frame())
        self.render_current_frame()

    def render_check(self, index):
        return self.app.dataset.get_pose(self.app.current_camera, self.app.current_hand).is_keypoint_drawable(index)

    def draw_keypoints(self):
        self.positions = self.app.dataset.get_pose(self.app.current_camera, self.app.current_hand).get_positions()
        self.draw()

    def render_current_frame(self):
        print(f"rendering frame {self.app.current_camera.current_frame_idx}")
        self.axes.set_title(f"Frame: {self.app.current_camera.current_frame_idx}")
        frame = self.app.current_camera.get_current_frame()
        if frame is None:
            print(f"Failed to read frame {self.app.current_camera.current_frame_idx}")
            return
        self.img_display.set_data(frame)
        self.draw_keypoints()
        self.axes.get_figure().canvas.draw()
        return

    def on_click(self, event):
        if event.inaxes != self.axes or self.app.window.toolbar.mode != Mode.PLACE:
            return
        x, y = event.xdata, event.ydata
        self.app.place_keypoint(x, y)

class Canvas(FigureCanvas):
    def __init__(self, app, parent=None):
        self.app = app
        figure, (ax1, ax2) = plt.subplots(1,2)
        super().__init__(figure)
        self.nav_toolbar = NavigationToolbar(self, parent)
        self.nav_toolbar.hide()
        self.viewport = Viewport(self.app, ax1)
        self.keypoint_picker = KeypointPicker(self.app, ax2)
