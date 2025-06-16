from pose import PoseData
from ui import Window
from PyQt5.QtWidgets import QApplication
import sys

class App(QApplication):
    def __init__(self, dataset_name):
        super().__init__(sys.argv)
        self.dataset = PoseData(dataset_name)
        self.current_camera = self.dataset.cameras.get(0)
        self.current_hand = 0
        self.current_keypoint = 0
        self.keypoint_advance = 0
        self.frame_step = 30
        self.keypoints_hidden = False
        self.window = Window(self)

    def run(self):
        return self.exec_()

    def change_camera(self, index):
        self.current_camera = self.dataset.cameras.get(index)
        self.window.canvas.viewport.render_current_frame()

    def change_hand(self, index):
        self.current_hand = index
        self.window.canvas.viewport.render_current_frame()

    def save(self):
        self.dataset.save()

    def goto_last_frame(self):
        self.current_camera.goto_frame(self.current_camera.frame_count - 1 - (self.current_camera.frame_count - 1) % self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def next_frame(self):
        self.current_camera.goto_frame(self.current_camera.current_frame_idx + self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def prev_frame(self):
        self.current_camera.goto_frame(self.current_camera.current_frame_idx - self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def goto_first_frame(self):
        self.current_camera.goto_frame(0)
        self.window.canvas.viewport.render_current_frame()

    def home_view(self):
        self.window.canvas.nav_toolbar.home()

    def set_keypoint_advance(self, idx):
        self.keypoint_advance = idx - 1

    def next_keypoint(self):
        self.set_current_keypoint(self.current_keypoint + 1)

    def prev_keypoint(self):
        self.set_current_keypoint(self.current_keypoint - 1)

    def set_current_keypoint(self, idx):
        if 0 <= idx <= 20:
            self.current_keypoint = idx
            self.window.canvas.keypoint_picker.draw()
            self.window.canvas.viewport.draw()

    def place_keypoint(self, x, y):
        self.dataset.get_pose(self.current_camera, self.current_hand).place_keypoint(self.current_keypoint, x, y)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw_keypoints()

    def delete_keypoint(self):
        self.dataset.get_pose(self.current_camera, self.current_hand).remove_keypoint(self.current_keypoint)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw_keypoints()

    def toggle_keypoint_visibility(self):
        self.keypoints_hidden = not self.keypoints_hidden
        self.window.toolbar.visibility_button.setIcon(self.window.toolbar.visibility_icons[self.keypoints_hidden])
        self.window.canvas.viewport.draw_keypoints()

    def flip_hand_side(self):
        for frame in range(0, self.current_camera.frame_count):
            tmp = self.dataset.poses[self.current_camera.name()][frame][0]
            self.dataset.poses[self.current_camera.name()][frame][0] = self.dataset.poses[self.current_camera.name()][frame][1]
            self.dataset.poses[self.current_camera.name()][frame][1] = tmp
        self.window.canvas.viewport.draw_keypoints()

def main():
    dataset_name = "cha1"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    return App(dataset_name).run()

main()
