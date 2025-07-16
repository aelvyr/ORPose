from pathlib import Path
from pose import PoseData
from ui import ProjectWindow
from camera import Cameras

class Project:
    """
    This class is responsible for the main application logic, since that is working on a project.
    It provides functions for performing all actions you can perform on the project and makes sure all other components are updated accordingly.
    """
    def __init__(self, app, dataset_name):
        """
        Initialize the application for the given dataset.
        """
        self.app = app
        self.dark_mode = False
        self.dataset_name = dataset_name
        self.cameras = Cameras(dataset_name, self.available_cameras())
        self.dataset = PoseData(self)
        self.current_camera = self.cameras.get(0)
        self.current_hand = 0
        self.current_keypoint = 0
        self.keypoint_advance = 0
        self.current_person = 0
        self.frame_step = 30
        self.keypoints_hidden = False
        self.window = ProjectWindow(self)
        

    def available_cameras(self):
        """
        Return a list of available cameras for the current project.
        """
        video_path = Path("inputs") / self.dataset_name
        cameras = []
        for camera_file in video_path.iterdir():
            cameras += [camera_file.stem]
        return cameras

    def change_camera(self, index):
        """
        Change the current camera to the one at the given index.
        """
        self.current_camera = self.cameras.get(index)
        self.window.canvas.viewport.render_camera_change()

    def change_person(self, index):
        """
        Change the current person to the one at the given index.
        """
        self.current_person = index
        self.window.canvas.viewport.draw()

    def change_hand(self, index):
        """
        Change the current hand to the one at the given index.
        """
        self.current_hand = index
        self.window.canvas.viewport.draw()

    def save(self):
        """
        Save the current dataset to disk.
        """
        self.dataset.save()

    def goto_last_frame(self):
        """
        Goes to the last frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.frame_count - 1 - (self.current_camera.frame_count - 1) % self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def next_frame(self):
        """
        Goes to the next frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.current_frame_idx + self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def prev_frame(self):
        """
        Goes to the previous frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.current_frame_idx - self.frame_step)
        self.window.canvas.viewport.render_current_frame()

    def goto_first_frame(self):
        """
        Goes to the first frame of the current camera.
        """
        self.current_camera.goto_frame(0)
        self.window.canvas.viewport.render_current_frame()

    def next_keypoint(self):
        """
        Advances to the next keypoint.
        """
        self.set_current_keypoint(self.current_keypoint + 1)

    def prev_keypoint(self):
        """
        Goes back to the previous keypoint.
        """
        self.set_current_keypoint(self.current_keypoint - 1)

    def resize_keypoints(self, size: int):
        """
        Resizes the keypoints.
        """
        self.window.canvas.viewport.set_radius(float(size)/10.0)
        self.window.canvas.viewport.draw()

    def set_current_keypoint(self, idx):
        """
        Sets the current keypoint index.
        """
        if 0 <= idx <= 20:
            self.current_keypoint = idx
        elif idx < 0:
            self.current_keypoint = 20
        elif idx > 20:
            self.current_keypoint = 0
        self.window.canvas.keypoint_picker.draw()
        self.window.canvas.viewport.draw()

    def place_keypoint(self, x, y):
        """
        Places a keypoint at the given coordinates.
        """
        self.dataset.get_pose(self.current_person, self.current_camera, self.current_hand).place_keypoint(self.current_keypoint, x, y)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw()

    def delete_keypoint(self):
        """
        Deletes the current keypoint.
        """
        self.dataset.get_pose(self.current_person, self.current_camera, self.current_hand).remove_keypoint(self.current_keypoint)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw()

    def toggle_keypoint_visibility(self):
        """
        Toggles the visibility of the keypoints.
        """
        self.keypoints_hidden = not self.keypoints_hidden
        self.window.toolbar.visibility_button.setIcon(self.window.toolbar.visibility_icons[self.keypoints_hidden])
        self.window.canvas.viewport.draw()

    def flip_hand_side(self):
        """
        Flips the data of the two hands.
        """
        self.dataset.flip_hands(self.current_person, self.current_camera)
        self.window.canvas.viewport.draw()

    def swap_people(self, other):
        self.dataset.flip_person(self.current_person, other, camera=self.current_camera)
        self.window.canvas.viewport.draw()
