"""
This module contains all the functionality related to interfacing with the underlying pose dataset.
"""

import numpy as np # Used to read and write the pose data format
import os # Used to handle file paths

from camera import Camera, Cameras # Used to handle camera metadata

class PoseData:
    """
    This object is an interface to load and store the data associated with a pose dataset.

    It contains the following attributes:
    - name: The name of the dataset.
    - path: The path to the dataset.
    - poses: A dictionary containing the 2D hand poses for each camera.
    - cameras: A Cameras object containing information about the cameras used in the dataset.
    """
    def __init__(self, dataset_name: str):
        """
        Initialize the PoseData object with the data specified in the dataset files located in "output_3d/dataset_name/".

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.name = dataset_name
        self.path = os.path.join("output_3d", dataset_name, 'hand_poses_2d.npz')
        self.load()
        self.cameras = Cameras(dataset_name, list(self.poses.keys()))
        print(f"Loaded camera metadata from the dataset")

    def load(self):
        """
        This method (re)loads pose data from the file with the path specified in self.path.
        """
        if not os.path.exists(self.path):
            return None
        data = np.load(self.path, allow_pickle=True)
        self.poses = data['poses_2d'].item()
        print(f"Loaded poses from {self.path}")

    def save(self):
        """
        This method saves the pose data to the file with the path specified in self.path.
        """
        np.savez(self.path, poses_2d=self.poses)
        print(f"Saved poses to {self.path}")

    def get_pose(self, camera: Camera, hand_idx: int):
        """
        This method allows you to get a pose object for the given camera and hand index.
        The pose object serves as an abstraction over the raw pose data,
        providing a convenient interface for accessing and manipulating the pose information.

        Args:
            camera (Camera): The camera object which is used to index into the pose data.
            hand_idx (int): The hand index which is used to index into the pose data.

        Returns:
            Pose: The pose object for the given camera and hand index.
        """
        return Pose(self, camera, hand_idx)

    def flip_hands(self, camera: Camera):
        """
        Flips the data of the two hands.
        """
        for frame in range(0, camera.frame_count):
            tmp = self.poses[camera.name()][frame][0]
            self.poses[camera.name()][frame][0] = self.poses[camera.name()][frame][1]
            self.poses[camera.name()][frame][1] = tmp

class Pose:
    """
    This class is an abstraction over the raw pose data,
    providing a convenient interface for accessing and manipulating the pose information.
    This class does not store any data itself,
    but rather acts as a wrapper around the PoseData object.
    An instance of this class is also not intended to be kept alive for more than one scope.
    Instead one should use the PoseData.get_pose method to obtain a Pose object every time it is needed.

    It contains the following attributes:
    - data: A reference to the PoseData object which actually stores all of the data.
    - camera: A reference to the Camera object which is used to index into the pose data.
    - hand_idx: The hand index which is used to index into the pose data.
    """
    def __init__(self, data: PoseData, camera: Camera, hand_idx: int):
        """
        Initialize a Pose object. Normally this method should not be called directly.
        Instead, use the PoseData.get_pose method to obtain a Pose object.

        Args:
            data (PoseData): The PoseData object which actually stores all of the data.
            camera (Camera): The Camera object which is used to index into the pose data.
            hand_idx (int): The hand index which is used to index into the pose data.
        """
        self.data = data
        self.camera = camera
        self.hand_idx = hand_idx

    def get_positions(self):
        """
        Returns a list of the positions in the current frame for each keypoint.
        """
        return list(self.gen_positions())

    def gen_positions(self):
        """
        Generates the positions in the current frame for each keypoint.
        """
        keypoints_data = self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints
        for i in range(0, 21):
            pos = keypoints_data[0, i]
            yield (pos[0], pos[1])

    def place_keypoint(self, keypoint_idx: int, x: float, y: float):
        """
        Places a keypoint with score 1.0 at the given position in the current frame.

        Args:
            keypoint_idx (int): The index of the keypoint to place.
            x (float): The x-coordinate of the keypoint.
            y (float): The y-coordinate of the keypoint.
        """
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints[0, keypoint_idx] = [x, y]
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 1.0

    def remove_keypoint(self, keypoint_idx: int):
        """
        Removes a keypoint by setting its score to 0.0.

        Args:
            keypoint_idx (int): The index of the keypoint to remove.
        """
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 0.0

    def is_keypoint_drawable(self, keypoint_idx: int):
        """
        Checks if a keypoint is drawable.

        Args:
            keypoint_idx (int): The index of the keypoint to check.

        Returns:
            bool: True if the keypoint is drawable, False otherwise.
        """
        return self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] > 0.3
