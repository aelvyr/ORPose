"""
This module contains all the functionality related to interfacing with the underlying pose dataset.
"""

import numpy as np # Used to read and write the pose data format
from pathlib import Path
import cv2 as cv

from camera import Camera, Cameras # Used to handle camera metadata

class FrameData:
    keypoints = {}
    keypoint_scores = {}

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
        output_3d = Path("output_3d")
        legacy = output_3d / dataset_name
        if legacy.exists():
            self.paths = [(0, legacy)]
        else:
            i = 0
            self.paths = []
            while True:
                path = output_3d / f"{dataset_name}_{i}"
                if not path.exists():
                    break
                self.paths += [(i, path)]
                i += 1
        self.load()
        self.cameras = Cameras(dataset_name, self.available_cameras())
        self.persons = len(self.paths)
        print(f"Loaded camera metadata from the dataset")

    def available_cameras(self):
        video_path = Path("inputs") / self.name
        cameras = []
        for camera_file in video_path.iterdir():
            cameras += [camera_file.stem]
        return cameras

    def empty_data(self, idx):
        self.data[idx] = {}
        for camera in self.available_cameras():
            self.data[idx][camera] = []
            video = cv.VideoCapture(Path("inputs") / self.name / f"{camera}.MP4")
            frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            for i in range(0,frames):
                self.data[idx][camera].append([])
                for j in range(2):
                    self.data[idx][camera][i].append(FrameData())
                    for k in range(21):
                        self.data[idx][camera][i][j].keypoints[0, k] = (0,0)
                        self.data[idx][camera][i][j].keypoint_scores[0, k] = 0
            print(f"Initialized empty data for person {idx}")

    def load(self):
        """
        This method (re)loads pose data from the file with the path specified in self.path.
        """
        self.data = []
        for idx, path in self.paths:
            path = path / "hand_poses_2d.npz"
            self.data.append([])
            if not path.exists():
                self.empty_data(idx)
                continue
            self.data[idx] = np.load(path, allow_pickle=True)["poses_2d"].item()
            print(f"Loaded pose data from {path}")

    def save(self):
        """
        This method saves the pose data to the file with the path specified in self.path.
        """
        for idx, path in self.paths:
            np.savez(path, poses_2d=self.data[idx])
            print(f"Saved poses to {path}")

    def get_pose(self, person: int, camera: Camera, hand_idx: int):
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
        return Pose(self.data[person], camera, hand_idx)

    def flip_hands(self, person: int, camera: Camera):
        """
        Flips the data of the two hands.
        """
        for frame in range(0, camera.frame_count):
            tmp = self.data[person][camera.name()][frame][0]
            self.data[person][camera.name()][frame][0] = self.data[person][camera.name()][frame][1]
            self.data[person][camera.name()][frame][1] = tmp

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
    def __init__(self, data, camera: Camera, hand_idx: int):
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
        keypoints_data = self.data[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints
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
        self.data[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints[0, keypoint_idx] = [x, y]
        self.data[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 1.0

    def remove_keypoint(self, keypoint_idx: int):
        """
        Removes a keypoint by setting its score to 0.0.

        Args:
            keypoint_idx (int): The index of the keypoint to remove.
        """
        self.data[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 0.0

    def is_keypoint_drawable(self, keypoint_idx: int):
        """
        Checks if a keypoint is drawable.

        Args:
            keypoint_idx (int): The index of the keypoint to check.

        Returns:
            bool: True if the keypoint is drawable, False otherwise.
        """
        return self.data[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] > 0.3
