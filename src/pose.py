"""
This module contains all the functionality related to interfacing with the underlying pose dataset.
"""

import numpy as np # Used to read and write the pose data format
import cv2 as cv
from pathlib import Path
from camera import Camera

class HandData:
    """
    This class represents a hand pose in a 2D image.

    It contains the following attributes:
    - keypoints: A 2D array of shape (21, 2) containing the 2D coordinates of the hand keypoints.
    - keypoint_scores: A 1D array of shape (21,) containing the confidence scores of the hand keypoints.
    for legacy reasons (ask valery) there is an array of length 1 wrapping the keypoints and scores.
    """
    def __init__(self):
        self.keypoints = np.zeros((1, 21, 2))
        self.keypoint_scores = np.zeros((1, 21))

class PoseData:
    """
    This object is an interface to load and store the data associated with a pose dataset.

    It contains the following attributes:
    - name: The name of the dataset.
    - paths: The paths to the datasets of each person.
    - poses: A dictionary containing the 2D hand poses for each camera.
    - cameras: A Cameras object containing information about the cameras used in the dataset.
    - persons: the number of persons in the dataset.
    """
    def __init__(self, project):
        """
        Initialize the PoseData object with the data specified in the dataset files located in "output_3d/dataset_name/".

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.name = project.dataset_name
        self.project = project
        output_3d = Path("output_3d")
        legacy = output_3d / project.dataset_name
        if legacy.exists():
            self.paths = [(0, legacy)]
        else:
            i = 0
            self.paths = []
            while True:
                path = output_3d / f"{project.dataset_name}_{i}"
                if not path.exists():
                    break
                self.paths += [(i, path)]
                i += 1
        self.load()
        self.persons = len(self.paths)
        self.verify()
        print("Loaded camera metadata from the dataset")

    def verify(self):
        """
        Verifies the datasets integrity.
        """
        for person in range(self.persons):
            for camera in self.data[person].keys():
                if camera not in self.project.cameras.data:
                    print(f"Person {person} has camera {camera}, but no footage of that camera was found")
                    print(f"This camera will be ignored. If you want to include it, please add it to 'inputs/{self.project.dataset_name}/{camera}.mp4'")
            for camera in self.project.cameras.data:
                if camera not in self.data[person].keys():
                    self.empty_camera(person, camera)
                    print(f"Camera {camera} was added to person {person}")
                video = cv.VideoCapture(Path("inputs")/self.project.dataset_name/f"{camera}.mp4")
                num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                if len(self.data[person][camera]) != num_frames:
                    print(f"Person {person} has {len(self.data[person][camera])} frames for camera {camera}, but {num_frames} were expected")
                for frame in self.data[person][camera]:
                    if len(frame) != 2:
                        raise ValueError(f"Invalid amount of hands for person {person} and camera {camera} at frame {frame}")
                    #for hand in frame:
                        #if len(hand.keypoints[0]) != 21:
                        #    raise ValueError(f"Invalid amount of keypoints for hand {hand} for person {person} and camera {camera} at frame {frame}. Was {len(hand.keypoints[0])} should be 21")
                        #if len(hand.keypoint_scores[0]) != 21:
                        #    raise ValueError(f"Invalid amount of keypoint scores for hand {hand} for person {person} and camera {camera} at frame {frame}. Was {len(hand.keypoint_scores[0])} should be 21")
            print(f"Verified data for person {person}")

    def empty_person(self, person):
        """
        Initializes the data for a person who is missing data.

        Args:
            person (int): The index of the person.
        """
        self.data[person] = {}
        for camera in self.project.cameras:
            self.empty_camera(person, camera.name())
        print(f"Initialized empty data for person {person}")

    def empty_camera(self, person, camera):
        """
        Initializes the data for a camera for a person which is missing data.

        Args:
            person (int): The index of the person.
            camera (str): The name of the camera.
        """
        self.data[person][camera] = []
        video = cv.VideoCapture(Path("inputs") / self.name / f"{camera}.mp4")
        frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        for i in range(0,frames):
            self.data[person][camera].append([])
            for j in range(2):
                self.data[person][camera][i].append(HandData())
                for k in range(21):
                    self.data[person][camera][i][j].keypoints[0, k] = (0,0)
                    self.data[person][camera][i][j].keypoint_scores[0, k] = 0

    def load(self):
        """
        This method (re)loads pose data from the file with the path specified in self.path.
        """
        self.data = []
        for idx, path in self.paths:
            path = path / "hand_poses_2d.npz"
            self.data.append({})
            if not path.exists():
                self.empty_person(idx)
                continue
            self.data[idx] = np.load(path, allow_pickle=True)["poses_2d"].item()
            print(f"Loaded pose data from {path}")

    def save(self):
        """
        This method saves the pose data to the file with the path specified in self.path.
        """
        for idx, path in self.paths:
            np.savez(path / "hand_poses_2d.npz", poses_2d=self.data[idx])
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
        return Pose(self, person, camera, hand_idx)

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
    def __init__(self, data, person, camera: Camera, hand_idx: int):
        """
        Initialize a Pose object. Normally this method should not be called directly.
        Instead, use the PoseData.get_pose method to obtain a Pose object.

        Args:
            data (PoseData): The PoseData object which actually stores all of the data.
            camera (Camera): The Camera object which is used to index into the pose data.
            hand_idx (int): The hand index which is used to index into the pose data.
        """
        self.data = data
        self.person = person
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
        keypoints_data = self.data.data[self.person][self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints
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
        self.data.data[self.person][self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints[0, keypoint_idx] = [x, y]
        self.data.data[self.person][self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 1.0

    def remove_keypoint(self, keypoint_idx: int):
        """
        Removes a keypoint by setting its score to 0.0.

        Args:
            keypoint_idx (int): The index of the keypoint to remove.
        """
        self.data.data[self.person][self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 0.0

    def is_keypoint_drawable(self, keypoint_idx: int):
        """
        Checks if a keypoint is drawable.

        Args:
            keypoint_idx (int): The index of the keypoint to check.

        Returns:
            bool: True if the keypoint is drawable, False otherwise.
        """
        return self.data.data[self.person][self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] > 0.3
