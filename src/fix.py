import cv2 as cv
import numpy as np
from pathlib import Path

"""
This script was created to fix a scaling issue in the pose data caused by a bug.
"""

class PoseData:
    def __init__(self, dataset_name):
        """
        Initialize the PoseData object with the data specified in the dataset files located in "output_3d/dataset_name/".

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.name = dataset_name
        output_3d = Path("output_3d")
        legacy = output_3d / self.name
        if legacy.exists():
            self.paths = [(0, legacy)]
        else:
            i = 0
            self.paths = []
            while True:
                path = output_3d / f"{self.name}_{i}"
                if not path.exists():
                    break
                self.paths += [(i, path)]
                i += 1
        self.load()
        self.persons = len(self.paths)
        print("Loaded camera metadata from the dataset")

    def load(self):
        """
        This method (re)loads pose data from the file with the path specified in self.path.
        """
        self.data = []
        for idx, path in self.paths:
            path = path / "hand_poses_2d.npz"
            self.data.append({})
            if not path.exists():
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

dataset_name = "chb8"
person = 1
resolution_of_camera = "ORX_camera"

pose = PoseData(dataset_name)

for i in range(5,12):
    to_fix_camera = f"gopro{i}"


    video_path = Path("inputs")/dataset_name

    to_fix_camera_video = cv.VideoCapture(video_path/f"{to_fix_camera}.mp4")
    resolution_of_camera_video = cv.VideoCapture(video_path/f"{resolution_of_camera}.mp4")

    to_fix_width = to_fix_camera_video.get(cv.CAP_PROP_FRAME_WIDTH)
    to_fix_height = to_fix_camera_video.get(cv.CAP_PROP_FRAME_HEIGHT)
    to_fix_frames = int(to_fix_camera_video.get(cv.CAP_PROP_FRAME_COUNT))
    resolution_of_width = resolution_of_camera_video.get(cv.CAP_PROP_FRAME_WIDTH)
    resolution_of_height = resolution_of_camera_video.get(cv.CAP_PROP_FRAME_HEIGHT)

    for frame in range(to_fix_frames):
        for hand in range(2):
            for keypoint in range(21):
                x, y = pose.data[person][to_fix_camera][frame][hand].keypoints[0,keypoint]
                if x == 0 or y == 0:
                    continue
                print(f"Original coordinates: ({x}, {y})")
                x /= resolution_of_width
                y /= resolution_of_height
                x *= to_fix_width
                y *= to_fix_height
                print(f"Adjusted coordinates: ({x}, {y})")
                pose.data[person][to_fix_camera][frame][hand].keypoints[0,keypoint] = (x, y)

    to_fix_camera_video.release()
    resolution_of_camera_video.release()

pose.save()
