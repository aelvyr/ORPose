"""
This module contains all the functionality related to interfacing with the underlying pose dataset.
"""

import numpy as np # Used to read and write the pose data format
import cv2 as cv
from pathlib import Path
from camera import Camera

class FrameData:
    """
    This class represents a hand pose in a 2D image.
    DO NOT EDIT THIS CLASS it will break backwards compatibility

    It contains the following attributes:
    - keypoints: A 2D array of shape (21, 2) containing the 2D coordinates of the hand keypoints.
    - keypoint_scores: A 1D array of shape (21,) containing the confidence scores of the hand keypoints.
    for legacy reasons (ask valery) there is an array of length 1 wrapping the keypoints and scores.
    """
    def __init__(self, frame, camera, person):
        self.frame = frame
        self.camera = camera
        self.person = person
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
        self.name = project.dataset_name
        self.project = project
        output_3d = Path("output_3d")

        # --- NEW PERSON DISCOVERY LOGIC ---
        # 1) Prefer numbered person folders: <name>_0, <name>_1, ...
        numbered_paths = []
        i = 0
        while True:
            path = output_3d / f"{project.dataset_name}_{i}"
            if not path.exists():
                break
            numbered_paths.append((i, path))
            i += 1

        if numbered_paths:
            # Multi-person project: use numbered dirs
            self.paths = numbered_paths
        else:
            # Single-person / legacy project: use unsuffixed folder if it exists
            legacy = output_3d / project.dataset_name
            if legacy.exists():
                self.paths = [(0, legacy)]
            else:
                # Nothing found – just start with an empty list and let verify() create if needed
                self.paths = []

        self.load()
        self.persons = len(self.paths)
        self.verify()
        print("Loaded camera metadata from the dataset")

    # ---------- helpers ----------

    def _frame_count_for_camera(self, camera_name: str) -> int:
        """
        Return the expected number of frames for a camera based on project mode.

        - Video mode: prefer the project's media_map (absolute paths), fall back to inputs/<dataset>/<camera>.mp4
        - Foto mode: use the number of images for that camera from project.cameras.foto_index
        """
        if getattr(self.project, "foto_mode", False):
            # foto_index: dict[camera_name] -> list[Path]
            file_list = getattr(self.project.cameras, "foto_index", {}).get(camera_name, [])
            return len(file_list)

        # --- VIDEO MODE ---
        # Prefer manifest-provided absolute path
        src_path = None
        try:
            media_map = getattr(self.project.cameras, "media_map", {}) or {}
            if camera_name in media_map:
                src_path = media_map[camera_name]
            else:
                # legacy fallback
                src_path = Path("inputs") / self.project.dataset_name / f"{camera_name}.mp4"
        except Exception:
            # ultra-defensive: legacy fallback
            src_path = Path("inputs") / self.project.dataset_name / f"{camera_name}.mp4"

        video = cv.VideoCapture(str(src_path))
        if not video.isOpened():
            return 0
        try:
            return int(video.get(cv.CAP_PROP_FRAME_COUNT))
        finally:
            video.release()


    def verify(self):
        """
        Verifies the dataset’s integrity for both video and foto modes.
        """
        for person in range(self.persons):
            # Warn about cameras present in saved data but missing in current project
            for camera in self.data[person].keys():
                if camera not in self.project.cameras.data:
                    print(f"Person {person} has camera {camera}, but no footage/images for that camera were found.")
                    expected_hint = (
                        f"inputs/{self.project.dataset_name}/{camera}.mp4"
                        if not getattr(self.project, 'foto_mode', False)
                        else f"inputs/{self.project.dataset_name}/{camera}_*.jpg (or .png/.jpeg)"
                    )
                    print(f"This camera will be ignored. If you want to include it, please add it to '{expected_hint}'")

            # Ensure every project camera exists in the pose data
            for camera in self.project.cameras.data:
                if camera not in self.data[person].keys():
                    self.empty_camera(person, camera)
                    print(f"Camera {camera} was added to person {person}")

                num_frames = self._frame_count_for_camera(camera)
                has_frames = len(self.data[person][camera])

                if has_frames != num_frames:
                    print(f"Person {person} has {has_frames} frames for camera {camera}, but {num_frames} were expected")

                # Basic per-frame shape sanity (kept from your code)
                for frame in self.data[person][camera]:
                    if len(frame) != 2:
                        raise ValueError(
                            f"Invalid amount of hands for person {person} and camera {camera} at frame {frame}"
                        )
                    # Optionally re-enable deeper checks here
            print(f"Verified data for person {person}")

    # ---------- initialization for missing data ----------

    def empty_person(self, person):
        """
        Initializes the data for a person who is missing data.
        """
        self.data[person] = {}
        for camera in self.project.cameras.data:
            self.empty_camera(person, camera)
        print(f"Initialized empty data for person {person}")

    def empty_camera(self, person, camera):
        """
        Initializes the data for a missing camera for a person.

        Creates an empty two-hand FrameData entry for every expected frame.
        """
        self.data[person][camera] = []
        frames = self._frame_count_for_camera(camera)
        for i in range(frames):
            # two hands per frame
            self.data[person][camera].append([])
            for j in range(2):
                fd = FrameData(i, camera, person)
                # Initialize to zeros (already zeros in constructor, but keep explicit for clarity)
                # keypoints: (1,21,2), keypoint_scores: (1,21)
                self.data[person][camera][i].append(fd)

    # ---------- IO ----------

    def load(self):
        """
        (Re)loads pose data from each person's hand_poses_2d.npz, or initializes empty if missing.
        """
        self.data = []
        for idx, path in self.paths:
            file_path = path / "hand_poses_2d.npz"
            self.data.append({})
            if not file_path.exists():
                self.empty_person(idx)
                continue
            self.data[idx] = np.load(file_path, allow_pickle=True)["poses_2d"].item()
            print(f"Loaded pose data from {file_path}")

    def save(self):
        """
        Saves pose data back to disk for each person.
        """
        for idx, path in self.paths:
            np.savez(path / "hand_poses_2d.npz", poses_2d=self.data[idx])
            print(f"Saved poses to {path}")

    # ---------- accessors ----------

    def get_pose(self, person: int, camera: Camera, hand_idx: int):
        return Pose(self, person, camera, hand_idx)

    def flip_hands(self, person: int, camera: Camera):
        for frame in range(0, camera.frame_count):
            tmp = self.data[person][camera.name()][frame][0]
            self.data[person][camera.name()][frame][0] = self.data[person][camera.name()][frame][1]
            self.data[person][camera.name()][frame][1] = tmp

    def flip_person(self, a, b, camera: Camera):
        camera_name = camera.name()
        for frame in range(camera.frame_count):
            self.data[a][camera_name][frame], self.data[b][camera_name][frame] = \
                self.data[b][camera_name][frame], self.data[a][camera_name][frame]



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
