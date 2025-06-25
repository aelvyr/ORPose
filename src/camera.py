import os
from typing import List
import cv2

class Camera:
    """
    This class is a wrapper around the video for a camera and its associated data.

    It contains the following attributes:
        - cameras: A reference to the cameras object, containing the path of the video files and the name of the camera.
        - idx: The index of the camera.
        - frame_count: The total number of frames in the video.
        - video: The OpenCV VideoCapture object.
        - current_frame_idx: The index of the current frame.
    """
    def __init__(self, cameras, camera_idx: int, current_frame_idx: int = 0):
        """
        Initialize the Camera object and loads the video.

        Args:
            cameras: A reference to the cameras object.
            camera_idx: The index of the camera.
            current_frame_idx: The index of the current frame.
        """
        self.cameras = cameras
        self.idx = camera_idx
        self.current_frame_idx = current_frame_idx
        self.frame_count = 0
        self.video = None
        self.load()

    def name(self):
        """
        Returns the name of the camera.
        """
        return self.cameras.data[self.idx]

    def video_path(self):
        """
        Returns the path to the video file.
        """
        path = os.path.join(self.cameras.video_path, f"{self.name()}.MP4")
        return path

    def load(self):
        """
        Loads the video.
        """
        if self.video is not None:
            self.video.release()
        self.video = cv2.VideoCapture(self.video_path())
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def goto_frame(self, frame_idx: int):
        """
        Sets the current frame index to the given frame index.
        """
        if 0 <= frame_idx < self.frame_count:
            print(f"went to frame {frame_idx}")
            self.current_frame_idx = frame_idx
            return True
        else:
            print(f"failed to goto frame {frame_idx}")
            return False

    def get_current_frame(self):
        """
        Returns the current frame as an image.
        """
        if self.video is None:
            return None
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        success, frame = self.video.read()
        if not success:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self):
        """
        Releases the video capture object.
        """
        if self.video is None:
            return
        self.video.release()



class Cameras:
    """
    Wrapper around the cameras referenced in the dataset.

    Contains the following attributes:
        - video_path: the path that contains the video files for each camera
        - data: a list of all the cameras in the dataset, this is also a map from the camera index to the camera name
    """

    def __init__(self, dataset_name: str, cameras: List[str]):
        self.video_path = os.path.join("inputs", dataset_name)
        self.data = cameras

    def get(self, idx: int, current_frame_idx: int=0):
        """
        Returns a Camera object for the camera with index idx.

        Args:
            idx: the index of the camera to retrieve
            current_frame_idx: the index of the current frame to set for the camera

        Returns:
            A Camera object for the camera with index idx.
        """
        return Camera(self, idx, current_frame_idx)
