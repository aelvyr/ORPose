import os
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np

class Camera:
    """
    Wrapper around either a video (mp4) OR a sequence of images for a camera.
    """

    def __init__(self, cameras: "Cameras", camera_idx: int, current_frame_idx: int = 0):
        self.cameras = cameras
        self.idx = camera_idx
        self.current_frame_idx = current_frame_idx

        # Common fields
        self.frame_count = 0

        # Video-specific
        self.video: Optional[cv2.VideoCapture] = None

        # Foto-specific
        self.image_files: List[Path] = []

        self.load()

    def name(self) -> str:
        """Return the camera name."""
        return self.cameras.data[self.idx]

    # ---------- paths ----------

    def video_path(self) -> str:
        """Return the path to the .mp4 file for this camera (video mode)."""
        return os.path.join(self.cameras.video_path, f"{self.name()}.mp4")

    # ---------- loading ----------

    def load(self):
        """Load video (video mode) or list images (foto mode)."""
        # Clean up any previous state
        if self.video is not None:
            self.video.release()
            self.video = None
        self.image_files = []
        self.frame_count = 0

        if self.cameras.foto_mode:
            # FOTO MODE: get the pre-built list of image files for this camera
            cam_name = self.name()
            self.image_files = list(self.cameras.foto_index.get(cam_name, []))
            self.frame_count = len(self.image_files)
        else:
            # VIDEO MODE: open the mp4 file
            vp = self.video_path()
            self.video = cv2.VideoCapture(vp)
            if not self.video.isOpened():
                # Fail gracefully: zero frames
                self.frame_count = 0
            else:
                self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Clamp current frame index into valid range
        self.current_frame_idx = max(0, min(self.current_frame_idx, max(0, self.frame_count - 1)))

    # ---------- navigation ----------

    def goto_frame(self, frame_idx: int) -> bool:
        """Set the current frame index, if in range."""
        if 0 <= frame_idx < self.frame_count:
            self.current_frame_idx = frame_idx
            return True
        return False

    # ---------- retrieval ----------

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Return the current frame as an RGB image (H, W, 3) or None on failure.
        """
        if self.frame_count == 0:
            return None

        if self.cameras.foto_mode:
            # Read from image sequence
            img_path = str(self.image_files[self.current_frame_idx])
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Read from video
            if self.video is None:
                return None
            # Seek to frame and read
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            success, frame_bgr = self.video.read()
            if not success or frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def __del__(self):
        if self.video is not None:
            self.video.release()


class Cameras:
    """
    Wrapper around cameras in the dataset.

    In video mode:
      - data: list of camera names (each corresponds to <name>.mp4 in video_path)

    In foto mode:
      - data: list of camera names (derived from prefixes)
      - foto_index: dict[camera_name] -> list[Path to image files] (sorted)
    """

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}

    def __init__(
        self,
        dataset_name: str,
        cameras: List[str],
        *,
        foto_mode: bool = False,
        foto_index: Optional[Dict[str, List[Path]]] = None
    ):
        self.video_path = os.path.join("inputs", dataset_name)
        self.data = cameras
        self.foto_mode = foto_mode
        self.foto_index: Dict[str, List[Path]] = foto_index or {}

    def get(self, idx: int, current_frame_idx: int = 0) -> Camera:
        return Camera(self, idx, current_frame_idx)
