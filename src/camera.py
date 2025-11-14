import os
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np


class Cameras:
    """
    Manager for a set of cameras in a dataset.
    - In video mode: 'data' is the list of camera names (stems)
    - In foto mode:  'data' is the list of camera names (prefixes),
                     and 'foto_index' maps camera_name -> [Path to images] (sorted)
    - media_map: optional mapping {camera_name -> absolute Path to video file}
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        dataset_name: str,
        cameras: List[str],
        *,
        foto_mode: bool = False,
        foto_index: Optional[Dict[str, List[Path]]] = None,
        media_map: Optional[Dict[str, Path]] = None,
    ):
        self.dataset_name = dataset_name
        self.input_dir = Path("inputs") / dataset_name
        self.data: List[str] = list(cameras)
        self.foto_mode: bool = foto_mode
        self.foto_index: Dict[str, List[Path]] = foto_index or {}
        self.media_map: Dict[str, Path] = media_map or {}

    def get(self, idx: int) -> "Camera":
        """Return a Camera instance at index idx (clamped and safe)."""
        n = len(self.data)
        if n == 0:
            raise RuntimeError(
                f"No cameras available for dataset '{self.dataset_name}'. "
                "Check your inputs folder or manifest."
            )
        if idx < 0 or idx >= n:
            idx = 0
        return Camera(self, idx)

    def __len__(self) -> int:
        return len(self.data)


class Camera:
    """
    A single camera view (video or foto sequence).
    """

    def __init__(self, cameras: Cameras, idx: int, current_frame_idx: int = 0):
        self.cameras = cameras
        self.idx = int(idx)
        self.current_frame_idx = max(0, int(current_frame_idx))
        self.video: Optional[cv2.VideoCapture] = None
        self.image_files: List[Path] = []
        self.frame_count: int = 0
        self.load()

    # ---------- identity ----------

    def name(self) -> str:
        return self.cameras.data[self.idx]

    # ---------- paths (video mode) ----------

    def video_path(self) -> str:
        """
        Path to the video file for this camera (video mode).
        Prefer explicit path from media_map; otherwise inputs/<dataset>/<name>.mp4.
        """
        name = self.name()
        if name in self.cameras.media_map:
            return str(self.cameras.media_map[name])
        return str(self.cameras.input_dir / f"{name}.mp4")

    # ---------- loading ----------

    def load(self) -> None:
        """Load video (video mode) or list images (foto mode)."""
        # cleanup any previous handle
        if self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None

        self.image_files = []
        self.frame_count = 0

        if self.cameras.foto_mode:
            cam_name = self.name()
            self.image_files = list(self.cameras.foto_index.get(cam_name, []))
            self.frame_count = len(self.image_files)
        else:
            vp = self.video_path()
            self.video = cv2.VideoCapture(vp)
            if not self.video.isOpened():
                self.frame_count = 0
            else:
                self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.current_frame_idx = (
            max(0, min(self.current_frame_idx, self.frame_count - 1))
            if self.frame_count > 0
            else 0
        )

    # ---------- navigation ----------

    def goto_frame(self, frame_idx: int) -> bool:
        if 0 <= frame_idx < self.frame_count:
            self.current_frame_idx = int(frame_idx)
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
            img_path = str(self.image_files[self.current_frame_idx])
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            if self.video is None:
                return None
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ok, frame_bgr = self.video.read()
            if not ok or frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def __del__(self):
        try:
            if self.video is not None:
                self.video.release()
        except Exception:
            pass
