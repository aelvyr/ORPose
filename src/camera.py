import os
import cv2

class Camera:
    def __init__(self, cameras, camera_idx: int, current_frame_idx: int = 0):
        self.cameras = cameras
        self.idx = camera_idx
        self.current_frame_idx = current_frame_idx
        self.frame_count = 0
        self.video = None
        self.load()

    def name(self):
        return self.cameras.data[self.idx]

    def video_path(self):
        path = os.path.join(self.cameras.video_path, f"{self.name()}_synced_cut.MP4")
        return path

    def load(self):
        if self.video is not None:
            self.video.release()
        self.video = cv2.VideoCapture(self.video_path())
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def goto_frame(self, frame_idx: int):
        if 0 <= frame_idx < self.frame_count:
            print(f"went to frame {frame_idx}")
            self.current_frame_idx = frame_idx
            return True
        else:
            print(f"failed to goto frame {frame_idx}")
            return False

    def get_current_frame(self):
        if self.video is None:
            return None
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        success, frame = self.video.read()
        if not success:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self):
        if self.video is None:
            return
        self.video.release()



class Cameras:
    def __init__(self, dataset_name: str, cameras):
        self.video_path = os.path.join("inputs", dataset_name)
        self.data = cameras

    def get(self, idx: int, current_frame_idx: int=0):
        return Camera(self, idx, current_frame_idx)
