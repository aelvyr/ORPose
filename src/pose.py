import numpy as np
import os

from camera import Camera, Cameras

class PoseData:
    def __init__(self, dataset_name: str):
        self.path = os.path.join("output_3d", dataset_name, 'hand_poses_2d.npz')
        self._load()
        self.cameras = Cameras(dataset_name, list(self.poses.keys()))

    def _load(self):
        if not os.path.exists(self.path):
            return None
        data = np.load(self.path, allow_pickle=True)
        self.poses = data['poses_2d'].item()
        print(f"Loaded poses from {self.path}")

    def save(self):
        np.savez(self.path, poses_2d=self.poses)

    def get_pose(self, camera: Camera, hand_idx: int):
        return Pose(self, camera, hand_idx)

class Pose:

    def __init__(self, data: PoseData, camera: Camera, hand_idx: int):
        self.data = data
        self.camera = camera
        self.hand_idx = hand_idx

    def get_positions(self):
        return list(self.gen_positions())

    def gen_positions(self):
        keypoints_data = self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints
        for i in range(0, 21):
            pos = keypoints_data[0, i]
            yield (pos[0], pos[1])

    def place_keypoint(self, keypoint_idx: int, x: float, y: float):
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoints[0, keypoint_idx] = [x, y]
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 1.0

    def remove_keypoint(self, keypoint_idx: int):
        self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] = 0.0

    def is_keypoint_drawable(self, keypoint_idx: int):
        return self.data.poses[self.camera.name()][self.camera.current_frame_idx][self.hand_idx].keypoint_scores[0, keypoint_idx] > 0.3
