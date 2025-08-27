import numpy as np
import pose
import os

INPUT_DIR = "inputs"
OUTPUT_DIR = "output_3d"
BASE_DIR = "."

DATASET_FILES = [os.path.join(OUTPUT_DIR, f, "hand_poses_2d.npz") for f in os.listdir(OUTPUT_DIR) if f.startswith("cha")]

data_base_file = np.load(os.path.join(BASE_DIR, "hand_poses_2d_base_file.npz"), allow_pickle=True)
base_file_struct = data_base_file["poses_2d"].item()
poses_2d_new = base_file_struct.copy()

for dataset_file in DATASET_FILES:
    # Load the dataset
    data = np.load(dataset_file, allow_pickle=True)
    poses_2d = data['poses_2d'].item()
    if poses_2d.keys() == base_file_struct.keys():
        for key in poses_2d.keys():
            for frame_idx, frame_data in enumerate(poses_2d[key]):
                try:
                    for hand_idx, hand_data in frame_data.items():
                        print(f"Fixing {dataset_file} at {key}, frame {frame_idx}, hand {hand_idx}")
                        poses_2d_new[key][frame_idx][hand_idx].keypoints = hand_data.keypoints
                        poses_2d_new[key][frame_idx][hand_idx].keypoint_scores = hand_data.keypoint_scores
                except AttributeError as e:
                    for hand_idx, hand_data in enumerate(frame_data):
                        print(f"Fixing {dataset_file} with inconsistent saving at {key}, frame {frame_idx}, hand {hand_idx}")
                        poses_2d_new[key][frame_idx][hand_idx].keypoints = hand_data.keypoints
                        poses_2d_new[key][frame_idx][hand_idx].keypoint_scores = hand_data.keypoint_scores

    else:
        for key in poses_2d.keys():
            for key_match in base_file_struct.keys():
                if key.startswith(key_match):
                    for frame_idx, frame_data in enumerate(poses_2d[key]):
                        try:
                            for hand_idx, hand_data in frame_data.items():
                                print(f"Fixing {dataset_file} at {key_match}, frame {frame_idx}, hand {hand_idx}")
                                poses_2d_new[key_match][frame_idx][hand_idx].keypoints = hand_data.keypoints
                                poses_2d_new[key_match][frame_idx][hand_idx].keypoint_scores = hand_data.keypoint_scores
                        except AttributeError as e:
                            for hand_idx, hand_data in enumerate(frame_data):
                                print(f"Fixing {dataset_file} with inconsistent saving at {key_match}, frame {frame_idx}, hand {hand_idx}")
                                poses_2d_new[key_match][frame_idx][hand_idx].keypoints = hand_data.keypoints
                                poses_2d_new[key_match][frame_idx][hand_idx].keypoint_scores = hand_data.keypoint_scores
    print("Saving fixed file...")
    np.savez(dataset_file.replace("hand_poses_2d.npz", "hand_poses_2d_fixed.npz"), poses_2d=poses_2d_new)