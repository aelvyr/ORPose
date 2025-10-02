import copy
from pathlib import Path
import numpy as np
import os

script_dir = Path(__file__).resolve().parent      # directory of this .py file
output_dir = (script_dir / "../output_3d/single_frames").resolve()

if not output_dir.exists():
    raise FileNotFoundError(f"Folder not found: {output_dir}")

folders = [p for p in output_dir.iterdir() if p.is_dir()]
print(folders)

base_npz = np.load(os.path.join(script_dir,'../', "hand_poses_2d_base_file.npz"), allow_pickle=True)
base_file_struct = base_npz["poses_2d"].item()  # dict: camera -> list[frames]
base_cams = set(base_file_struct.keys())

for folder in folders:
    dataset_file = os.path.join(folder, "hand_poses_2d.npz")
    data = np.load(dataset_file, allow_pickle=True)
    poses_2d = data['poses_2d'].item()  # dict: camera -> list[frames]
    ds_cams = set(poses_2d.keys())

    # Use only the cameras present in this dataset AND available in base
    cams_to_use = sorted(ds_cams & base_cams)
    if not cams_to_use:
        print(f"[SKIP] {dataset_file}: no overlapping cameras with base.")
        

    # Fresh deep copy of base (so we don't mutate across iterations), then filter to cams_to_use
    poses_2d_new = {cam: copy.deepcopy(base_file_struct[cam]) for cam in cams_to_use}

    # Copy keypoints/scores ONLY for the cameras we keep
    for cam in cams_to_use:
        frames_ds = poses_2d[cam]
        frames_new = poses_2d_new[cam]

        # Align by min length to be safe
        n_frames = min(len(frames_ds), len(frames_new))

        for frame_idx in range(n_frames):
            frame_ds = frames_ds[frame_idx]
            frame_new = frames_new[frame_idx]

            # Case A: dict of {person_idx: hand_obj}
            if isinstance(frame_ds, dict):
                for hand_idx, hand_data in frame_ds.items():
                    if hand_data is None:
                        continue
                    try:
                        frame_new[hand_idx].keypoints = hand_data.keypoints
                        frame_new[hand_idx].keypoint_scores = hand_data.keypoint_scores
                    except AttributeError:
                        # Fallback if some frames were saved inconsistently
                        frame_new[hand_idx].keypoints = hand_data.keypoints
                        frame_new[hand_idx].keypoint_scores = hand_data.keypoint_scores

            # Case B: list/sequence indexed by person
            else:
                # Align by min length
                n_hands = min(len(frame_ds), len(frame_new))
                for hand_idx in range(n_hands):
                    hand_data = frame_ds[hand_idx]
                    if hand_data is None:
                        continue
                    try:
                        frame_new[hand_idx].keypoints = hand_data.keypoints
                        frame_new[hand_idx].keypoint_scores = hand_data.keypoint_scores
                    except AttributeError:
                        frame_new[hand_idx].keypoints = hand_data.keypoints
                        frame_new[hand_idx].keypoint_scores = hand_data.keypoint_scores

            print(f"[FIXED] {Path(dataset_file).parent.name} | {cam} | frame {frame_idx}")

    out_path = Path(dataset_file).with_name("hand_poses_2d_fixed.npz")
    np.savez_compressed(out_path, poses_2d=np.array(poses_2d_new, dtype=object))

