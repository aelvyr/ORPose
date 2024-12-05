import numpy as np
import os

from helpers.json_handling import read_keypoints_mmpose
from helpers.camera import *
from helpers.definitions import *
from helpers.plotting import *
from helpers.pose_construction import *

folder_path = "pose_outputs"  # Folder containing JSON files
num_cameras = len(cam_names)  # Adjust to the number of cameras in your setup
num_frames = 750
output_folder = "output_3d"
render = True

# Example usage
if __name__ == '__main__':

    # Read keypoints from JSON files
    poses_2d_body, poses_2d_hands = read_keypoints_mmpose(folder_path, cam_names, num_frames, suffix='_selected_hands_2')
    

    # Mask keypoints based on confidence and visibility
    poses_2d_body_masked = mask_keypoints(poses_2d_body, confidence_threshold=0.3, min_cameras=2, num_keypoints=26)
    poses_2d_hands_masked = mask_keypoints(poses_2d_hands, confidence_threshold=0.3, min_cameras=2, num_keypoints=42)
    
    # with open('poses_2d_hands.json', 'w+') as f:
    #     json.dump(poses_2d_hands, f)
        
    # with open('poses_2d_hands_masked.json', 'w+') as f:
    #     json.dump(poses_2d_hands_masked, f)

    camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices = load_camera_params('intrinsics/', 'extrinsics/cameras_poses_proposed.json')

    # Reconstruct the 3D pose
    poses_3d_body, poses_3d_hands = construct_fast_3d(poses_2d_body_masked, poses_2d_hands_masked, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices)
    
    print("Reconstructed 3D poses for the body:", poses_3d_body)
    print("Reconstructed 3D poses for the hands:", poses_3d_hands)

    os.makedirs(output_folder, exist_ok=True)
    with open('extrinsics/origin_camera_poses.json', 'r') as f:
        extr = json.load(f)
    rot = np.array(extr["gopro1"]["rot_mat"])
    t = np.array(extr["gopro1"]["t"]).flatten()

    poses_3d_world_body = np.zeros_like(poses_3d_body)
    poses_3d_world_hands = np.zeros_like(poses_3d_hands)
    for i in range(len(poses_3d_world_body)):
        for j in range(len(poses_3d_body[i])):
            poses_3d_world_body[i, j, :] = t + rot @ poses_3d_body[i, j, :]
        for j in range(len(poses_3d_hands[i])):
            poses_3d_world_hands[i, j, :] = t + rot @ poses_3d_hands[i, j, :]
        
    np.savez(os.path.join(output_folder, 'poses_init_hands.npz'), poses_2d=poses_2d_body, poses_3d_body=poses_3d_world_body, poses_3d_hands=poses_3d_world_hands)

    if render:
        
        pose2video(poses_3d_world_body, 'vis_dir/', 'plt_ani', poses_3d_hands=poses_3d_world_hands, xlim=(1, -1), ylim=(1, -1), zlim=(0, 2.5))
