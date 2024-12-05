import cv2
import numpy as np

from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pycalib.calib import triangulate

from helpers.definitions import *
from helpers.camera import *

import numpy as np
import cv2

import numpy as np
import cv2

def create_3d_hand_poses_multi_frame(
    hand_poses_2d_per_camera_per_frame,
    body_keypoints_3d_per_frame,
    projection_matrices,
    distortion_coefficients,
    left_wrist_index,
    right_wrist_index,
    wrist_hand_index,
    num_keypoints_hand
):
    """
    Create 3D hand poses from 2D detections across multiple cameras and frames.

    Parameters:
        hand_poses_2d_per_camera_per_frame (list of list of list of np.ndarray):
            A list where each element corresponds to a frame, and contains lists of 2D hand keypoints for each camera.
        body_keypoints_3d_per_frame (list of np.ndarray): List of interpolated 3D body keypoints for each frame.
        projection_matrices (list of np.ndarray): Projection matrices for each camera.
        distortion_coefficients (list of np.ndarray): Distortion coefficients for each camera.
        left_wrist_index (int): Index of the left wrist in the body keypoints.
        right_wrist_index (int): Index of the right wrist in the body keypoints.
        wrist_hand_index (int): Index of the wrist in the hand keypoints.
        num_keypoints_hand (int): The number of keypoints in the hand.

    Returns:
        list of np.ndarray: A list of 42x3 arrays for each frame, where the first 21 rows are the left-hand keypoints,
                            and the next 21 rows are the right-hand keypoints.
    """
    def is_left_hand(hand_wrist_2d, left_wrist_2d, right_wrist_2d):
        left_dist = np.linalg.norm(hand_wrist_2d - left_wrist_2d)
        right_dist = np.linalg.norm(hand_wrist_2d - right_wrist_2d) 
        return left_dist < right_dist
    
    
    def triangulate_hands(hands_2d, wrist_3d):
        if not hands_2d:
            return np.full((21,3), np.nan)  # Default to zero if no detections

        points_3d = np.full((21,3), np.nan)
        
        for i in range(num_keypoints_hand):  # Assume 21 keypoints per hand
            valid_points_2d = []
            valid_proj_matrices = []

            for cam_idx, hand in hands_2d:
                if hand[i] is not None:
                    valid_points_2d.append(hand[i])
                    valid_proj_matrices.append(projection_matrices[cam_idx])

            if len(valid_points_2d) >= 2:  # Need at least two views to triangulate

                A = []
                for j, p2d in enumerate(valid_points_2d):
                    x, y = p2d
                    P = valid_proj_matrices[j]
                    A.append(x * P[2, :] - P[0, :])
                    A.append(y * P[2, :] - P[1, :])
                A = np.array(A)

                _, _, V = np.linalg.svd(A)
                point_3d = V[-1]
                points_3d[i] = (point_3d[:3] / point_3d[3])
                
        points_3d[wrist_hand_index] = wrist_3d

        return np.array(points_3d)

    all_frames_hand_poses_3d = []

    # Process each frame
    for frame_idx, (hand_poses_2d_per_camera, body_keypoints_3d) in enumerate(zip(hand_poses_2d_per_camera_per_frame, body_keypoints_3d_per_frame)):
        left_hand_2d = []
        right_hand_2d = []

        left_wrist_3d = body_keypoints_3d[left_wrist_index]
        right_wrist_3d = body_keypoints_3d[right_wrist_index]

        # Project wrist keypoints to 2D for comparison
        left_wrist_2d_per_camera = []
        right_wrist_2d_per_camera = []
        for cam_idx, proj_matrix in enumerate(projection_matrices):
            left_wrist_2d, _ = cv2.projectPoints(
                left_wrist_3d,
                np.zeros(3),
                np.zeros(3),
                proj_matrix[:3, :3],
                distortion_coefficients[cam_idx]
            )
            right_wrist_2d, _ = cv2.projectPoints(
                right_wrist_3d,
                np.zeros(3),
                np.zeros(3),
                proj_matrix[:3, :3],
                distortion_coefficients[cam_idx]
            )
            left_wrist_2d_per_camera.append(left_wrist_2d.reshape(-1, 2))
            right_wrist_2d_per_camera.append(right_wrist_2d.reshape(-1, 2))

        # Classify hands as left or right
        for cam_idx, hands in enumerate(hand_poses_2d_per_camera):
            for idx_hand in range(0, len(hands), num_keypoints_hand):
                if hands[wrist_hand_index+idx_hand] is not None:
                    if is_left_hand(np.array(hands[wrist_hand_index+idx_hand]).reshape(-1, 2), left_wrist_2d_per_camera[cam_idx], right_wrist_2d_per_camera[cam_idx]):
                        left_hand_2d.append((cam_idx, hands[idx_hand:idx_hand+num_keypoints_hand]))
                    else:
                        right_hand_2d.append((cam_idx, hands[idx_hand:idx_hand+num_keypoints_hand]))

        # Triangulate 3D points for left and right hands
        left_hand_3d = triangulate_hands(left_hand_2d, left_wrist_3d)
        right_hand_3d = triangulate_hands(right_hand_2d, right_wrist_3d)

        # Combine results for the frame
        frame_hand_poses_3d = np.vstack([left_hand_3d, right_hand_3d])
        all_frames_hand_poses_3d.append(frame_hand_poses_3d)

    return np.array(all_frames_hand_poses_3d)



def interpolate_keypoints(keypoints):
    """
    Interpolates None values in the 3D keypoints array.

    Args:
        keypoints: A numpy array of shape (num_frames, num_keypoints, None) where
                    None indicates missing values.

    Returns:
        A numpy array of shape (num_frames, num_keypoints, 3) with None values
        interpolated.
    """
    num_frames, num_keypoints, _ = keypoints.shape
    interpolated_keypoints = np.zeros((num_frames, num_keypoints, 3))

    for kp_idx in range(num_keypoints):
        # Extract the 3D coordinates for the current keypoint across all frames
        keypoint_data = keypoints[:, kp_idx, :]
        
        # Create a mask for valid (non-None) entries
        mask = ~np.isnan(keypoint_data[:, 0])  # Check if x-coordinate is not NaN
        
        # If there are enough valid points to interpolate
        if np.sum(mask) > 1:
            # Interpolate for each coordinate
            for dim in range(3):
                valid_indices = np.arange(num_frames)[mask]
                valid_values = keypoint_data[mask, dim]

                # Create an interpolation function
                interp_func = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')

                # Apply interpolation to all frames
                interpolated_keypoints[:, kp_idx, dim] = interp_func(np.arange(num_frames))
        else:
            # If no valid points, set to zero or leave as is
            interpolated_keypoints[:, kp_idx, :] = keypoint_data[:, :]

    return interpolated_keypoints



def initialize_pose_with_triangulation(poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs):
    """
    Initialize the 3D pose using triangulation from 2D poses.
    
    Args:
        poses_2d: A list of 2D keypoints for multiple cameras and frames.
                  poses_2d[frame][camera][keypoint] -> (x, y).
        camera_matrices: A list of 3x4 projection matrices (P = K[R|t]) for each camera.
    
    Returns:
        An initial estimate of the 3D keypoints.
    """
    num_frames = len(poses_2d)
    num_keypoints = len(poses_2d[0][0])
    num_cameras = len(camera_matrices)
    
    # Initial 3D pose for each frame
    initial_poses_3d = []
    
    for frame_idx in range(num_frames):
        frame_3d_pose = []
        
        for keypoint_idx in range(num_keypoints):
            valid_2d_points = []
            projection_matrices = []
            cam_idxs = []
            
            # Collect 2D points with confidence >= threshold
            for camera_idx in range(num_cameras):
                if poses_2d[frame_idx][camera_idx][keypoint_idx] is not None:
                    x, y = poses_2d[frame_idx][camera_idx][keypoint_idx]
                    x_undis, y_undis = undistort_point(x, y, camera_intrinsics[camera_idx].reshape(3,3), distortion_coeffs[camera_idx])
                    valid_2d_points.append([x_undis, y_undis])
                    projection_matrices.append(camera_matrices[camera_idx])
                    cam_idxs.append(camera_idx)


            # print(valid_2d_points)
            # Only triangulate if we have at least two views for the keypoint
            if len(valid_2d_points) == 2:
                point_3dh : np.ndarray = cv2.triangulatePoints(np.array(projection_matrices)[0], np.array(projection_matrices)[1], np.array(valid_2d_points)[0], np.array(valid_2d_points)[1]).flatten()
                point_3d = point_3dh[:3] / point_3dh[3]
                # print(point_3dh)
                # print(point_3d)
                # print([project_point(point_3d, camera_intrinsics[idx].reshape(3,3), camera_extrinsics[idx][0], camera_extrinsics[idx][1], distortion_coeffs[idx], cam_name=cam_names[idx]) for idx in cam_idxs])
                frame_3d_pose.append(point_3d)
            elif len(valid_2d_points) >= 2:
                # Triangulate
                point_3d = triangulate(np.array(valid_2d_points), np.array(projection_matrices))[:3]
                # print(point_3d)
                # print([project_point(point_3d, camera_intrinsics[idx].reshape(3,3), camera_extrinsics[idx][0], camera_extrinsics[idx][1], distortion_coeffs[idx], cam_name=cam_names[idx]) for idx in cam_idxs])
                frame_3d_pose.append(point_3d)
            else:
                frame_3d_pose.append([None, None, None])  # Mark this keypoint as not initialized yet
        
        initial_poses_3d.append(frame_3d_pose)
    
    return initial_poses_3d

def mask_keypoints(poses_2d, confidence_threshold=0.5, min_cameras=2, num_keypoints=num_keypoints):
    """
    Masks keypoints based on confidence and camera visibility.
    Keypoints with confidence below the threshold or visible in fewer than min_cameras cameras will be masked.
    
    Args:
        poses_2d: The 2D poses list in format poses_2d[frame][camera][keypoint] -> (x, y, confidence)
        confidence_threshold: The confidence threshold below which keypoints will be masked.
        min_cameras: Minimum number of cameras in which a keypoint should be visible to be valid.
        num_keypoints: Number of keypoints in the 2d pose (default: num_keypoints from helpers.definitions)
    
    Returns:
        Masked 2D poses: List of 2D poses where invalid keypoints are replaced with `None`.
    """
    num_frames = len(poses_2d)

    # Initialize masked poses (same structure as poses_2d, but with None for invalid keypoints)
    masked_poses = [[[None for _ in range(num_keypoints)] for _ in range(num_cameras)] for _ in range(num_frames)]

    # Loop through each frame and each keypoint
    for frame_idx in range(num_frames):
        for keypoint_idx in range(num_keypoints):
            # Count how many cameras have valid keypoints for this frame and keypoint
            visible_in_cameras = 0
            valid_keypoint = [None] * num_cameras
            
            for camera_idx in range(num_cameras):
                try:
                    x, y, confidence = poses_2d[frame_idx][camera_idx][keypoint_idx]
                    
                    # Check if the keypoint has a valid confidence
                    if confidence >= confidence_threshold and x > 0 and y > 0:
                        valid_keypoint[camera_idx] = (x, y)
                        visible_in_cameras += 1
                except:
                    print(f"poses[{frame_idx}][{camera_idx}][{keypoint_idx}] not found")
            
            # Only keep the keypoint if it's visible in at least `min_cameras` cameras
            if visible_in_cameras >= min_cameras:
                for camera_idx in range(num_cameras):
                    if valid_keypoint[camera_idx] is not None:
                        masked_poses[frame_idx][camera_idx][keypoint_idx] = valid_keypoint[camera_idx]
    
    return masked_poses


def compute_reprojection_error(pose_3d, pose_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs):
    """
    Computes reprojection error between the 3D poses and 2D poses for all frames and cameras.
    
    Args:
        pose_3d: Reconstructed 3D pose (num_keypoints x 3).
        pose_2d: 2D pose (num_cameras x num_keypoints -> (x, y)).
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
    
    Returns:
        Total reprojection error.
    """
    
    total_error = 0.0
    num_valid_points = 0
    
    for camera_idx in range(num_cameras):
        # Get the camera parameters
        K = camera_intrinsics[camera_idx].reshape(3,3)
        (rot, t) = camera_extrinsics[camera_idx]
        distortion = distortion_coeffs[camera_idx]

        for keypoint_idx in range(num_keypoints):
            # Skip masked keypoints (i.e., keypoints not seen in enough cameras)
            if pose_2d[camera_idx][keypoint_idx] is None:
                continue
            
            # Project the 3D keypoint to the 2D plane
            point_3d = pose_3d[keypoint_idx]
            x_proj, y_proj = project_point(point_3d, K, rot, t, distortion, cam_name=cam_names[camera_idx])
            
            # Get the actual 2D point from the data
            x_actual, y_actual = pose_2d[camera_idx][keypoint_idx]
            
            # Compute the reprojection error for this keypoint
            # print(camera_idx)
            # print(keypoint_idx)
            # print(x_proj, y_proj)
            # print(x_actual, y_actual)
            error = np.sqrt((x_proj - x_actual)**2 + (y_proj - y_actual)**2)
            total_error += error
            num_valid_points += 1
    
    # Average reprojection error over all valid points
    if num_valid_points > 0:
        return total_error / num_valid_points
    else:
        return 0.0


def reconstruct_3d_pose(poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices, regularization_weight=1.0, reprojection_weight=1.0, return_init=False, only_init=False, split_opt=None, smoothing=False):
    """
    Reconstructs the 3D pose from 2D poses using multiple cameras, including regularization and reprojection error.
    Args:
        poses_2d: The 2D poses for each frame and each camera (masked based on occlusion and confidence).
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
        camera_matrices: A list of 3x4 projection matrices (P = K[R|t]) for each camera.
        regularization_weight: The weight of the regularization term in the optimization.
        reprojection_weight: The weight of the reprojection error in the optimization.
        return_init: Return a tuple with both the result and the initial estimate (default: False)
    Returns:
        Optimized 3D poses for all frames.
    """
    num_frames = len(poses_2d)
    num_cameras = len(poses_2d[0])
    num_keypoints = len(poses_2d[0][0])

    # Initialize the 3D poses by triangulating each keypoint independently
    initial_poses_3d = np.array(initialize_pose_with_triangulation(poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs))
    #print(initial_poses_3d)

    if num_frames > 1:
        initial_poses_3d = np.where(initial_poses_3d == None, np.nan, initial_poses_3d).astype(np.float32)

        # Interpolate keypoints
        interpolated_result = interpolate_keypoints(initial_poses_3d)

    else:
        interpolated_result = np.where(initial_poses_3d == None, 0, initial_poses_3d).astype(np.float32)

    if only_init:
        return interpolated_result
    
    if smoothing:
        smoothed_result = savgol_filter(interpolated_result, 20, 10, axis=0)
        return smoothed_result

    if split_opt is None:
        print(f"init loss: {objective_function(interpolated_result.flatten())}")
        algo = 'CG'
        print(f"Algorithm: {algo}")

        # Define the objective function
        def objective_function(poses_3d_flat):

            poses_3d = poses_3d_flat.reshape(num_frames, num_keypoints, 3)
            loss = 0.0

            # Regularization: fixed length between keypoints for idx 0    
            for pair, fixed_length in FIXED_LENGTHS.items():
                idx1, idx2 = pair
                p1 = poses_3d[0, idx1]
                p2 = poses_3d[0, idx2]
                length = np.linalg.norm(p1 - p2)
                loss += regularization_weight * (length - fixed_length)**2
            
            # Reprojection error for frame 0
            reprojection_error = compute_reprojection_error(poses_3d[0], poses_2d[0], camera_intrinsics, camera_extrinsics, distortion_coeffs)
            loss += reprojection_weight * reprojection_error**2

            for frame_idx in range(1, num_frames):
                # Regularization: fixed length between keypoints
                for pair, fixed_length in FIXED_LENGTHS.items():
                    idx1, idx2 = pair
                    p1 = poses_3d[frame_idx, idx1]
                    p2 = poses_3d[frame_idx, idx2]
                    length = np.linalg.norm(p1 - p2)
                    loss += regularization_weight * (length - fixed_length)**2

                # Regularization term (minimize displacement over time)
                displacement = np.linalg.norm(poses_3d[frame_idx] - poses_3d[frame_idx - 1])
                loss += regularization_weight * displacement**2  # Penalize large movements

                # Reprojection error 
                reprojection_error = compute_reprojection_error(poses_3d[frame_idx], poses_2d[frame_idx], camera_intrinsics, camera_extrinsics, distortion_coeffs)
                loss += reprojection_weight * reprojection_error**2

            return loss

        # Optimize the 3D pose using the objective function
        #result = least_squares(objective_function, interpolated_result, x_scale='jac', method='trf', verbose=2, tr_solver='lsmr', tr_options={'regularize':True}, bounds=(-10,10))
        result = minimize(objective_function, interpolated_result.flatten(), method=algo, jac='2-point', options={'disp': True} )
        print(f"minimized loss: {objective_function(result.x)}")

        if return_init:
            return result.x.reshape(num_frames, num_keypoints, 3), interpolated_result.reshape(num_frames, num_keypoints, 3)
        return result.x.reshape(num_frames, num_keypoints, 3)
    
    else:
        algo = 'L-BFGS-B'
        print(f"Algorithm: {algo}")
        result = np.zeros_like(interpolated_result)
        
        for frame_idx in range(0, num_frames, split_opt):

            # Define the objective function
            def objective_function(poses_3d_flat):

                poses_3d = poses_3d_flat.reshape(split_opt, num_keypoints, 3)
                loss = 0.0

                # Regularization: fixed length between keypoints for idx 0    
                for pair, fixed_length in FIXED_LENGTHS.items():
                    idx1, idx2 = pair
                    p1 = poses_3d[0, idx1]
                    p2 = poses_3d[0, idx2]
                    length = np.linalg.norm(p1 - p2)
                    loss += regularization_weight * (length - fixed_length)**2
                
                # Reprojection error for frame 0
                reprojection_error = compute_reprojection_error(poses_3d[0], poses_2d[frame_idx], camera_intrinsics, camera_extrinsics, distortion_coeffs)
                loss += reprojection_weight * reprojection_error**2

                for fidx in range(1, split_opt):
                    # Regularization: fixed length between keypoints
                    for pair, fixed_length in FIXED_LENGTHS.items():
                        idx1, idx2 = pair
                        p1 = poses_3d[fidx, idx1]
                        p2 = poses_3d[fidx, idx2]
                        length = np.linalg.norm(p1 - p2)
                        loss += regularization_weight * (length - fixed_length)**2

                    # Regularization term (minimize displacement over time)
                    displacement = np.linalg.norm(poses_3d[fidx] - poses_3d[fidx - 1])
                    loss += regularization_weight * displacement**2  # Penalize large movements

                    # Reprojection error 
                    reprojection_error = compute_reprojection_error(poses_3d[fidx], poses_2d[fidx+frame_idx], camera_intrinsics, camera_extrinsics, distortion_coeffs)
                    loss += reprojection_weight * reprojection_error**2

                return loss
            
            interim_result_0 = interpolated_result[frame_idx:frame_idx+split_opt]
            print(f"init loss [{frame_idx}-{frame_idx+split_opt}]: {objective_function(interim_result_0.flatten())}")
            interim_result = minimize(objective_function, interim_result_0.flatten(), method=algo, jac='2-point', options={'disp': True})
            print(f"minimized loss [{frame_idx}-{frame_idx+split_opt}]: {objective_function(interim_result.x)}")
            result[frame_idx:frame_idx+split_opt] = interim_result.x.reshape(split_opt, num_keypoints, 3)

            np.savez('temp_poses_3d.npz', poses_3d=result, idx=frame_idx+split_opt)

        return result


    
    

def validate_cams(poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices):

    # Initialize the 3D poses by triangulating each keypoint independently
    initial_poses_3d = np.array(initialize_pose_with_triangulation(poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs))

    return np.where(initial_poses_3d == None, 0, initial_poses_3d).astype(np.float32).flatten()

def construct_fast_3d(poses_2d_body, poses_2d_hands, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices, wrist_idx_left=9, wrist_idx_right=10, wrist_idx_hands=0):
    """
    Reconstructs the 3D pose from 2D poses using multiple cameras by simple triangulation.
    Args:
        poses_2d_body: The 2D poses for each frame and each camera for the body keypoints (masked based on occlusion and confidence).
        poses_2d_hands: The 2D poses for each frame and each camera for the hand keypoints (masked based on occlusion and confidence).
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
        camera_matrices: A list of 3x4 projection matrices (P = K[R|t]) for each camera.
        wrist_idx_left: Index which corresponds to left wrist in body (default: 9)
        wrist_idx_right: Index which corresponds to right wrist in body (default: 10)
        wrist_idx_hands: Index which corresponds to wrist in hand (default: 0)
    Returns:
        Triangulated 3D poses for all frames.
    """
    num_frames = len(poses_2d_body)
    num_cameras = len(poses_2d_body[0])
    num_keypoints_body = len(poses_2d_body[0][0])
    
    initial_poses_3d_body = np.array(initialize_pose_with_triangulation(poses_2d_body, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs))
    
    # Interpolate keypoints
    initial_poses_3d_body = np.where(initial_poses_3d_body == None, np.nan, initial_poses_3d_body).astype(np.float32)

    # Interpolate keypoints
    interpolated_poses_3d_body = interpolate_keypoints(initial_poses_3d_body)
    
    if poses_2d_hands is None:

        smoothed_result = savgol_filter(interpolated_poses_3d_body, 20, 10, axis=0)
        return smoothed_result

    else:
        
        assert num_frames == len(poses_2d_hands)
        assert num_cameras == len(poses_2d_hands[0])

        num_keypoints_hands = len(poses_2d_hands[0][0])//2
        print(num_keypoints_hands)
        
        initial_poses_3d_hands = create_3d_hand_poses_multi_frame(poses_2d_hands, interpolated_poses_3d_body, camera_matrices, distortion_coeffs, wrist_idx_left, wrist_idx_right, wrist_idx_hands, num_keypoints_hands)
        
        interpolated_poses_3d_hands = interpolate_keypoints(initial_poses_3d_hands)
        smoothed_result_body = savgol_filter(interpolated_poses_3d_body, 20, 10, axis=0)
        
        return smoothed_result_body, interpolated_poses_3d_hands