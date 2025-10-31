import cv2
import numpy as np
import os
import json
from itertools import combinations
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from pycalib.calib import triangulate

from helpers.definitions import *
from helpers.camera import *

# Import PyTorch for optimization
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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



def initialize_pose_with_triangulation(poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs, exhaustive_search=False):
    """
    Initialize the 3D pose using triangulation from 2D poses.
    
    Args:
        poses_2d: A list of 2D keypoints for multiple cameras and frames.
                  poses_2d[frame][camera][keypoint] -> (x, y) or (x, y, confidence).
        camera_matrices: A list of 3x4 projection matrices (P = K[R|t]) for each camera.
        camera_intrinsics: Camera intrinsic matrices for each camera.
        camera_extrinsics: Camera extrinsic parameters for each camera.
        distortion_coeffs: Camera distortion coefficients for each camera.
        exhaustive_search: If True, performs exhaustive search to find the best triangulation
                          using highest confidence keypoints and lowest reprojection error.
    
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
            confidences = []
            
            # Collect 2D points with their confidence scores
            for camera_idx in range(num_cameras):
                if poses_2d[frame_idx][camera_idx][keypoint_idx] is not None:
                    keypoint_data = poses_2d[frame_idx][camera_idx][keypoint_idx]
                    
                    # Handle both (x, y) and (x, y, confidence) formats
                    if len(keypoint_data) == 2:
                        x, y = keypoint_data
                        confidence = 1.0  # Default confidence if not provided
                    else:
                        x, y, confidence = keypoint_data
                    
                    x_undis, y_undis = undistort_point(x, y, camera_intrinsics[camera_idx].reshape(3,3), distortion_coeffs[camera_idx])
                    valid_2d_points.append([x_undis, y_undis])
                    projection_matrices.append(camera_matrices[camera_idx])
                    cam_idxs.append(camera_idx)
                    confidences.append(confidence)

            # Only triangulate if we have at least two views for the keypoint
            if len(valid_2d_points) >= 2:
                if exhaustive_search and len(valid_2d_points) > 2:
                    # Exhaustive search: try all combinations and pick the best one
                    best_point_3d = None
                    best_error = float('inf')
                    
                    # Sort cameras by confidence (highest first)
                    confidence_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
                    
                    # Try combinations starting with highest confidence cameras
                    from itertools import combinations
                    for combo_size in range(2, min(len(valid_2d_points) + 1, 5)):  # Limit to max 4 cameras for efficiency
                        for combo_indices in combinations(confidence_indices, combo_size):
                            try:
                                # Extract points for this combination
                                combo_2d_points = [valid_2d_points[i] for i in combo_indices]
                                combo_proj_matrices = [projection_matrices[i] for i in combo_indices]
                                combo_cam_idxs = [cam_idxs[i] for i in combo_indices]
                                
                                # Triangulate
                                if len(combo_2d_points) == 2:
                                    point_3dh = cv2.triangulatePoints(
                                        np.array(combo_proj_matrices[0]), 
                                        np.array(combo_proj_matrices[1]), 
                                        np.array(combo_2d_points[0]), 
                                        np.array(combo_2d_points[1])
                                    ).flatten()
                                    point_3d = point_3dh[:3] / point_3dh[3]
                                else:
                                    point_3d = triangulate(np.array(combo_2d_points), np.array(combo_proj_matrices))[:3]
                                
                                # Calculate reprojection error for this combination
                                total_error = 0
                                for i, cam_idx in enumerate(combo_cam_idxs):
                                    projected_2d = project_point(
                                        point_3d,
                                        camera_intrinsics[cam_idx].reshape(3, 3),
                                        camera_extrinsics[cam_idx][0],
                                        camera_extrinsics[cam_idx][1],
                                        distortion_coeffs[cam_idx]
                                    )
                                    original_2d = [valid_2d_points[combo_indices[i]][0], valid_2d_points[combo_indices[i]][1]]
                                    error = np.linalg.norm(np.array(projected_2d) - np.array(original_2d))
                                    total_error += error
                                
                                avg_error = total_error / len(combo_cam_idxs)
                                
                                # Update best if this is better
                                if avg_error < best_error:
                                    best_error = avg_error
                                    best_point_3d = point_3d
                                    
                            except Exception:
                                continue
                    
                    if best_point_3d is not None:
                        frame_3d_pose.append(best_point_3d)
                    else:
                        frame_3d_pose.append([None, None, None])
                        
                else:
                    # Standard triangulation (use all available cameras)
                    if len(valid_2d_points) == 2:
                        point_3dh = cv2.triangulatePoints(
                            np.array(projection_matrices)[0], 
                            np.array(projection_matrices)[1], 
                            np.array(valid_2d_points)[0], 
                            np.array(valid_2d_points)[1]
                        ).flatten()
                        point_3d = point_3dh[:3] / point_3dh[3]
                        frame_3d_pose.append(point_3d)
                    else:
                        # Triangulate using all cameras
                        point_3d = triangulate(np.array(valid_2d_points), np.array(projection_matrices))[:3]
                        frame_3d_pose.append(point_3d)
            else:
                frame_3d_pose.append([None, None, None])  # Mark this keypoint as not initialized yet
        
        initial_poses_3d.append(frame_3d_pose)
    
    return initial_poses_3d

def mask_keypoints(poses_2d, confidence_threshold=0.5, min_cameras=2, num_keypoints=num_keypoints, multi_person=False):
    """
    Masks keypoints based on confidence and camera visibility.
    Keypoints with confidence below the threshold or visible in fewer than min_cameras cameras will be masked.
    
    Args:
        poses_2d: The 2D poses list/dict in format:
                 - Single person: poses_2d[frame][camera][keypoint] -> (x, y, confidence)
                 - Multi-person: poses_2d[frame][camera][keypoint] -> (x, y, confidence) (for single person data)
        confidence_threshold: The minimum confidence threshold below which keypoints will be masked.
        min_cameras: Minimum number of cameras in which a keypoint should be visible to be valid.
        num_keypoints: Number of keypoints in the 2d pose (default: num_keypoints from helpers.definitions)
        multi_person: Whether this is multi-person data (affects return format, not processing)
    
    Returns:
        Masked 2D poses: List/dict of 2D poses where invalid keypoints are replaced with `None`.
                        Format matches input format.
    """
    if multi_person:
        # For multi-person, the input should already be a single person's data
        # but we maintain the same processing logic
        pass
    
    num_frames = len(poses_2d)
    num_cameras = len(poses_2d[0]) if num_frames > 0 else 0

    # Initialize masked poses (same structure as poses_2d, but with None for invalid keypoints)
    masked_poses = [[[None for _ in range(num_keypoints)] for _ in range(num_cameras)] for _ in range(num_frames)]

    # Loop through each frame and each keypoint
    for frame_idx in range(num_frames):
        for keypoint_idx in range(num_keypoints):
            camera_count = 0
            valid_cameras = []
            
            # Count how many cameras see this keypoint with sufficient confidence
            for camera_idx in range(num_cameras):
                try:
                    x, y, confidence = poses_2d[frame_idx][camera_idx][keypoint_idx]
                    
                    # Check if keypoint meets confidence threshold and basic validity
                    if confidence >= confidence_threshold and x > 0 and y > 0:
                        camera_count += 1
                        valid_cameras.append(camera_idx)
                except:
                    continue
            
            # Only keep keypoints that are visible in at least min_cameras
            if camera_count >= min_cameras:
                for camera_idx in valid_cameras:
                    x, y, confidence = poses_2d[frame_idx][camera_idx][keypoint_idx]
                    masked_poses[frame_idx][camera_idx][keypoint_idx] = (x, y)
    
    return masked_poses


def compute_reprojection_error(pose_3d, pose_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, named=False):
    """
    Computes reprojection error between the 3D poses and 2D poses for all frames and cameras.
    
    Args:
        pose_3d: Reconstructed 3D pose (num_keypoints x 3).
        pose_2d: 2D pose (num_cameras x num_keypoints -> (x, y)).
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
        named: If True, use camera names instead of indices (default: False).
    
    Returns:
        Total reprojection error.
    """
    
    total_error = 0.0
    num_valid_points = 0

    if named:
        cam_names = list(pose_2d.keys())

        for cam_name in cam_names:
            camera_idx = cam_names.index(cam_name)
            # Get the camera parameters
            K = camera_intrinsics[cam_name].reshape(3,3)
            (rot, t) = camera_extrinsics[cam_name]
            distortion = distortion_coeffs[cam_name]

            for keypoint_idx in range(len(pose_2d[cam_name])):
                # Skip masked keypoints (i.e., keypoints not seen in enough cameras)
                if pose_2d[cam_name][keypoint_idx] is None:
                    continue
                
                # Project the 3D keypoint to the 2D plane
                point_3d = pose_3d[keypoint_idx]
                x_proj, y_proj = project_point(point_3d, K, rot, t, distortion)
                
                # Get the actual 2D point from the data
                x_actual, y_actual = pose_2d[cam_name][keypoint_idx]
                
                # Compute the reprojection error for this keypoint
                error = np.sqrt((x_proj - x_actual)**2 + (y_proj - y_actual)**2)
                total_error += error
                num_valid_points += 1

    else:

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
                x_proj, y_proj = project_point(point_3d, K, rot, t, distortion)
                
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


def reconstruct_3d_pose(poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices, 
                     regularization_weight=1.0, reprojection_weight=1.0, return_init=False, only_init=False, 
                     split_opt=None, smoothing=False, use_torch=True, multi_person=False):
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
        only_init: Return only the initial estimate without optimization (default: False)
        split_opt: Split the optimization into chunks of this size (default: None, optimize all frames at once)
        smoothing: Apply smoothing to the result (default: False)
        use_torch: Use PyTorch for optimization if available (default: True)
        multi_person: Whether this is multi-person data (affects processing but not core logic)
    Returns:
        Optimized 3D poses for all frames.
    """
    num_frames = len(poses_2d)
    num_cameras = len(poses_2d[0])
    num_keypoints = len(poses_2d[0][0])

    # Initialize the 3D poses by triangulating each keypoint independently
    initial_poses_3d = np.array(initialize_pose_with_triangulation(poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, distortion_coeffs))

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

    # Check if PyTorch is available and requested
    if use_torch and TORCH_AVAILABLE:
        return optimize_pose_torch(interpolated_result, poses_2d, camera_intrinsics, camera_extrinsics, 
                                  distortion_coeffs, regularization_weight, reprojection_weight, 
                                  return_init, split_opt)
    else:
        return optimize_pose_scipy(interpolated_result, poses_2d, camera_intrinsics, camera_extrinsics, 
                                  distortion_coeffs, regularization_weight, reprojection_weight, 
                                  return_init, split_opt)

def reconstruct_3d_pose_weighted(poses_2d_with_confidence, camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices, 
                               confidence_threshold=0.3, min_cameras=2, regularization_weight=1.0, reprojection_weight=1.0, 
                               return_init=False, only_init=False, split_opt=None, smoothing=False, use_torch=True, 
                               exhaustive_search=False):
    """
    Reconstructs the 3D pose from 2D poses with confidence weighting, without explicit masking.
    Uses confidence values for weighted triangulation and confidence-weighted optimization.
    
    Args:
        poses_2d_with_confidence: The 2D poses for each frame and each camera with confidence values.
                                 Format: poses_2d[frame][camera][keypoint] -> (x, y, confidence)
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
        camera_matrices: A list of 3x4 projection matrices (P = K[R|t]) for each camera.
        confidence_threshold: Minimum confidence threshold for considering keypoints (default: 0.3)
        min_cameras: Minimum number of cameras required for triangulation (default: 2)
        regularization_weight: The weight of the regularization term in the optimization.
        reprojection_weight: The weight of the reprojection error in the optimization.
        return_init: Return a tuple with both the result and the initial estimate (default: False)
        only_init: Return only the initial estimate without optimization (default: False)
        split_opt: Split the optimization into chunks of this size (default: None, optimize all frames at once)
        smoothing: Apply smoothing to the result (default: False)
        use_torch: Use PyTorch for optimization if available (default: True)
        exhaustive_search: Enable exhaustive search in triangulation initialization (default: False)
    
    Returns:
        Optimized 3D poses for all frames with confidence weighting.
    """
    num_frames = len(poses_2d_with_confidence)
    num_cameras = len(poses_2d_with_confidence[0])
    num_keypoints = len(poses_2d_with_confidence[0][0])
    
    # Filter poses based on confidence and min_cameras, but keep confidence values
    filtered_poses_2d = []
    confidence_weights = []
    
    for frame_idx in range(num_frames):
        frame_poses = []
        frame_weights = []
        
        for camera_idx in range(num_cameras):
            camera_poses = []
            camera_weights = []
            
            for keypoint_idx in range(num_keypoints):
                try:
                    x, y, confidence = poses_2d_with_confidence[frame_idx][camera_idx][keypoint_idx]
                    
                    # Count how many cameras see this keypoint above threshold
                    camera_count = 0
                    for c_idx in range(num_cameras):
                        try:
                            _, _, c_conf = poses_2d_with_confidence[frame_idx][c_idx][keypoint_idx]
                            if c_conf >= confidence_threshold:
                                camera_count += 1
                        except:
                            continue
                    
                    # Include keypoint if confidence is above threshold and enough cameras see it
                    if confidence >= confidence_threshold and camera_count >= min_cameras:
                        camera_poses.append((x, y, confidence))
                        camera_weights.append(confidence)
                    else:
                        camera_poses.append(None)
                        camera_weights.append(0.0)
                except:
                    camera_poses.append(None)
                    camera_weights.append(0.0)
                    
            frame_poses.append(camera_poses)
            frame_weights.append(camera_weights)
            
        filtered_poses_2d.append(frame_poses)
        confidence_weights.append(frame_weights)
    
    # Initialize the 3D poses using confidence-weighted triangulation
    initial_poses_3d = np.array(initialize_pose_with_triangulation(
        filtered_poses_2d, camera_matrices, camera_intrinsics, camera_extrinsics, 
        distortion_coeffs, exhaustive_search=exhaustive_search
    ))
    
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

    # Check if PyTorch is available and requested
    # Note: Currently falling back to standard optimization - confidence weighting in optimization to be implemented
    if use_torch and TORCH_AVAILABLE:
        return optimize_pose_torch(interpolated_result, filtered_poses_2d, camera_intrinsics, camera_extrinsics, 
                                 distortion_coeffs, regularization_weight, reprojection_weight, 
                                 return_init, split_opt)
    else:
        return optimize_pose_scipy(interpolated_result, filtered_poses_2d, camera_intrinsics, camera_extrinsics, 
                                 distortion_coeffs, regularization_weight, reprojection_weight, 
                                 return_init, split_opt)

def optimize_pose_scipy(interpolated_result, poses_2d, camera_intrinsics, camera_extrinsics, 
                      distortion_coeffs, regularization_weight, reprojection_weight, 
                      return_init, split_opt):
    """
    Optimize 3D pose using SciPy's minimize function.
    """
    num_frames, num_keypoints, _ = interpolated_result.shape
    
    if split_opt is None:
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
            
        print(f"init loss: {objective_function(interpolated_result.flatten())}")
        algo = 'CG'
        print(f"Algorithm: {algo}")
        
        # Optimize the 3D pose using the objective function
        result = minimize(objective_function, interpolated_result.flatten(), method=algo, jac='2-point', options={'disp': True})
        print(f"minimized loss: {objective_function(result.x)}")

        if return_init:
            return result.x.reshape(num_frames, num_keypoints, 3), interpolated_result.reshape(num_frames, num_keypoints, 3)
        return result.x.reshape(num_frames, num_keypoints, 3)
    
    else:
        algo = 'L-BFGS-B'
        print(f"Algorithm: {algo}")
        result = np.zeros_like(interpolated_result)
        
        for frame_idx in range(0, num_frames, split_opt):
            max_idx = min(frame_idx + split_opt, num_frames)
            chunk_size = max_idx - frame_idx
            
            # Define the objective function
            def objective_function(poses_3d_flat):
                poses_3d = poses_3d_flat.reshape(chunk_size, num_keypoints, 3)
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

                for fidx in range(1, chunk_size):
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
            
            interim_result_0 = interpolated_result[frame_idx:max_idx]
            print(f"init loss [{frame_idx}-{max_idx}]: {objective_function(interim_result_0.flatten())}")
            interim_result = minimize(objective_function, interim_result_0.flatten(), method=algo, jac='2-point', options={'disp': True})
            print(f"minimized loss [{frame_idx}-{max_idx}]: {objective_function(interim_result.x)}")
            result[frame_idx:max_idx] = interim_result.x.reshape(chunk_size, num_keypoints, 3)

            np.savez('temp_poses_3d.npz', poses_3d=result, idx=frame_idx+chunk_size)

        return result

def optimize_pose_torch(interpolated_result, poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, 
                       regularization_weight, reprojection_weight, return_init, split_opt):
    """
    Optimize 3D pose using PyTorch.
    """
    import torch
    from torch import nn
    from torch.optim import Adam

    num_frames, num_keypoints, _ = interpolated_result.shape

    # Convert data to PyTorch tensors
    camera_intrinsics_torch = [torch.tensor(K, dtype=torch.float32).reshape(3, 3) for K in camera_intrinsics]
    camera_extrinsics_torch = [(torch.tensor(rot, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)) for rot, t in camera_extrinsics]
    distortion_coeffs_torch = [torch.tensor(distortion, dtype=torch.float32) for distortion in distortion_coeffs]
    interpolated_result_torch = torch.tensor(interpolated_result, dtype=torch.float32, requires_grad=True)

    

    # Define the loss function
    def compute_loss(poses_3d):
        loss = 0.0

        # Regularization: fixed length between keypoints for idx 0    
        for pair, fixed_length in FIXED_LENGTHS.items():
            idx1, idx2 = pair
            p1 = poses_3d[0, idx1]
            p2 = poses_3d[0, idx2]
            length = torch.norm(p1 - p2)
            loss += regularization_weight * (length - fixed_length)**2
        
        # Reprojection error for frame 0
        reprojection_error = compute_reprojection_error_torch(poses_3d[0], poses_2d[0], camera_intrinsics_torch, camera_extrinsics_torch, distortion_coeffs_torch)
        loss += reprojection_weight * reprojection_error**2

        for frame_idx in range(1, num_frames):
            # Regularization: fixed length between keypoints
            for pair, fixed_length in FIXED_LENGTHS.items():
                idx1, idx2 = pair
                p1 = poses_3d[frame_idx, idx1]
                p2 = poses_3d[frame_idx, idx2]
                length = torch.norm(p1 - p2)
                loss += regularization_weight * (length - fixed_length)**2

            # Regularization term (minimize displacement over time)
            displacement = torch.norm(poses_3d[frame_idx] - poses_3d[frame_idx - 1])
            loss += regularization_weight * displacement**2  # Penalize large movements

            # Reprojection error 
            reprojection_error = compute_reprojection_error_torch(poses_3d[frame_idx], poses_2d[frame_idx], camera_intrinsics_torch, camera_extrinsics_torch, distortion_coeffs_torch)
            loss += reprojection_weight * reprojection_error**2

        return loss
    
    # Define the optimizer
    optimizer = torch.optim.LBFGS(
        [interpolated_result_torch], 
        lr=1,
        max_iter=10,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )

    # Define closure function for optimizer
    def closure():
        loss = compute_loss(interpolated_result_torch)
        loss.backward()
        return loss

    # Optimization loop

    # Run optimization
    # Callback to print loss during optimization
    def print_loss(epoch, loss):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Run optimization
    for i in range(100): # LBFGS is called once and runs its own loop
        loss = optimizer.step(closure)
        print_loss(i * optimizer.state_dict()['state'][0]['n_iter'], loss)

    
    optimized_poses_3d = interpolated_result_torch.detach().numpy().reshape(num_frames, num_keypoints, 3)

    if return_init:
        return optimized_poses_3d, interpolated_result.reshape(num_frames, num_keypoints, 3)
    return optimized_poses_3d

def compute_reprojection_error_torch(pose_3d, pose_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs):
    """
    Computes reprojection error between the 3D poses and 2D poses for all frames and cameras (PyTorch version).
    
    Args:
        pose_3d: Reconstructed 3D pose (num_keypoints x 3).
        pose_2d: 2D pose (num_cameras x num_keypoints -> (x, y, conf)).
        camera_intrinsics: Intrinsic matrices for each camera.
        camera_extrinsics: Extrinsic parameters Rotation matrix, translation vector for each camera
        distortion_coeffs: Distortion coefficients for each camera (k1, k2, p1, p2, k3).
    
    Returns:
        Total reprojection error.
    """
    import torch

    total_error = 0.0
    num_valid_points = 0
    
    for camera_idx in range(len(camera_intrinsics)):
        valid_points_3d = []
        valid_points_2d = []
        valid_indices = []
        
        # Get the camera parameters
        K = camera_intrinsics[camera_idx]
        (rot, t) = camera_extrinsics[camera_idx]
        distortion = distortion_coeffs[camera_idx]

        for keypoint_idx in range(len(pose_3d)):
            # Skip masked keypoints (i.e., keypoints not seen in enough cameras)
            if pose_2d[camera_idx][keypoint_idx] is None:
                continue
            
            # Collect the 3D keypoint and corresponding 2D point
            valid_points_3d.append(pose_3d[keypoint_idx])
            valid_points_2d.append(torch.tensor(pose_2d[camera_idx][keypoint_idx])[:2])
            valid_indices.append(keypoint_idx)
        
        if valid_points_3d:
            # Project all valid 3D points at once
            points_3d_tensor = torch.stack(valid_points_3d)
            points_2d_tensor = torch.stack(valid_points_2d)
            
            # Project the 3D keypoints to the 2D plane
            projected_points = project_points_torch(points_3d_tensor, K, rot, t, distortion)
            
            # Compute the reprojection error for these keypoints
            errors = torch.sqrt(torch.sum((projected_points - points_2d_tensor)**2, dim=1))
            total_error += torch.sum(errors)
            num_valid_points += len(valid_points_3d)
    
    # Average reprojection error over all valid points
    if num_valid_points > 0:
        return total_error / num_valid_points
    else:
        return torch.tensor(0.0)

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

def project_points_torch(points_3d, intrinsics, rotation, translation, distortion):
    """Differentiable projection of 3D points to 2D using PyTorch.
    
    Args:
        points_3d: 3D points tensor of shape (num_points, 3)
        intrinsics: Camera intrinsic matrix (3x3)
        rotation: Camera rotation matrix (3x3)
        translation: Camera translation vector (3,)
        distortion: Distortion coefficients
        
    Returns:
        Projected 2D points tensor of shape (num_points, 2)
    """
    # Apply rotation and translation
    points_rotated = torch.matmul(points_3d, rotation.transpose(-1, -2)) + translation
    
    # Project to image plane (normalized coordinates)
    x = points_rotated[:, 0] / points_rotated[:, 2]
    y = points_rotated[:, 1] / points_rotated[:, 2]
    
    # Apply radial distortion
    r2 = x*x + y*y
    radial = 1.0 + distortion[0]*r2 + distortion[1]*(r2*r2)
    
    xd = x * radial
    yd = y * radial
    
    # Apply tangential distortion
    if distortion.shape[0] >= 4:
        tangential_x = 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
        tangential_y = distortion[2] * (r2 + 2 * y * y) + 2 * distortion[3] * x * y
        
        xd = xd + tangential_x
        yd = yd + tangential_y
    
    # Apply camera matrix
    u = intrinsics[0, 0] * xd + intrinsics[0, 2]
    v = intrinsics[1, 1] * yd + intrinsics[1, 2]
    
    return torch.stack([u, v], dim=-1)

def match_hand_poses(oriented_hands, pose_2d_dict, camera_intrinsics, rotation, translation, distortion, 
                    threshold=200, just_wrist=True, visualize=False, frame=None, hand_id_mapping=None):
    """
    Match 3D hand pose with corresponding 2D hand detection based on reprojection error.
    If hand_id_mapping is provided, use predetermined mapping instead of matching by reprojection.
    
    Args:
        oriented_hand: numpy array (2*num_keypoints_hands, 3) representing 3D hand poses
        pose_2d_dict: dictionary mapping object IDs to 2D hand poses
        camera_intrinsics: camera intrinsic matrix
        rotation: camera rotation matrix
        translation: camera translation vector
        distortion: distortion coefficients
        threshold: maximum allowed reprojection error for matching
        just_wrist: whether to match only the wrist keypoint or all keypoints
        visualize: whether to visualize the matching results
        frame: original image frame for visualization (required if visualize=True)
        hand_id_mapping: dictionary mapping object IDs to hand side (0=left, 1=right)
    
    Returns:
        best_match: best matching 2D pose or None if no good match found
        min_error: minimum reprojection error for best match
        visualization: visualization frame (only if visualize=True)
    """
    if not pose_2d_dict:
        if visualize:
            return (None, float('inf')), (None, float('inf')), frame
        return (None, float('inf')), (None, float('inf'))

    oriented_hand1 = oriented_hands[:num_keypoints_hands]
    oriented_hand2 = oriented_hands[num_keypoints_hands:]

    # If we already have mappings for object IDs, use them directly
    if hand_id_mapping is not None:
        best_match1 = None
        min_error1 = float('inf')
        best_match2 = None
        min_error2 = float('inf')
        
        for obj_id in pose_2d_dict.keys():
            if obj_id in hand_id_mapping:
                # If this object is mapped to left hand (0)
                if hand_id_mapping[obj_id] == 0:
                    best_match1 = obj_id
                    min_error1 = 0  # We're not calculating error here since mapping is predetermined
                # If this object is mapped to right hand (1)
                elif hand_id_mapping[obj_id] == 1:
                    best_match2 = obj_id
                    min_error2 = 0  # We're not calculating error here since mapping is predetermined
        
        # Create visualization if requested
        if visualize:
            vis_frame = frame.copy() if frame is not None else None
            if vis_frame is not None:
                # Draw projected points for initial hand poses
                if not just_wrist:
                    # Left hand (blue)
                    projected_points1 = project_points(oriented_hand1, camera_intrinsics, 
                                        rotation, translation, distortion)
                    for pt in projected_points1:
                        cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
                    # Right hand (green)
                    projected_points2 = project_points(oriented_hand2, camera_intrinsics, 
                                        rotation, translation, distortion)
                    for pt in projected_points2:
                        cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                
                # Draw matched hand poses
                if best_match1 is not None and best_match1 in pose_2d_dict:
                    keypoints = pose_2d_dict[best_match1].keypoints.squeeze()
                    scores = pose_2d_dict[best_match1].keypoint_scores.squeeze()
                    # Draw left hand keypoints (red)
                    for i, (pt, score) in enumerate(zip(keypoints, scores)):
                        if score > 0.3:  # Only draw high-confidence keypoints
                            cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                            cv2.putText(vis_frame, str(i), (int(pt[0])+5, int(pt[1])+5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if best_match2 is not None and best_match2 in pose_2d_dict:
                    keypoints = pose_2d_dict[best_match2].keypoints.squeeze()
                    scores = pose_2d_dict[best_match2].keypoint_scores.squeeze()
                    # Draw right hand keypoints (magenta)
                    for i, (pt, score) in enumerate(zip(keypoints, scores)):
                        if score > 0.3:  # Only draw high-confidence keypoints
                            cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)
                            cv2.putText(vis_frame, str(i), (int(pt[0])+5, int(pt[1])+5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add legend
                cv2.putText(vis_frame, "Blue: Left hand (projected)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(vis_frame, "Green: Right hand (projected)", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, "Red: Matched left hand", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(vis_frame, "Yellow: Matched right hand", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                return (best_match1, min_error1), (best_match2, min_error2), vis_frame
        
        return (best_match1, min_error1), (best_match2, min_error2)
    
    # Otherwise calculate the best match using reprojection error
    if just_wrist:
        
        # Project 3D points to 2D
        projected_points1 = project_point(oriented_hand1[0], camera_intrinsics, 
                                    rotation, translation, distortion)
        # Project 3D points to 2D
        projected_points2 = project_point(oriented_hand2[0], camera_intrinsics, 
                                    rotation, translation, distortion)

    else:
        # Project 3D points to 2D
        projected_points1 = project_points(oriented_hand1, camera_intrinsics, 
                                    rotation, translation, distortion)
        # Project 3D points to 2D
        projected_points2 = project_points(oriented_hand2, camera_intrinsics, 
                                    rotation, translation, distortion)
        
    best_match1 = None
    min_error1 = float('inf')

    best_match2 = None
    min_error2 = float('inf')
    
    # Create visualization frame if requested
    vis_frame = None
    if visualize:
        if frame is None:
            raise ValueError("Frame must be provided for visualization")
        vis_frame = frame.copy()
        
        # Draw projected points for initial hand poses
        if not just_wrist:
            # Left hand (blue)
            for pt in projected_points1:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
            # Right hand (green)
            for pt in projected_points2:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    
    # Check each detected hand
    for obj_id, pose_2d in pose_2d_dict.items():
        if pose_2d is None or len(pose_2d.keypoints) == 0:
            continue

        keypoint_scores = pose_2d.keypoint_scores.squeeze()
        keypoints = pose_2d.keypoints.squeeze()

        if just_wrist:
            keypoints = keypoints[0]
            keypoint_scores = keypoint_scores[0]

            error1 = np.linalg.norm(
                    projected_points1 - keypoints
                )

            error2 = np.linalg.norm(
                    projected_points2 - keypoints
                )
            
            if error1 < min_error1 and error1 < threshold and error1 <= error2:
                min_error1 = error1
                best_match1 = obj_id

            elif error2 < min_error1 and error2 < threshold and error2 <= error1:
                min_error2 = error2
                best_match2 = obj_id

        else:
            
            # Calculate error for valid keypoints
            valid_mask = keypoint_scores > 0.3
            if np.any(valid_mask) and np.sum(valid_mask) > 5:
                errors1 = np.linalg.norm(
                    projected_points1[valid_mask] - keypoints[valid_mask], 
                    axis=1
                )
                mean_error1 = np.mean(errors1)

                errors2 = np.linalg.norm(
                    projected_points2[valid_mask] - keypoints[valid_mask],
                    axis=1
                )
                mean_error2 = np.mean(errors2)
                
                if mean_error1 < min_error1 and mean_error1 < threshold and mean_error1 <= mean_error2:
                    min_error1 = mean_error1
                    best_match1 = obj_id

                elif mean_error2 < min_error2 and mean_error2 < threshold and mean_error2 <= mean_error1:
                    min_error2 = mean_error2
                    best_match2 = obj_id
                
    # Add visualization of matched hand poses if requested
    if visualize and vis_frame is not None:
        # Draw matched hand poses
        if best_match1 is not None and best_match1 in pose_2d_dict:
            keypoints = pose_2d_dict[best_match1].keypoints.squeeze()
            scores = pose_2d_dict[best_match1].keypoint_scores.squeeze()
            # Draw left hand keypoints (red)
            for i, (pt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:  # Only draw high-confidence keypoints
                    cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                    cv2.putText(vis_frame, str(i), (int(pt[0])+5, int(pt[1])+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if best_match2 is not None and best_match2 in pose_2d_dict:
            keypoints = pose_2d_dict[best_match2].keypoints.squeeze()
            scores = pose_2d_dict[best_match2].keypoint_scores.squeeze()
            # Draw right hand keypoints (magenta)
            for i, (pt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:  # Only draw high-confidence keypoints
                    cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)
                    cv2.putText(vis_frame, str(i), (int(pt[0])+5, int(pt[1])+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add legend
        cv2.putText(vis_frame, "Blue: Left hand (projected)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_frame, "Green: Right hand (projected)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, "Red: Matched left hand", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, "Yellow: Matched right hand", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return (best_match1, min_error1), (best_match2, min_error2), vis_frame
    
    if visualize:
        return (best_match1, min_error1), (best_match2, min_error2), vis_frame
        
    return (best_match1, min_error1), (best_match2, min_error2)

def visualize_matched_persons(video_paths, matched_results_path, output_dir, 
                          start_frame=0, end_frame=None, show_keypoints=True, 
                          show_bbox=True, fps=30, colormap=None, use_png_folders=False):
    """
    Visualize matched persons across multiple cameras.
    
    Args:
        video_paths: Dict mapping camera names to video paths (MP4) or image folders (PNG)
        matched_results_path: Path to the matched results JSON file
        output_dir: Directory to save the visualization videos
        start_frame: First frame to visualize
        end_frame: Last frame to visualize (None for all frames)
        show_keypoints: Whether to show keypoints
        show_bbox: Whether to show bounding boxes
        fps: FPS of the output videos
        colormap: Optional colormap for person IDs
        use_png_folders: If True, treat video_paths as folders containing PNG images.
                        If False, treat video_paths as MP4 video files.
    
    Returns:
        Dict mapping camera names to output video paths
    """
    import os
    import json
    import cv2
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matched results
    with open(matched_results_path, 'r') as f:
        matched_results = json.load(f)
    
    # Get the total number of frames
    num_frames = len(matched_results)
    
    if end_frame is None:
        end_frame = num_frames
    
    # Ensure end_frame is within bounds
    end_frame = min(end_frame, num_frames)
    start_frame = max(0, start_frame)
    
    # Initialize video captures and writers
    captures = {}
    writers = {}
    output_paths = {}
    png_file_lists = {}  # For PNG mode: store sorted file lists
    
    for camera_name, video_path in video_paths.items():
        if use_png_folders:
            # Handle PNG folder mode
            if not os.path.exists(video_path):
                print(f"Warning: PNG folder not found for camera {camera_name}: {video_path}")
                continue
            
            # Get list of PNG files and sort them
            png_files = [f for f in os.listdir(video_path) if f.lower().endswith('.png')]
            if not png_files:
                print(f"Warning: No PNG files found in folder for camera {camera_name}: {video_path}")
                continue
            
            # Sort PNG files numerically (assuming format like frame_0001.png or 0001.png)
            try:
                png_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            except ValueError:
                # Fallback to alphabetical sort if numeric extraction fails
                png_files.sort()
            
            png_file_lists[camera_name] = [os.path.join(video_path, f) for f in png_files]
            
            # Read first image to get dimensions
            first_image = cv2.imread(png_file_lists[camera_name][0])
            if first_image is None:
                print(f"Warning: Could not read first PNG for camera {camera_name}")
                continue
            
            height, width = first_image.shape[:2]
            
        else:
            # Handle MP4 video mode
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found for camera {camera_name}: {video_path}")
                continue
                
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video for camera {camera_name}: {video_path}")
                continue
                
            captures[camera_name] = cap
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video file
        output_path = os.path.join(output_dir, f"{camera_name}_matched_visualization.mp4")
        writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )
        
        writers[camera_name] = writer
        output_paths[camera_name] = output_path
    
    # Set up colormap for person IDs
    if colormap is None:
        # Generate distinct colors for each person ID
        def get_distinct_colors(n):
            import colorsys
            HSV_tuples = [(x*1.0/n, 0.8, 0.9) for x in range(n)]
            return list(map(lambda x: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*x)), HSV_tuples))
        
        # Get all unique person IDs
        person_ids = set()
        for frame in matched_results:
            for person in frame.get('matched_instances', []):
                person_ids.add(person.get('person_id'))
        
        colormap = {pid: color for pid, color in zip(sorted(person_ids), get_distinct_colors(len(person_ids)))}
    
    # Process each frame
    for frame_idx in range(start_frame, end_frame):
        if frame_idx >= len(matched_results):
            break
            
        frame_data = matched_results[frame_idx]
        
        # Process each camera
        if use_png_folders:
            # PNG folder mode
            for camera_name in png_file_lists.keys():
                if camera_name not in writers:
                    continue
                    
                # Check if frame index is within available PNG files
                if frame_idx >= len(png_file_lists[camera_name]):
                    print(f"Warning: Frame {frame_idx} not available for camera {camera_name} (only {len(png_file_lists[camera_name])} PNG files)")
                    continue
                
                # Read the PNG file
                png_path = png_file_lists[camera_name][frame_idx]
                frame = cv2.imread(png_path)
                
                if frame is None:
                    print(f"Warning: Could not read PNG file {png_path} for camera {camera_name}")
                    continue
                
                # Draw matched persons for this camera
                for person in frame_data.get('matched_instances', []):
                    person_id = person.get('person_id')
                    color = colormap.get(person_id, (255, 255, 255))  # Default to white if ID not in colormap
                    
                    # Find the instance for this camera
                    camera_instances = [
                        inst for inst in person.get('instances', []) 
                        if inst.get('camera_name') == camera_name
                    ]
                    
                    for instance in camera_instances:
                        # Draw bounding box if requested
                        if show_bbox and 'bbox' in instance:
                            bbox = instance['bbox']
                            if bbox is not None and len(bbox) == 4:
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                # Add person ID label
                                label = f"ID: {person_id}"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw keypoints if requested
                        if show_keypoints and 'keypoints' in instance and 'keypoint_scores' in instance:
                            keypoints = instance['keypoints']
                            scores = instance['keypoint_scores']
                            
                            # Draw keypoints as circles
                            for kp_idx, (kp, score) in enumerate(zip(keypoints, scores)):
                                if score > 0.5:  # Only draw high-confidence keypoints
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(frame, (x, y), 3, color, -1)
                            
                            # Draw skeleton connections if available
                            try:
                                from helpers.definitions import WHOLEBODY_KEYPOINT_PAIRS
                                
                                for pair in WHOLEBODY_KEYPOINT_PAIRS:
                                    idx1, idx2 = pair
                                    if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                                        scores[idx1] > 0.5 and scores[idx2] > 0.5):
                                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                                        cv2.line(frame, pt1, pt2, color, 1)
                            except ImportError:
                                print("Warning: Could not import WHOLEBODY_KEYPOINT_PAIRS, skipping skeleton connections")
                
                # Write the frame
                writers[camera_name].write(frame)
                
                # Display progress
                if frame_idx % 10 == 0:
                    print(f"Processed frame {frame_idx}/{end_frame-start_frame} for camera {camera_name}")
        
        else:
            # MP4 video mode
            for camera_name, cap in captures.items():
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                
                if not success:
                    print(f"Warning: Could not read frame {frame_idx} from camera {camera_name}")
                    continue
                    
                # Draw matched persons for this camera
                for person in frame_data.get('matched_instances', []):
                    person_id = person.get('person_id')
                    color = colormap.get(person_id, (255, 255, 255))  # Default to white if ID not in colormap
                    
                    # Find the instance for this camera
                    camera_instances = [
                        inst for inst in person.get('instances', []) 
                        if inst.get('camera_name') == camera_name
                    ]
                    
                    for instance in camera_instances:
                        # Draw bounding box if requested
                        if show_bbox and 'bbox' in instance:
                            bbox = instance['bbox']
                            if bbox is not None and len(bbox) == 4:
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                # Add person ID label
                                label = f"ID: {person_id}"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw keypoints if requested
                        if show_keypoints and 'keypoints' in instance and 'keypoint_scores' in instance:
                            keypoints = instance['keypoints']
                            scores = instance['keypoint_scores']
                            
                            # Draw keypoints as circles
                            for kp_idx, (kp, score) in enumerate(zip(keypoints, scores)):
                                if score > 0.5:  # Only draw high-confidence keypoints
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(frame, (x, y), 3, color, -1)
                            
                            # Draw skeleton connections if available
                            try:
                                from helpers.definitions import WHOLEBODY_KEYPOINT_PAIRS
                                
                                for pair in WHOLEBODY_KEYPOINT_PAIRS:
                                    idx1, idx2 = pair
                                    if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                                        scores[idx1] > 0.5 and scores[idx2] > 0.5):
                                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                                        cv2.line(frame, pt1, pt2, color, 1)
                            except ImportError:
                                print("Warning: Could not import WHOLEBODY_KEYPOINT_PAIRS, skipping skeleton connections")
                
                # Write the frame
                writers[camera_name].write(frame)
                
                # Display progress
                if frame_idx % 10 == 0:
                    print(f"Processed frame {frame_idx}/{end_frame-start_frame} for camera {camera_name}")
    
    # Clean up
    if not use_png_folders:
        # Only release video captures if we were using MP4 mode
        for cap in captures.values():
            cap.release()
    
    for writer in writers.values():
        writer.release()
    
    print("Visualization completed!")
    return output_paths

# Add this function at the end of the file

def process_and_match_poses(pose_output_dir, cam_names, cam_matrices, cam_intrinsics, 
                          cam_extrinsics, cam_distortions, matched_output_dir,
                          confidence_threshold=0.3, reprojection_threshold=20.0,
                          temporal_similarity_threshold=200.0, min_matched_cameras=2):
    """
    Match person IDs across cameras for the entire sequence at once.
    Creates consistent cross-camera person ID mappings that remain stable throughout the sequence.
    
    Args:
        pose_output_dir: Directory containing the wholebody JSON files
        cam_names: List of camera names
        cam_matrices: Dict mapping camera names to camera projection matrices
        cam_intrinsics: Dict mapping camera names to camera intrinsic matrices  
        cam_extrinsics: Dict mapping camera names to camera extrinsic parameters
        cam_distortions: Dict mapping camera names to camera distortion coefficients
        matched_output_dir: Output directory for matched results
        confidence_threshold: Minimum confidence for keypoints
        reprojection_threshold: Maximum reprojection error for matching
        temporal_similarity_threshold: Threshold for temporal consistency (unused in this implementation)
        min_matched_cameras: Minimum number of cameras a person must be matched in
        
    Returns:
        Path to the saved matched results JSON file
    """    
    def load_pose_data(pose_output_dir, cam_names):
        """Load pose data from all camera JSON files and organize by person_id."""
        pose_data = {}
        
        for cam_name in cam_names:
            json_file = os.path.join(pose_output_dir, f'{cam_name}_wholebody.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    pose_data[cam_name] = data['body_instances']
            else:
                print(f"Warning: Pose file not found for camera {cam_name}")
                
        return pose_data

    def extract_person_sequences(pose_data, cam_names, confidence_threshold):
        """Extract sequences for each person_id in each camera."""
        person_sequences = {}  # {cam_name: {person_id: [frame_data, ...]}}
        
        for cam_name in cam_names:
            if cam_name not in pose_data:
                continue
                
            person_sequences[cam_name] = {}
            
            for frame_idx, frame_data in enumerate(pose_data[cam_name]):
                if 'instances' not in frame_data:
                    continue
                    
                for instance in frame_data['instances']:
                    person_id = instance['person_id']
                    keypoints = np.array(instance['keypoints'])
                    confidence_scores = np.array(instance['keypoint_scores'])
                    
                    # Filter for upper body keypoints (more stable)
                    upper_body_indices = list(range(11)) + [11, 12]  # nose to wrists + hips
                    upper_kps = []
                    upper_confs = []
                    
                    for idx in upper_body_indices:
                        if idx < len(keypoints):
                            upper_kps.append(keypoints[idx])
                            upper_confs.append(confidence_scores[idx] if idx < len(confidence_scores) else 0.0)
                    
                    upper_kps = np.array(upper_kps)
                    upper_confs = np.array(upper_confs)
                    
                    # Only include if enough confident keypoints
                    valid_kps = upper_confs > confidence_threshold
                    if np.sum(valid_kps) >= 5:  # Need at least 5 confident keypoints
                        if person_id not in person_sequences[cam_name]:
                            person_sequences[cam_name][person_id] = []
                        
                        person_sequences[cam_name][person_id].append({
                            'frame_idx': frame_idx,
                            'keypoints': upper_kps,
                            'confidence_scores': upper_confs,
                            'bbox': instance.get('bbox', None),
                            'original_keypoints': keypoints,
                            'original_confidence_scores': confidence_scores
                        })
        
        return person_sequences

    def calculate_sequence_reprojection_error(seq1, seq2, cam1_name, cam2_name, 
                                           cam_matrices, cam_intrinsics, cam_extrinsics, cam_distortions):
        """Calculate average reprojection error between two person sequences."""
        total_error = 0
        valid_frames = 0
        
        # Create frame mapping for both sequences
        seq1_frames = {item['frame_idx']: item for item in seq1}
        seq2_frames = {item['frame_idx']: item for item in seq2}
        
        # Find common frames
        common_frames = set(seq1_frames.keys()) & set(seq2_frames.keys())
        
        if len(common_frames) < 10:  # Need at least 10 common frames for reliable matching
            return float('inf')
        
        for frame_idx in common_frames:
            item1 = seq1_frames[frame_idx]
            item2 = seq2_frames[frame_idx]
            
            kps1 = item1['keypoints']
            kps2 = item2['keypoints']
            conf1 = item1['confidence_scores']
            conf2 = item2['confidence_scores']
            
            frame_error = 0
            valid_kps = 0
            
            # Calculate reprojection error for each keypoint
            for kp_idx in range(len(kps1)):
                if conf1[kp_idx] > 0.3 and conf2[kp_idx] > 0.3:
                    # Triangulate keypoint
                    kp1_2d = kps1[kp_idx]
                    kp2_2d = kps2[kp_idx]
                    
                    # Use DLT triangulation
                    P1 = cam_matrices[cam1_name]
                    P2 = cam_matrices[cam2_name]
                    
                    A = np.array([
                        kp1_2d[0] * P1[2, :] - P1[0, :],
                        kp1_2d[1] * P1[2, :] - P1[1, :],
                        kp2_2d[0] * P2[2, :] - P2[0, :],
                        kp2_2d[1] * P2[2, :] - P2[1, :]
                    ])
                    
                    _, _, V = np.linalg.svd(A)
                    X = V[-1]
                    
                    if X[3] != 0:
                        point_3d = X[:3] / X[3]
                        
                        # Calculate reprojection errors
                        proj1 = project_points(
                            point_3d.reshape(1, 3),
                            cam_intrinsics[cam1_name],
                            cam_extrinsics[cam1_name][0],
                            cam_extrinsics[cam1_name][1],
                            cam_distortions[cam1_name]
                        )
                        
                        proj2 = project_points(
                            point_3d.reshape(1, 3),
                            cam_intrinsics[cam2_name],
                            cam_extrinsics[cam2_name][0],
                            cam_extrinsics[cam2_name][1],
                            cam_distortions[cam2_name]
                        )
                        
                        if proj1.size > 0 and proj2.size > 0:
                            error1 = np.linalg.norm(proj1.squeeze() - kp1_2d)
                            error2 = np.linalg.norm(proj2.squeeze() - kp2_2d)
                            frame_error += (error1 + error2) / 2
                            valid_kps += 1
            
            if valid_kps > 0:
                total_error += frame_error / valid_kps
                valid_frames += 1
        
        return total_error / valid_frames if valid_frames > 0 else float('inf')

    def match_person_sequences_across_cameras(person_sequences, cam_names, cam_matrices, 
                                            cam_intrinsics, cam_extrinsics, cam_distortions, 
                                            reprojection_threshold):
        """Match person IDs across all camera pairs for the entire sequence."""
        print("Matching person sequences across cameras...")
        
        # Create pairwise camera combinations using names directly
        camera_pairs = []
        for i in range(len(cam_names)):
            for j in range(i + 1, len(cam_names)):
                camera_pairs.append((cam_names[i], cam_names[j]))
        
        # For each camera pair, find best person ID matches
        pairwise_matches = {}  # {(cam1, cam2): {person_id1: person_id2}}
        
        for cam1_name, cam2_name in camera_pairs:
            if cam1_name not in person_sequences or cam2_name not in person_sequences:
                continue
                
            print(f"Matching {cam1_name} with {cam2_name}...")
            
            # Get person IDs for both cameras
            cam1_persons = list(person_sequences[cam1_name].keys())
            cam2_persons = list(person_sequences[cam2_name].keys())
            
            if not cam1_persons or not cam2_persons:
                continue
            
            # Calculate reprojection error matrix
            error_matrix = np.full((len(cam1_persons), len(cam2_persons)), float('inf'))
            
            for i, pid1 in enumerate(cam1_persons):
                for j, pid2 in enumerate(cam2_persons):
                    seq1 = person_sequences[cam1_name][pid1]
                    seq2 = person_sequences[cam2_name][pid2]
                    
                    error = calculate_sequence_reprojection_error(
                        seq1, seq2, cam1_name, cam2_name,
                        cam_matrices, cam_intrinsics, cam_extrinsics, cam_distortions
                    )
                    error_matrix[i, j] = error
            
            # Find best matches using Hungarian algorithm (or greedy approach)
            matches = {}
            used_cam2_persons = set()
            
            # Sort by error and assign greedily
            flat_indices = np.argsort(error_matrix.flatten())
            
            for flat_idx in flat_indices:
                i, j = np.unravel_index(flat_idx, error_matrix.shape)
                error = error_matrix[i, j]
                
                if error > reprojection_threshold:
                    break
                    
                pid1 = cam1_persons[i]
                pid2 = cam2_persons[j]
                
                # Check if either person is already matched
                if pid1 not in matches and pid2 not in used_cam2_persons:
                    matches[pid1] = pid2
                    used_cam2_persons.add(pid2)
                    print(f"  Matched {cam1_name} person {pid1} -> {cam2_name} person {pid2} (error: {error:.2f})")
            
            pairwise_matches[(cam1_name, cam2_name)] = matches
        
        return pairwise_matches

    def create_global_person_mapping(pairwise_matches, cam_names, person_sequences):
        """Create global person IDs from pairwise matches."""
        print("Creating global person mapping...")
        
        # Create graph of person connections
        from collections import defaultdict, deque
        
        # Node format: (camera_name, person_id)
        graph = defaultdict(set)
        
        # Add edges from pairwise matches
        for (cam1, cam2), matches in pairwise_matches.items():
            for pid1, pid2 in matches.items():
                node1 = (cam1, pid1)
                node2 = (cam2, pid2)
                graph[node1].add(node2)
                graph[node2].add(node1)
        
        # Find connected components using BFS
        visited = set()
        global_person_id = 0
        global_mapping = {}  # {(cam_name, person_id): global_person_id}
        
        for cam_name in cam_names:
            if cam_name not in person_sequences:
                continue
            for person_id in person_sequences[cam_name].keys():
                node = (cam_name, person_id)
                
                if node not in visited:
                    # BFS to find all connected nodes
                    component = []
                    queue = deque([node])
                    visited.add(node)
                    
                    while queue:
                        current = queue.popleft()
                        component.append(current)
                        
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    
                    # Assign same global ID to all nodes in this component
                    # But only if component has minimum number of cameras
                    cameras_in_component = set(cam for cam, pid in component)
                    
                    if len(cameras_in_component) >= min_matched_cameras:
                        for cam, pid in component:
                            global_mapping[(cam, pid)] = global_person_id
                        print(f"  Global person {global_person_id}: {component}")
                        global_person_id += 1
                    else:
                        print(f"  Rejected component (insufficient cameras): {component}")
        
        return global_mapping

    def create_matched_poses_output(person_sequences, global_mapping, pose_data, cam_names):
        """Create the matched_poses.json output format."""
        print("Creating matched poses output...")
        
        # Determine total number of frames
        max_frames = 0
        for cam_data in pose_data.values():
            max_frames = max(max_frames, len(cam_data))
        
        matched_results = []
        
        for frame_idx in range(max_frames):
            frame_result = {
                'frame_id': frame_idx,
                'matched_instances': []
            }
            
            # Group instances by global person ID for this frame
            global_instances = defaultdict(list)  # {global_person_id: [instance, ...]}
            
            for cam_name in cam_names:
                if cam_name not in person_sequences:
                    continue
                    
                for person_id, sequence in person_sequences[cam_name].items():
                    # Check if this person has a global mapping
                    if (cam_name, person_id) not in global_mapping:
                        continue
                        
                    global_pid = global_mapping[(cam_name, person_id)]
                    
                    # Find frame data for this frame_idx
                    frame_data = None
                    for item in sequence:
                        if item['frame_idx'] == frame_idx:
                            frame_data = item
                            break
                    
                    if frame_data is not None:
                        instance = {
                            'camera_name': cam_name,
                            'original_person_id': person_id,
                            'keypoints': frame_data['original_keypoints'].tolist(),
                            'confidence_scores': frame_data['original_confidence_scores'].tolist(),
                            'bbox': frame_data['bbox']
                        }
                        global_instances[global_pid].append(instance)
            
            # Create matched instances
            for global_pid, instances in global_instances.items():
                matched_instance = {
                    'person_id': global_pid,
                    'instances': instances,
                    'cameras_matched': [inst['camera_name'] for inst in instances]
                }
                frame_result['matched_instances'].append(matched_instance)
            
            matched_results.append(frame_result)
        
        return matched_results

    # Main processing
    print("Loading pose data...")
    pose_data = load_pose_data(pose_output_dir, cam_names)
    
    print("Extracting person sequences...")
    person_sequences = extract_person_sequences(pose_data, cam_names, confidence_threshold)
    
    # Print statistics
    print("Person sequences found:")
    total_persons = 0
    for cam_name, persons in person_sequences.items():
        print(f"  {cam_name}: {len(persons)} persons")
        total_persons += len(persons)
    
    if total_persons == 0:
        print("No person sequences found!")
        return None
    
    print("Matching person sequences across cameras...")
    pairwise_matches = match_person_sequences_across_cameras(
        person_sequences, cam_names, cam_matrices, 
        cam_intrinsics, cam_extrinsics, cam_distortions, 
        reprojection_threshold
    )
    
    print("Creating global person mapping...")
    global_mapping = create_global_person_mapping(pairwise_matches, cam_names, person_sequences)
    
    print("Creating matched poses output...")
    matched_results = create_matched_poses_output(person_sequences, global_mapping, pose_data, cam_names)
    
    # Save results
    os.makedirs(matched_output_dir, exist_ok=True)
    output_file = os.path.join(matched_output_dir, 'matched_poses.json')
    with open(output_file, 'w') as f:
        json.dump(matched_results, f, indent=2)
    
    # Save summary statistics
    summary_file = os.path.join(matched_output_dir, 'matching_summary.json')
    
    # Calculate statistics
    valid_global_persons = set(global_mapping.values())
    person_stats = {}
    
    for global_pid in valid_global_persons:
        cameras = set()
        frame_count = 0
        
        for (cam_name, person_id), gpid in global_mapping.items():
            if gpid == global_pid:
                cameras.add(cam_name)
                if cam_name in person_sequences and person_id in person_sequences[cam_name]:
                    frame_count = max(frame_count, len(person_sequences[cam_name][person_id]))
        
        person_stats[str(global_pid)] = {
            'cameras_seen': sorted(cameras),
            'camera_count': len(cameras),
            'max_frame_appearances': frame_count
        }
    
    summary_data = {
        'total_frames': max([len(cam_data) for cam_data in pose_data.values()]) if pose_data else 0,
        'total_cameras': len(cam_names),
        'camera_names': cam_names,
        'valid_person_ids': sorted(valid_global_persons),
        'person_id_statistics': person_stats,
        'pairwise_matches': {f"{k[0]}-{k[1]}": v for k, v in pairwise_matches.items()},
        'global_mapping': {f"{k[0]}_{k[1]}": v for k, v in global_mapping.items()},
        'filtering_criteria': {
            'min_matched_cameras': min_matched_cameras,
            'confidence_threshold': confidence_threshold,
            'reprojection_threshold': reprojection_threshold
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Sequence matching complete!")
    print(f"Found {len(valid_global_persons)} global persons across {len(cam_names)} cameras")
    print(f"Results saved to: {output_file}")
    print(f"Summary statistics saved to: {summary_file}")
    
    return output_file

