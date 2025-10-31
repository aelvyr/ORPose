import os
import cv2
import torch
import numpy as np
from helpers.weakloss import BMCLoss
from helpers.definitions import num_keypoints_hands
from helpers.pose_construction import match_hand_poses

bmc = BMCLoss(1, 1, 1)

def optimize_hand_pose(initial_pose, filtered_hand_poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, previous_pose=None, lambdas=[1.0, 30.0, 50.0, 20.0, 5.0], scale_by_bbox=True, wrist_position=None):
    """
    Optimize 3D hand pose with BMC constraints using PyTorch operations and autograd.
    """
    # Convert inputs to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if wrist_position is not None:
        wrist_position_tensor = torch.tensor(wrist_position, dtype=torch.float32, device=device)
    # Initialize the optimization variable as a PyTorch tensor requiring gradients
    x_tensor = torch.tensor(initial_pose.flatten(), dtype=torch.float32, device=device, requires_grad=True)
    
    if previous_pose is not None:
        prev_pose_tensor = torch.tensor(previous_pose, dtype=torch.float32, device=device)
        prev_pose_tensor_norm = prev_pose_tensor - prev_pose_tensor[0]
    else:
        prev_pose_tensor = None
        
    # Setup camera parameters as tensors
    camera_params = {}
    for cam_name, hand_pose_2d in filtered_hand_poses_2d.items():
        intr = torch.tensor(camera_intrinsics[cam_name].reshape(3, 3), dtype=torch.float32, device=device)
        rotation = torch.tensor(camera_extrinsics[cam_name][0], dtype=torch.float32, device=device)
        translation = torch.tensor(camera_extrinsics[cam_name][1], dtype=torch.float32, device=device)
        distortion = torch.tensor(distortion_coeffs[cam_name], dtype=torch.float32, device=device)
        
        keypoints = torch.tensor(hand_pose_2d.keypoints.squeeze(), dtype=torch.float32, device=device)
        keypoint_scores = torch.tensor(hand_pose_2d.keypoint_scores.squeeze(), dtype=torch.float32, device=device)
        
        # Calculate bounding box size for scaling the reprojection error
        valid_keypoints = keypoints[keypoint_scores > 0.1]
        if len(valid_keypoints) > 0 and scale_by_bbox:
            min_x = torch.min(valid_keypoints[:, 0])
            max_x = torch.max(valid_keypoints[:, 0])
            min_y = torch.min(valid_keypoints[:, 1])
            max_y = torch.max(valid_keypoints[:, 1])
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            bbox_size = torch.sqrt(bbox_width * bbox_height)
        else:
            bbox_size = torch.tensor(0.0, device=device)  # Default value if no valid keypoints
        
        camera_params[cam_name] = {
            'intrinsics': intr,
            'rotation': rotation,
            'translation': translation,
            'distortion': distortion,
            'keypoints': keypoints,
            'keypoint_scores': keypoint_scores,
            'bbox_size': bbox_size
        }
        
    # Define helper functions for numerical stability
    def safe_tensor(tensor, name=""):
        """Check if tensor contains NaN or Inf values and print debug info."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN or Inf detected in {name}")
            nan_mask = torch.isnan(tensor)
            inf_mask = torch.isinf(tensor)
            if torch.any(nan_mask):
                print(f"NaN count: {torch.sum(nan_mask).item()}")
            if torch.any(inf_mask):
                print(f"Inf count: {torch.sum(inf_mask).item()}")
            return False
        return True
    
    # Define the project_points_torch function for differentiable projection
    def project_points_torch(points_3d, intrinsics, rotation, translation, distortion):
        """Differentiable projection of 3D points to 2D using PyTorch."""
        # Apply rotation and translation
        points_rotated = torch.matmul(points_3d, rotation.transpose(-1, -2)) + translation
        
        # Add stability check for z values close to zero
        z_values = points_rotated[:, 2]
        eps = 1e-10
        z_values = torch.where(torch.abs(z_values) < eps, 
                               torch.ones_like(z_values) * eps * torch.sign(z_values).clamp(min=1.0), 
                               z_values)
        
        # Project to image plane (normalized coordinates)
        x = points_rotated[:, 0] / z_values
        y = points_rotated[:, 1] / z_values
        
        # Apply radial distortion (simplified, can be expanded)
        r2 = x*x + y*y
        radial = 1.0 + distortion[0]*r2 + distortion[1]*(r2*r2)
        
        xd = x * radial
        yd = y * radial
        
        # Apply tangential distortion (simplified)
        tangential_x = 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
        tangential_y = distortion[2] * (r2 + 2 * y * y) + 2 * distortion[3] * x * y
        
        xd = xd + tangential_x
        yd = yd + tangential_y
        
        # Apply camera matrix
        u = intrinsics[0, 0] * xd + intrinsics[0, 2]
        v = intrinsics[1, 1] * yd + intrinsics[1, 2]
        
        result = torch.stack([u, v], dim=-1)
        
        # Validate result for NaN values
        if not safe_tensor(result, "projected points"):
            # Replace NaN values with a default value to allow optimization to continue
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
        return result
    
    # Define the loss function with PyTorch operations
    def loss_fn(x_param, verbose=False):
        x_reshaped = x_param.reshape(-1, 3)
        
        # Check for NaN or Inf in the input parameters
        if verbose and not safe_tensor(x_param, "optimization parameters"):
            print("Input parameters contain NaN or Inf values")
        
        # Reprojection loss
        repr_loss = torch.tensor(0.0, device=device)
        total_score = torch.tensor(0.0, device=device)
        num_cams = 0
        
        for cam_name, params in camera_params.items():
            # Project 3D points to 2D using our PyTorch-based function
            projected_points = project_points_torch(
                x_reshaped, 
                params['intrinsics'], 
                params['rotation'], 
                params['translation'], 
                params['distortion']
            )
            
            # Calculate reprojection error with valid keypoints
            valid_mask = params['keypoint_scores'] > 0.1
            valid_projection = (projected_points > 0.0).any(dim=1)
            valid_mask = valid_mask & valid_projection

            if torch.any(valid_mask) and params['bbox_size'] > 0.0:
                num_cams += 1
                # Calculate distance between projected and observed keypoints
                error = torch.norm(
                    projected_points[valid_mask] - params['keypoints'][valid_mask],
                    dim=1
                )
                
                # Scale error by bounding box size to make it fair for different distances
                # Make sure bbox_size is not too small to avoid division by very small numbers
                bbox_size = torch.max(params['bbox_size'], torch.tensor(1.0, device=device))
                scaled_error = error / bbox_size * 100.0  # Multiply by 100 to maintain similar scale
                valid_errors = ~torch.isnan(scaled_error)
                #valid_errors = valid_errors & (scaled_error < 500.0)

                # Check for NaNs in the error calculation
                if verbose and not safe_tensor(scaled_error, f"scaled error for camera {cam_name}"):
                    print(f"Error values causing NaN: {error}")
                    print(f"Bbox size: {bbox_size}")
                
                # Weight error by keypoint scores
                weighted_error = scaled_error[valid_errors] * params['keypoint_scores'][valid_mask][valid_errors]
                repr_loss += torch.sum(weighted_error)
                total_score += torch.sum(params['keypoint_scores'][valid_mask][valid_errors])
        
        # Avoid division by zero
        if total_score > 0 and num_cams >= 2:
            repr_loss = repr_loss / total_score
            # Check for NaN in final reprojection loss
            if torch.isnan(repr_loss).any() or torch.isinf(repr_loss).any():
                print(f"Warning: NaN or Inf detected in reprojection loss")
                repr_loss = torch.tensor(1000.0, device=device)  # Replace with a high value
        else:
            repr_loss = torch.tensor(1000.0, device=device)  # Replace with a high value
            print("Warning: total_score is zero, no valid keypoints found")
        
        # Temporal consistency loss
        temp_loss = torch.tensor(0.0, device=device)
        # Shape consistency loss
        shape_loss = torch.tensor(0.0, device=device)
        if prev_pose_tensor is not None:
            x_reshaped_norm = x_reshaped - x_reshaped[0]
            if lambdas[1] > 0:
                temp_loss += torch.norm(x_reshaped - prev_pose_tensor)
                
                temp_loss += torch.norm(x_reshaped_norm - prev_pose_tensor_norm)
                                    

            if lambdas[2] > 0:
                # Compute covariance matrix
                H = torch.matmul(x_reshaped_norm.transpose(-1, -2), prev_pose_tensor_norm)
                
                # SVD decomposition
                U, _, V = torch.svd(H)
                
                # Compute rotation matrix
                R = torch.matmul(V, U.transpose(-1, -2))
                
                # Handle reflection case
                det = torch.det(R)
                if det < 0:
                    V_adj = V.clone()
                    V_adj[:, -1] = -V_adj[:, -1]
                    R = torch.matmul(V_adj, U.transpose(-1, -2))
                
                # Apply rotation to align current pose with previous pose
                aligned_current = torch.matmul(x_reshaped_norm, R.transpose(-1, -2))
                
                # Calculate shape loss as the norm between aligned poses
                shape_loss += torch.norm(aligned_current - prev_pose_tensor_norm)

        if wrist_position is not None and lambdas[3] > 0:
            wrist_loss = torch.norm(x_reshaped[0] - wrist_position_tensor)
        else:
            wrist_loss = torch.tensor(0.0, device=device)

        # BMC loss
        if lambdas[4] > 0:
            bmc_loss, _ = bmc.compute_loss(x_reshaped.unsqueeze(0))
        else:
            bmc_loss = torch.tensor(0.0, device=device)

        # Check for NaN in BMC loss 
        if torch.isnan(bmc_loss).any() or torch.isinf(bmc_loss).any():
            print(f"Warning: NaN or Inf detected in BMC loss")
            bmc_loss = torch.tensor(0.0, device=device)
        
        # Combine all losses with weights
        total_loss = lambdas[0] * repr_loss + lambdas[1] * temp_loss + lambdas[2] * shape_loss +  lambdas[3] * wrist_loss + lambdas[4] * bmc_loss

        if verbose:
            if type(repr_loss) == float:
                print(f"Reprojection loss: {repr_loss}")
            else:
                print(f"Reprojection loss: {repr_loss.item()}")
            print(f"Temporal loss: {temp_loss.item() if torch.is_tensor(temp_loss) else temp_loss}")
            print(f"Shape loss: {shape_loss.item() if torch.is_tensor(shape_loss) else shape_loss}")
            print(f"Wrist loss: {wrist_loss.item() if torch.is_tensor(wrist_loss) else wrist_loss}")
            print(f"BMC loss: {bmc_loss.item() if torch.is_tensor(bmc_loss) else bmc_loss}")
            print(f"Total loss: {total_loss.item()}")
        
        return total_loss
    
    # Initialize LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [x_tensor], 
        lr=1,
        line_search_fn='strong_wolfe',
        max_iter=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )
    
    # Initial loss calculation
    print("Initial Loss:")
    initial_loss = loss_fn(x_tensor, verbose=True)
    
    # Define closure function for optimizer
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(x_tensor)
        loss.backward()
        return loss
    
    for i in range(10):
        # Run optimization
        optimizer.step(closure)
    
    # Final loss and results
    print(f"Final loss:")
    final_loss = loss_fn(x_tensor, verbose=True)
    
    # Convert result back to numpy
    optimized_hand = x_tensor.detach().cpu().numpy().reshape(-1, 3)
    
    return optimized_hand, final_loss

def optimize_poses(initial_hand_poses_3d, hand_poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, frame_idx=0, previous_pose=None, visualize=False, input_dir=None, hand_id_mapping=None, lambdas=[1.0, 50.0, 50.0, 50.0, 5.0], body_pose=None):
    """
    Optimize 3D hand poses by minimizing reprojection error across multiple views.
    
    Args:
        initial_hand_poses_3d: Initial 3D hand poses
        hand_poses_2d: Dictionary of 2D hand poses
        camera_intrinsics: Camera intrinsic parameters
        camera_extrinsics: Camera extrinsic parameters
        distortion_coeffs: Camera distortion coefficients
        frame_idx: Current frame index
        previous_pose: Previous optimized pose for temporal consistency
        visualize: Whether to visualize the matching process
        input_dir: Directory containing input videos (required if visualize=True)
        hand_id_mapping: Dictionary mapping camera names to a dict of object IDs to hand side (0=left, 1=right)
    """
    optimized_hands = []
    filtered_hand_poses_2d1 = {}  # Left hand
    filtered_hand_poses_2d2 = {}  # Right hand
    
    if previous_pose is not None:
        previous_pose1 = previous_pose[:num_keypoints_hands]  # Left hand
        previous_pose2 = previous_pose[num_keypoints_hands:]  # Right hand
    else:
        previous_pose1 = None
        previous_pose2 = None

    for cam_name, hand_pose_2d in hand_poses_2d.items():
        # Check if we have a predefined mapping for this camera
        cam_mapping = None if hand_id_mapping is None else hand_id_mapping.get(cam_name)
        
        if cam_mapping is not None:
            # Use predefined mapping instead of matching
            for obj_id, hand_side in cam_mapping.items():
                if obj_id in hand_pose_2d[frame_idx]:
                    if hand_side == 0:  # Left hand
                        filtered_hand_poses_2d1[cam_name] = hand_pose_2d[frame_idx][obj_id]
                    elif hand_side == 1:  # Right hand
                        filtered_hand_poses_2d2[cam_name] = hand_pose_2d[frame_idx][obj_id]
        else:
            # No mapping available, perform matching
            if visualize and input_dir is not None:
                # Visualize the hand poses on the original frame
                video_path = os.path.join(input_dir, cam_name + '_synced_cut.MP4')
                cap = cv2.VideoCapture(video_path)
                 
                if not cap.isOpened():
                    print(f"Error opening video file: {video_path}")
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                res, frame = cap.read()
                cap.release()
                if frame is None:
                    print(f"Frame not found: {frame_idx}")
                    continue

                (best_match1, min_error1), (best_match2, min_error2), vis_frame = match_hand_poses(
                    initial_hand_poses_3d, hand_pose_2d[frame_idx], camera_intrinsics[cam_name],
                    camera_extrinsics[cam_name][0], camera_extrinsics[cam_name][1],
                    distortion_coeffs[cam_name], just_wrist=False, visualize=visualize, frame=frame
                )
                print(f"Best match for left hand: {best_match1}, error: {min_error1}")
                print(f"Best match for right hand: {best_match2}, error: {min_error2}")
                cv2.namedWindow("Matched Hands", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Matched Hands", 1920, 1080)  # Resize to a comfortable size
                cv2.imshow("Matched Hands", vis_frame)
                # Wait for user input (press any key to continue, ESC to exit)
                key = cv2.waitKey(0)  # 0 means wait indefinitely until a key is pressed
                if key == 27:  # ESC key to exit
                    cv2.destroyAllWindows()
            else:
                (best_match1, min_error1), (best_match2, min_error2) = match_hand_poses(
                    initial_hand_poses_3d, hand_pose_2d[frame_idx], camera_intrinsics[cam_name],
                    camera_extrinsics[cam_name][0], camera_extrinsics[cam_name][1],
                    distortion_coeffs[cam_name], just_wrist=False
                )
            
            # Store matched poses
            if best_match1 is not None:
                filtered_hand_poses_2d1[cam_name] = hand_pose_2d[frame_idx][best_match1]
            if best_match2 is not None:
                filtered_hand_poses_2d2[cam_name] = hand_pose_2d[frame_idx][best_match2]

    wrist_idx_right = 10
    wrist_idx_left = 9

    # Optimize hand poses
    optimized_hand1, final_loss1 = optimize_hand_pose(
        initial_hand_poses_3d[:num_keypoints_hands], filtered_hand_poses_2d1,
        camera_intrinsics, camera_extrinsics, distortion_coeffs, 
        previous_pose=previous_pose1, lambdas=lambdas,wrist_position=body_pose[wrist_idx_left]
    )

    optimized_hand2, final_loss2 = optimize_hand_pose(
        initial_hand_poses_3d[num_keypoints_hands:], filtered_hand_poses_2d2,
        camera_intrinsics, camera_extrinsics, distortion_coeffs, 
        previous_pose=previous_pose2, lambdas=lambdas, wrist_position=body_pose[wrist_idx_right]
    )

    # Combine optimized hands
    optimized_hands.append(np.concatenate([optimized_hand1, optimized_hand2], axis=0))
    return np.array(optimized_hands).squeeze(), (final_loss1 + final_loss2) / 2.0

def optimize_hand_pose_no_bmc(initial_pose, filtered_hand_poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, previous_pose=None, lambdas=[1.0, 50.0, 50.0, 50.0], scale_by_bbox=True, wrist_position=None):
    """
    Optimize 3D hand pose without BMC constraints, using only weighted reprojection error and temporal consistency.
    Uses PyTorch operations and autograd for faster optimization.
    """
    # Convert inputs to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if wrist_position is not None:
        wrist_position_tensor = torch.tensor(wrist_position, dtype=torch.float32, device=device)
    # Initialize the optimization variable as a PyTorch tensor requiring gradients
    x_tensor = torch.tensor(initial_pose.flatten(), dtype=torch.float32, device=device, requires_grad=True)
    
    if previous_pose is not None:
        prev_pose_tensor = torch.tensor(previous_pose, dtype=torch.float32, device=device)
        prev_pose_tensor_norm = prev_pose_tensor - prev_pose_tensor[0]
    else:
        prev_pose_tensor = None
        
    # Setup camera parameters as tensors
    camera_params = {}
    for cam_name, hand_pose_2d in filtered_hand_poses_2d.items():
        intr = torch.tensor(camera_intrinsics[cam_name].reshape(3, 3), dtype=torch.float32, device=device)
        rotation = torch.tensor(camera_extrinsics[cam_name][0], dtype=torch.float32, device=device)
        translation = torch.tensor(camera_extrinsics[cam_name][1], dtype=torch.float32, device=device)
        distortion = torch.tensor(distortion_coeffs[cam_name], dtype=torch.float32, device=device)
        
        try:
            keypoints = torch.tensor(hand_pose_2d.keypoints.squeeze(), dtype=torch.float32, device=device)
            keypoint_scores = torch.tensor(hand_pose_2d.keypoint_scores.squeeze(), dtype=torch.float32, device=device)

        
        except Exception as e:
            keypoints = torch.tensor(np.array(hand_pose_2d['keypoints']).squeeze(), dtype=torch.float32, device=device)
            keypoint_scores = torch.tensor(np.array(hand_pose_2d['keypoint_scores']).squeeze(), dtype=torch.float32, device=device)

        if torch.sum(keypoints[0]) > 0 and keypoint_scores[0] > 0:
            keypoint_scores[0] = 1.0  # Ensure wrist keypoint has high confidence

        # Calculate bounding box size for scaling the reprojection error
        valid_keypoints = keypoints[keypoint_scores > 0.1]
        if len(valid_keypoints) > 0 and scale_by_bbox:
            min_x = torch.min(valid_keypoints[:, 0])
            max_x = torch.max(valid_keypoints[:, 0])
            min_y = torch.min(valid_keypoints[:, 1])
            max_y = torch.max(valid_keypoints[:, 1])
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            bbox_size = torch.sqrt(bbox_width * bbox_height)
        else:
            bbox_size = torch.tensor(0.0, device=device)  # Default value if no valid keypoints
        
        camera_params[cam_name] = {
            'intrinsics': intr,
            'rotation': rotation,
            'translation': translation,
            'distortion': distortion,
            'keypoints': keypoints,
            'keypoint_scores': keypoint_scores,
            'bbox_size': bbox_size
        }
        
    # Define helper functions for numerical stability
    def safe_tensor(tensor, name=""):
        """Check if tensor contains NaN or Inf values and print debug info."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN or Inf detected in {name}")
            nan_mask = torch.isnan(tensor)
            inf_mask = torch.isinf(tensor)
            if torch.any(nan_mask):
                print(f"NaN count: {torch.sum(nan_mask).item()}")
            if torch.any(inf_mask):
                print(f"Inf count: {torch.sum(inf_mask).item()}")
            return False
        return True
    
    # Define the project_points_torch function for differentiable projection
    def project_points_torch(points_3d, intrinsics, rotation, translation, distortion):
        """Differentiable projection of 3D points to 2D using PyTorch."""
        # Apply rotation and translation
        points_rotated = torch.matmul(points_3d, rotation.transpose(-1, -2)) + translation
        
        # Add stability check for z values close to zero
        z_values = points_rotated[:, 2]
        eps = 1e-10
        z_values = torch.where(torch.abs(z_values) < eps, 
                               torch.ones_like(z_values) * eps * torch.sign(z_values).clamp(min=1.0), 
                               z_values)
        
        # Project to image plane (normalized coordinates)
        x = points_rotated[:, 0] / z_values
        y = points_rotated[:, 1] / z_values
        
        # Apply radial distortion (simplified, can be expanded)
        r2 = x*x + y*y
        radial = 1.0 + distortion[0]*r2 + distortion[1]*(r2*r2)
        
        xd = x * radial
        yd = y * radial
        
        # Apply tangential distortion (simplified)
        tangential_x = 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
        tangential_y = distortion[2] * (r2 + 2 * y * y) + 2 * distortion[3] * x * y
        
        xd = xd + tangential_x
        yd = yd + tangential_y
        
        # Apply camera matrix
        u = intrinsics[0, 0] * xd + intrinsics[0, 2]
        v = intrinsics[1, 1] * yd + intrinsics[1, 2]
        
        result = torch.stack([u, v], dim=-1)
        
        # Validate result for NaN values
        if not safe_tensor(result, "projected points"):
            # Replace NaN values with a default value to allow optimization to continue
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
        return result
    
    # Define the loss function with PyTorch operations
    def loss_fn(x_param, verbose=False):
        x_reshaped = x_param.reshape(-1, 3)
        
        # Check for NaN or Inf in the input parameters
        if verbose and not safe_tensor(x_param, "optimization parameters"):
            print("Input parameters contain NaN or Inf values")
        
        # Reprojection loss
        repr_loss = torch.tensor(0.0, device=device)
        total_score = torch.tensor(0.0, device=device)
        num_cams = 0
        
        for cam_name, params in camera_params.items():
            # Project 3D points to 2D using our PyTorch-based function
            projected_points = project_points_torch(
                x_reshaped, 
                params['intrinsics'], 
                params['rotation'], 
                params['translation'], 
                params['distortion']
            )
            
            # Calculate reprojection error with valid keypoints
            valid_mask = params['keypoint_scores'] > 0.1
            valid_projection = (projected_points > 0.0).any(dim=1)
            valid_mask = valid_mask & valid_projection

            if torch.any(valid_mask) and params['bbox_size'] > 0.0:
                num_cams += 1
                # Calculate distance between projected and observed keypoints
                error = torch.norm(
                    projected_points[valid_mask] - params['keypoints'][valid_mask],
                    dim=1
                )
                
                # Scale error by bounding box size to make it fair for different distances
                # Make sure bbox_size is not too small to avoid division by very small numbers
                bbox_size = torch.max(params['bbox_size'], torch.tensor(1.0, device=device))
                scaled_error = error / bbox_size * 100.0  # Multiply by 100 to maintain similar scale
                valid_errors = ~torch.isnan(scaled_error)
                #valid_errors = valid_errors & (scaled_error < 500.0)

                # Check for NaNs in the error calculation
                if verbose and not safe_tensor(scaled_error, f"scaled error for camera {cam_name}"):
                    print(f"Error values causing NaN: {error}")
                    print(f"Bbox size: {bbox_size}")
                
                # Weight error by keypoint scores
                weighted_error = scaled_error[valid_errors] * params['keypoint_scores'][valid_mask][valid_errors]
                repr_loss += torch.sum(weighted_error)
                total_score += torch.sum(params['keypoint_scores'][valid_mask][valid_errors])
        
        # Avoid division by zero
        if total_score > 0 and num_cams >= 2:
            repr_loss = repr_loss / total_score
            # Check for NaN in final reprojection loss
            if torch.isnan(repr_loss).any() or torch.isinf(repr_loss).any():
                print(f"Warning: NaN or Inf detected in reprojection loss")
                repr_loss = torch.tensor(1000.0, device=device)  # Replace with a high value
        else:
            repr_loss = torch.tensor(1000.0, device=device)  # Replace with a high value
            print("Warning: total_score is zero, no valid keypoints found")
        
        # Temporal consistency loss
        temp_loss = torch.tensor(0.0, device=device)
        # Shape consistency loss
        shape_loss = torch.tensor(0.0, device=device)
        if prev_pose_tensor is not None:
            x_reshaped_norm = x_reshaped - x_reshaped[0]
            if lambdas[1] > 0:
                temp_loss += torch.norm(x_reshaped - prev_pose_tensor)
                temp_loss += torch.norm(x_reshaped_norm - prev_pose_tensor_norm)

            if lambdas[2] > 0:                        
                # Compute covariance matrix
                H = torch.matmul(x_reshaped_norm.transpose(-1, -2), prev_pose_tensor_norm)
                
                # SVD decomposition
                U, _, V = torch.svd(H)
                
                # Compute rotation matrix
                R = torch.matmul(V, U.transpose(-1, -2))
                
                # Handle reflection case
                det = torch.det(R)
                if det < 0:
                    V_adj = V.clone()
                    V_adj[:, -1] = -V_adj[:, -1]
                    R = torch.matmul(V_adj, U.transpose(-1, -2))
                
                # Apply rotation to align current pose with previous pose
                aligned_current = torch.matmul(x_reshaped_norm, R.transpose(-1, -2))
                
                # Calculate shape loss as the norm between aligned poses
                shape_loss += torch.norm(aligned_current - prev_pose_tensor_norm)

        if wrist_position is not None and lambdas[3] > 0:
            wrist_loss = torch.norm(x_reshaped[0] - wrist_position_tensor)
        else:
            wrist_loss = torch.tensor(0.0, device=device)

        # Combine all losses with weights
        total_loss = lambdas[0] * repr_loss + lambdas[1] * temp_loss +  lambdas[2] * shape_loss + lambdas[3] * wrist_loss

        if verbose:
            if type(repr_loss) == float:
                print(f"Reprojection loss: {repr_loss}")
            else:
                print(f"Reprojection loss: {repr_loss.item()}")
            print(f"Temporal loss: {temp_loss}")
            print(f"Shape loss: {shape_loss}")
            print(f"Wrist loss: {wrist_loss}")
            if type(total_loss) == float:
                print(f"Total loss: {total_loss}")
            else:
                print(f"Total loss: {total_loss.item()}")

            if repr_loss == 1000.0:
                print("Invalid Configuration found!")
                return total_loss, False
            
            else:
                return total_loss, True
            

        return total_loss
      # Initialize LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [x_tensor], 
        lr=1,
        line_search_fn='strong_wolfe',
        max_iter=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )
    
    # Initial loss calculation
    print("Initial Loss (no BMC):")
    initial_loss, valid = loss_fn(x_tensor, verbose=True)

    if not valid:
        print("Invalid initial configuration found!")
        return initial_pose, initial_loss
    
    # Define closure function for optimizer
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(x_tensor)
        loss.backward()
        return loss
    
    # Run optimization
    optimizer.step(closure)
    
    # Final loss and results
    print(f"Final loss (no BMC):")
    final_loss, valid = loss_fn(x_tensor, verbose=True)
    if not valid:
        print("Invalid final configuration found!")
        return initial_pose, final_loss

    # Convert result back to numpy
    optimized_hand = x_tensor.detach().cpu().numpy().reshape(-1, 3)
    
    return optimized_hand, final_loss

def optimize_poses_no_bmc(initial_hand_poses_3d, hand_poses_2d, camera_intrinsics, camera_extrinsics, distortion_coeffs, frame_idx=0, previous_pose=None, hand_id_mapping=None, num_keypoints_hands=21, lambdas=[1.0, 50.0, 50.0, 50.0], body_pose=None):
    """
    Optimize 3D hand poses by minimizing reprojection error across multiple views without BMC constraints.
    
    Args:
        initial_hand_poses_3d: Initial 3D hand poses
        hand_poses_2d: Dictionary of 2D hand poses for each camera
        camera_intrinsics: Dictionary of camera intrinsic matrices
        camera_extrinsics: Dictionary of camera extrinsic parameters
        distortion_coeffs: Dictionary of camera distortion coefficients
        frame_idx: Current frame index
        previous_pose: Previous optimized pose for temporal consistency
        hand_id_mapping: Dictionary mapping camera names to a dict of object IDs to hand side (0=left, 1=right)
        num_keypoints_hands: Number of keypoints per hand
    """
    from helpers.pose_construction import match_hand_poses
    
    optimized_hands = []
    filtered_hand_poses_2d1 = {}  # Left hand
    filtered_hand_poses_2d2 = {}  # Right hand
    
    if previous_pose is not None:
        previous_pose1 = previous_pose[:num_keypoints_hands]  # Left hand
        previous_pose2 = previous_pose[num_keypoints_hands:]  # Right hand
    else:
        previous_pose1 = None
        previous_pose2 = None

    for cam_name, hand_pose_2d in hand_poses_2d.items():
        # Check if we have a predefined mapping for this camera
        if hand_id_mapping is not None and cam_name in hand_id_mapping:
            # Use predefined mapping instead of matching
            for obj_id, hand_side in hand_id_mapping[cam_name].items():
                if frame_idx < len(hand_pose_2d) and obj_id in hand_pose_2d[frame_idx]:
                    if hand_side == 0:  # Left hand
                        filtered_hand_poses_2d1[cam_name] = hand_pose_2d[frame_idx][obj_id]
                    elif hand_side == 1:  # Right hand
                        filtered_hand_poses_2d2[cam_name] = hand_pose_2d[frame_idx][obj_id]
        else:
            # No mapping available, perform matching
            if frame_idx < len(hand_pose_2d):
                (best_match1, min_error1), (best_match2, min_error2) = match_hand_poses(
                    initial_hand_poses_3d, hand_pose_2d[frame_idx], camera_intrinsics[cam_name],
                    camera_extrinsics[cam_name][0], camera_extrinsics[cam_name][1],
                    distortion_coeffs[cam_name], just_wrist=False
                )
                if best_match1 is not None:
                    filtered_hand_poses_2d1[cam_name] = hand_pose_2d[frame_idx][best_match1]
                if best_match2 is not None:
                    filtered_hand_poses_2d2[cam_name] = hand_pose_2d[frame_idx][best_match2]

    wrist_idx_right = 10
    wrist_idx_left = 9
    # Use our PyTorch-based optimization without BMC for both hands
    # Optimize left hand
    optimized_hand1, loss1 = optimize_hand_pose_no_bmc(
        initial_hand_poses_3d[:num_keypoints_hands], filtered_hand_poses_2d1,
        camera_intrinsics, camera_extrinsics, distortion_coeffs, 
        previous_pose=previous_pose1, lambdas=lambdas, wrist_position=body_pose[wrist_idx_left]
    )

    # Optimize right hand
    optimized_hand2, loss2 = optimize_hand_pose_no_bmc(
        initial_hand_poses_3d[num_keypoints_hands:], filtered_hand_poses_2d2,
        camera_intrinsics, camera_extrinsics, distortion_coeffs, 
        previous_pose=previous_pose2, lambdas=lambdas, wrist_position=body_pose[wrist_idx_right]
    )

    # Combine optimized hands
    combined_hands = np.concatenate([optimized_hand1, optimized_hand2], axis=0)
    return combined_hands, (loss1 + loss2)/2.0
