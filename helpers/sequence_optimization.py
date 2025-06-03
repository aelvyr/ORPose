import os
import cv2
import torch
import numpy as np
from helpers.weakloss import BMCLoss
from helpers.definitions import num_keypoints_hands
from helpers.pose_construction import match_hand_poses
from helpers.hand_transform import initialize_hand_pose_for_next_frame
from helpers.hand_transform import orient_canonical_hand

def optimize_sequence_torch(
    initial_poses_sequence, 
    hand_poses_2d, 
    camera_intrinsics, 
    camera_extrinsics, 
    distortion_coeffs, 
    poses_3d_body,
    lambdas=[1.0, 1.0, 1.0, 1.0, 1.0],  # [reprojection, temporal, spatial, bmc, shape]
    use_bmc=False,
    max_iter=50,
    hand_id_mapping=None,
    scale_by_bbox=True
):
    """
    Optimize the entire sequence of 3D hand poses at once using PyTorch and LBFGS.
    
    Args:
        initial_poses_sequence: Initial 3D hand poses for all frames [num_frames, num_keypoints*2, 3]
        hand_poses_2d: Dictionary of 2D hand poses for each camera and frame
        camera_intrinsics: Dictionary of camera intrinsic matrices
        camera_extrinsics: Dictionary of camera extrinsic parameters
        distortion_coeffs: Dictionary of camera distortion coefficients
        poses_3d_body: 3D body poses for the sequence
        lambdas: Weights for different loss components:
                - lambdas[0]: Reprojection error weight (default: 1.0)
                - lambdas[1]: Temporal consistency weight (default: 1.0)
                    This includes two components:
                    1. Acceleration-based consistency: enforces smooth changes in velocity
                    2. Position-based consistency: directly limits large position changes between frames
                - lambdas[2]: Spatial consistency weight (default: 1.0)
                    This includes two components:
                    1. Reference-based consistency: enforces consistent bone lengths relative to reference frame
                    2. Inter-frame consistency: minimizes variance of bone lengths across the sequence
                - lambdas[3]: BMC constraint weight (default: 1.0)
                - lambdas[4]: Shape preservation weight (default: 1.0)
        use_bmc: Whether to use BMC constraints (default: False)
        max_iter: Maximum number of LBFGS iterations
        hand_id_mapping: Dictionary mapping camera names to hand side mappings
        scale_by_bbox: Whether to scale reprojection errors by bounding box size
        
    Returns:
        Optimized 3D hand poses for the sequence
    """
    print(f"Optimizing sequence with {len(initial_poses_sequence)} frames...")
    
    # Initialize BMC loss if needed
    if use_bmc and lambdas[3] > 0:
        bmc = BMCLoss(1, 1, 1)
    
    # Convert inputs to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Flatten the entire sequence for optimization
    initial_sequence_flat = initial_poses_sequence.reshape(-1)
    
    # Initialize the optimization variable as a PyTorch tensor requiring gradients
    x_tensor = torch.tensor(initial_sequence_flat, dtype=torch.float32, device=device, requires_grad=True)
    
    # Precompute number of frames and keypoints
    num_frames = len(initial_poses_sequence)
    total_keypoints = initial_poses_sequence.shape[1]  # Total for both hands
    
    # Setup camera parameters as tensors
    camera_params = {}
    for cam_name in hand_poses_2d.keys():
        intr = torch.tensor(camera_intrinsics[cam_name].reshape(3, 3), dtype=torch.float32, device=device)
        rotation = torch.tensor(camera_extrinsics[cam_name][0], dtype=torch.float32, device=device)
        translation = torch.tensor(camera_extrinsics[cam_name][1], dtype=torch.float32, device=device)
        distortion = torch.tensor(distortion_coeffs[cam_name], dtype=torch.float32, device=device)
        
        camera_params[cam_name] = {
            'intrinsics': intr,
            'rotation': rotation,
            'translation': translation,
            'distortion': distortion
        }
    
    # Define the project_points_torch function for differentiable projection
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
        
        # Apply radial distortion
        r2 = x*x + y*y
        radial = 1.0 + distortion[0]*r2 + distortion[1]*(r2*r2)
        
        xd = x * radial
        yd = y * radial
        
        # Apply tangential distortion
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
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return result
      # Define helper function to handle NaN in loss
    def safe_loss(loss_tensor, name=""):
        """Handle NaN and Inf values in loss tensors"""
        if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
            print(f"Warning: NaN or Inf detected in {name} loss")
            # Return a safe value or a high penalty value
            return torch.tensor(1000.0, device=loss_tensor.device, dtype=loss_tensor.dtype)
        return loss_tensor
    
    # Define the loss function
    def loss_fn(x_param, verbose=False):
        # Reshape the flattened parameter tensor to recover the sequence
        sequence = x_param.reshape(num_frames, total_keypoints, 3)
        
        # Check for NaN or Inf in the input parameters
        if verbose and not safe_tensor(x_param, "input parameters"):
            print("Input parameters contain NaN or Inf values")
            
        total_loss = 0.0
        loss_components = {'reprojection': 0.0, 'temporal': 0.0, 'spatial': 0.0, 'bmc': 0.0, 'shape': 0.0}
        
        # Calculate reprojection error for all frames
        if lambdas[0] > 0:  # Reprojection weight
            repr_loss = 0.0
            total_score = 0.0
            
            for frame_idx in range(num_frames):
                current_pose = sequence[frame_idx]
                left_hand = current_pose[:num_keypoints_hands]
                right_hand = current_pose[num_keypoints_hands:]
                
                # Process each camera
                for cam_name, hand_pose_2d in hand_poses_2d.items():
                    if frame_idx >= len(hand_pose_2d) or not hand_pose_2d[frame_idx]:
                        continue
                    
                    # Get camera parameters
                    params = camera_params[cam_name]
                    
                    # Check if we have a predefined mapping for this camera
                    if hand_id_mapping is not None and cam_name in hand_id_mapping:
                        # Use predefined mapping instead of matching
                        for obj_id, hand_side in hand_id_mapping[cam_name].items():
                            if obj_id in hand_pose_2d[frame_idx]:
                                hand_data = hand_pose_2d[frame_idx][obj_id]
                                kp_torch = torch.tensor(hand_data.keypoints.squeeze(), dtype=torch.float32, device=device)
                                score_torch = torch.tensor(hand_data.keypoint_scores.squeeze(), dtype=torch.float32, device=device)
                                
                                # Calculate bounding box size for scaling the reprojection error
                                valid_keypoints = kp_torch[score_torch > 0.1]
                                if len(valid_keypoints) > 0 and scale_by_bbox:
                                    min_x = torch.min(valid_keypoints[:, 0])
                                    max_x = torch.max(valid_keypoints[:, 0])
                                    min_y = torch.min(valid_keypoints[:, 1])
                                    max_y = torch.max(valid_keypoints[:, 1])
                                    bbox_width = max_x - min_x
                                    bbox_height = max_y - min_y
                                    bbox_size = torch.sqrt(bbox_width * bbox_height)
                                else:
                                    bbox_size = torch.tensor(100.0, device=device)  # Default value if no valid keypoints
                                
                                # Project the appropriate hand
                                if hand_side == 0:  # Left hand
                                    projected_points = project_points_torch(
                                        left_hand, params['intrinsics'], params['rotation'], 
                                        params['translation'], params['distortion']
                                    )
                                else:  # Right hand
                                    projected_points = project_points_torch(
                                        right_hand, params['intrinsics'], params['rotation'], 
                                        params['translation'], params['distortion']
                                    )
                                
                                # Calculate reprojection error with valid keypoints
                                valid_mask = score_torch > 0.1
                                
                                if torch.any(valid_mask):
                                    # Calculate distance between projected and observed keypoints
                                    error = torch.norm(
                                        projected_points[valid_mask] - kp_torch[valid_mask],
                                        dim=1
                                    )
                                      # Scale error by bounding box size
                                    # Make sure bbox_size is not too small
                                    bbox_size = torch.max(bbox_size, torch.tensor(1.0, device=device))
                                    scaled_error = error / bbox_size * 100.0  # Multiply by 100 to maintain similar scale
                                    
                                    # Check for NaNs in the error calculation
                                    if verbose and not safe_tensor(scaled_error, "scaled reprojection error"):
                                        print(f"Frame {frame_idx}, Camera {cam_name}, Hand side {hand_side}")
                                    
                                    # Weight error by keypoint scores
                                    weighted_error = scaled_error * score_torch[valid_mask]
                                    repr_loss += torch.sum(weighted_error)
                                    total_score += torch.sum(score_torch[valid_mask])
              # Avoid division by zero
            if total_score > 0:
                repr_loss = repr_loss / total_score
                # Check for NaN in reprojection loss
                if torch.isnan(repr_loss).any() or torch.isinf(repr_loss).any():
                    print(f"Warning: NaN or Inf detected in reprojection loss")
                    repr_loss = torch.tensor(1000.0, device=device)
                loss_components['reprojection'] = repr_loss.item()
                total_loss += lambdas[0] * repr_loss
        
        # Calculate temporal consistency loss (smooth motion between frames)
        if lambdas[1] > 0 and num_frames > 1:  # Temporal weight
            temp_loss = 0.0
            position_diff_loss = 0.0  # New component for direct position differences
            
            for frame_idx in range(1, num_frames):
                # Calculate position difference (direct change between consecutive frames)
                pos_diff = sequence[frame_idx] - sequence[frame_idx-1]
                position_diff_loss += torch.norm(pos_diff)
                
                # Calculate acceleration (second derivative)
                if frame_idx > 1:
                    prev_vel = sequence[frame_idx-1] - sequence[frame_idx-2]
                    curr_vel = sequence[frame_idx] - sequence[frame_idx-1]
                    accel = curr_vel - prev_vel
                    temp_loss += torch.norm(accel)
                else:
                    # For the first frame pair, just use velocity
                    velocity = sequence[frame_idx] - sequence[frame_idx-1]
                    temp_loss += torch.norm(velocity)
            
            # Normalize by number of frames
            temp_loss = temp_loss / (num_frames - 1)
            position_diff_loss = position_diff_loss / (num_frames - 1)
            
            # Combine acceleration-based loss with direct position difference loss
            # Weight them equally (0.5 each) within the temporal term
            combined_temp_loss = 0.5 * temp_loss + 0.5 * position_diff_loss
            
            loss_components['temporal'] = combined_temp_loss.item()
            total_loss += lambdas[1] * combined_temp_loss
            
        # Calculate spatial consistency loss (maintain bone lengths and joint angles)
        if lambdas[2] > 0:  # Spatial weight
            spatial_loss = 0.0
            
            # Define pairs of connected joints for bone length constraints
            # This is a simplified example; you may want to define more precise bone connections
            bone_connections = [
                # Left hand (5 fingers x 4 bones)
                (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),     # Index
                (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                
                # Right hand (5 fingers x 4 bones)
                (21, 22), (22, 23), (23, 24), (24, 25),     # Thumb
                (21, 26), (26, 27), (27, 28), (28, 29),     # Index
                (21, 30), (30, 31), (31, 32), (32, 33),     # Middle
                (21, 34), (34, 35), (35, 36), (36, 37),     # Ring
                (21, 38), (38, 39), (39, 40), (40, 41),     # Pinky
            ]
            
            # Two components for spatial consistency:
            # 1. Reference-based consistency (use first frame as reference)
            # 2. Inter-frame consistency (enforce constant bone lengths between consecutive frames)
            
            # Calculate reference bone lengths from first frame
            reference_frame = sequence[0]
            reference_lengths = {}
            
            for connection in bone_connections:
                start_idx, end_idx = connection
                start_point = reference_frame[start_idx]
                end_point = reference_frame[end_idx]
                bone_length = torch.norm(end_point - start_point)
                reference_lengths[connection] = bone_length
            
            # 1. Reference-based consistency: enforce consistent bone lengths relative to reference
            reference_loss = 0.0
            for frame_idx in range(num_frames):
                frame_pose = sequence[frame_idx]
                
                for connection, ref_length in reference_lengths.items():
                    start_idx, end_idx = connection
                    start_point = frame_pose[start_idx]
                    end_point = frame_pose[end_idx]
                    current_length = torch.norm(end_point - start_point)
                    
                    # Squared difference from reference length
                    reference_loss += (current_length - ref_length)**2
            
            # 2. Inter-frame consistency: ensure bone lengths remain constant across the sequence
            interframe_loss = 0.0
            
            # For each bone, calculate variance of its length across all frames
            bone_lengths_across_frames = {}
            
            # First collect all bone lengths
            for connection in bone_connections:
                bone_lengths_across_frames[connection] = []
                
                for frame_idx in range(num_frames):
                    frame_pose = sequence[frame_idx]
                    start_idx, end_idx = connection
                    start_point = frame_pose[start_idx]
                    end_point = frame_pose[end_idx]
                    current_length = torch.norm(end_point - start_point)
                    bone_lengths_across_frames[connection].append(current_length)
            
            # Then calculate variance for each bone
            for connection, lengths in bone_lengths_across_frames.items():
                # Convert list to tensor
                lengths_tensor = torch.stack(lengths)
                
                # Calculate mean and variance
                mean_length = torch.mean(lengths_tensor)
                variance = torch.sum((lengths_tensor - mean_length)**2) / len(lengths_tensor)
                
                # Add variance to loss
                interframe_loss += variance
            
            # Combine the two losses
            spatial_loss = reference_loss / (len(bone_connections) * num_frames) + interframe_loss / len(bone_connections)
            loss_components['spatial'] = spatial_loss.item()
            total_loss += lambdas[2] * spatial_loss
        
        # Calculate BMC loss if enabled
        if use_bmc and lambdas[3] > 0:
            bmc_loss = 0.0
            
            for frame_idx in range(num_frames):
                current_pose = sequence[frame_idx]
                left_hand = current_pose[:num_keypoints_hands]
                right_hand = current_pose[num_keypoints_hands:]
                
                # Apply BMC constraints to both hands
                left_bmc, _ = bmc.compute_loss(torch.unsqueeze(torch.tensor(left_hand, device=device), 0))
                right_bmc, _ = bmc.compute_loss(torch.unsqueeze(torch.tensor(right_hand, device=device), 0))
                # Check for NaN or Inf in BMC losses
                if torch.isnan(left_bmc).any() or torch.isinf(left_bmc).any():
                    print(f"Warning: NaN or Inf detected in left BMC loss at frame {frame_idx}")
                    left_bmc = torch.tensor(0.0, device=device)
                if torch.isnan(right_bmc).any() or torch.isinf(right_bmc).any():
                    print(f"Warning: NaN or Inf detected in right BMC loss at frame {frame_idx}")
                    right_bmc = torch.tensor(0.0, device=device)    
                # Sum BMC losses
                bmc_loss += left_bmc + right_bmc
            
            # Normalize by number of frames
            bmc_loss = bmc_loss / num_frames
            loss_components['bmc'] = bmc_loss.item()
            total_loss += lambdas[3] * bmc_loss.item()
        # Calculate shape preservation loss (ensure hand shape doesn't change drastically between frames)
        if lambdas[4] > 0 and num_frames > 1:  # Shape preservation weight
            shape_loss = 0.0
            
            # For each frame, calculate shape preservation loss relative to previous frame
            for frame_idx in range(1, num_frames):
                # Get current and previous frame poses
                current_frame = sequence[frame_idx]
                prev_frame = sequence[frame_idx-1]
                
                # Process left hand
                current_left = current_frame[:num_keypoints_hands]
                prev_left = prev_frame[:num_keypoints_hands]
                
                # Normalize by root joint (wrist) to focus on relative positions
                current_left_norm = current_left - current_left[0]
                prev_left_norm = prev_left - prev_left[0]
                
                # Calculate shape difference (Frobenius norm of difference)
                left_shape_diff = torch.norm(current_left_norm - prev_left_norm)
                
                # Process right hand
                current_right = current_frame[num_keypoints_hands:]
                prev_right = prev_frame[num_keypoints_hands:]
                
                # Normalize by root joint (wrist)
                current_right_norm = current_right - current_right[0]
                prev_right_norm = prev_right - prev_right[0]
                
                # Calculate shape difference
                right_shape_diff = torch.norm(current_right_norm - prev_right_norm)
                
                # Add to total shape loss
                shape_loss += left_shape_diff + right_shape_diff
            
            # Normalize by number of frames
            shape_loss = shape_loss / (num_frames - 1)
            
            # Check for NaN in shape loss
            if torch.isnan(shape_loss).any() or torch.isinf(shape_loss).any():
                print(f"Warning: NaN or Inf detected in shape preservation loss")
                shape_loss = torch.tensor(0.0, device=device)  # Ignore this term if it's invalid
            
            loss_components['shape'] = shape_loss.item()
            total_loss += lambdas[4] * shape_loss

        if verbose:
            print(f"Loss components:")
            for component, value in loss_components.items():
                print(f"  {component}: {value}")
            # Add more detailed breakdown for temporal loss
            if 'temporal' in loss_components and loss_components['temporal'] > 0:
                print(f"  - acceleration component: {temp_loss.item()}")
                print(f"  - position difference component: {position_diff_loss.item()}")
            print(f"Total loss: {total_loss.item()}")
        
        return total_loss
    
    # Initialize LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [x_tensor], 
        line_search_fn='strong_wolfe',
        max_iter=max_iter,
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
    
    # Run optimization
    optimizer.step(closure)
    
    # Final loss and results
    print(f"Final loss:")
    final_loss = loss_fn(x_tensor, verbose=True)
    
    # Convert result back to numpy
    optimized_sequence = x_tensor.detach().cpu().numpy().reshape(num_frames, total_keypoints, 3)
    
    return optimized_sequence

def optimize_sequence(
    initial_poses, 
    hand_poses_2d, 
    camera_intrinsics, 
    camera_extrinsics, 
    distortion_coeffs, 
    poses_3d_body,
    hand_id_mapping=None,
    use_bmc=False,
    lambdas=[1.0, 1.0, 1.0, 1.0, 1.0]  # [reprojection, temporal, spatial, bmc, shape]
):
    """
    Optimize a sequence of hand poses using global optimization.
    
    Args:
        initial_poses: Initial hand poses for all frames (can be from triangulation or canonical)
        hand_poses_2d: Dictionary of 2D hand poses for each camera
        camera_intrinsics: Camera intrinsics
        camera_extrinsics: Camera extrinsics
        distortion_coeffs: Camera distortion coefficients
        poses_3d_body: Body poses for the sequence
        hand_id_mapping: Dictionary mapping camera names to hand ID mappings
        use_bmc: Whether to use BMC constraints
        lambdas: Weights for different loss components:
                - lambdas[0]: Reprojection error weight (default: 1.0)
                - lambdas[1]: Temporal consistency weight (default: 1.0)
                    This includes two components:
                    1. Acceleration-based consistency: enforces smooth changes in velocity
                    2. Position-based consistency: directly limits large position changes between frames
                - lambdas[2]: Spatial consistency weight (default: 1.0)
                    This includes two components:
                    1. Reference-based consistency: enforces consistent bone lengths relative to reference frame
                    2. Inter-frame consistency: minimizes variance of bone lengths across the sequence
                - lambdas[3]: BMC constraint weight (default: 1.0)
                - lambdas[4]: Shape preservation weight (default: 1.0)

    Returns:
        Optimized hand poses for the sequence
    """
    num_frames = len(initial_poses)  # Number of frames in the sequence
    print(f"Optimizing sequence of {num_frames} hand poses...")
    
    # Check if we need to generate initial poses
    if initial_poses is None or len(initial_poses) == 0:
        from helpers.definitions import CANONICAL_HAND_POSE_3D
        initial_sequence = []
        
        # Generate initial canonical hand poses based on body model's wrists
        for frame_idx in range(len(poses_3d_body)):
            left_hand = orient_canonical_hand(CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='left')
            right_hand = orient_canonical_hand(CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='right')
            initial_frame_pose = np.concatenate([left_hand, right_hand])
            initial_sequence.append(initial_frame_pose)
        
        initial_sequence = np.array(initial_sequence)
    else:
        initial_sequence = initial_poses
    
    # Use global sequence optimization
    optimized_sequence = optimize_sequence_torch(
        initial_sequence,
        hand_poses_2d,
        camera_intrinsics,
        camera_extrinsics,
        distortion_coeffs,
        poses_3d_body,
        lambdas=lambdas,
        use_bmc=use_bmc,
        hand_id_mapping=hand_id_mapping
    )
    
    return optimized_sequence
