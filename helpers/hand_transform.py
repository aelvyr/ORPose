import numpy as np

def orient_canonical_hand(canonical_hand, body_pose_3d, side='right'):
    """
    Orient the canonical hand pose based on arm orientation from body pose.
    
    Args:
        canonical_hand: numpy array of shape (num_keypoints_hands, 3) representing the canonical hand pose
        body_pose_3d: numpy array of shape (num_keypoints, 3) representing the body pose
        side: str, 'right' or 'left' indicating which arm to use
    
    Returns:
        oriented_hand: numpy array of shape (num_keypoints_hands, 3) with oriented hand pose
    """
    # Get indices for arm keypoints based on side
    if side == 'right':
        shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10  # Right arm indices
    else:
        shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9   # Left arm indices
        
    # Extract arm keypoints
    shoulder = body_pose_3d[shoulder_idx]
    elbow = body_pose_3d[elbow_idx]
    wrist = body_pose_3d[wrist_idx]
    
    # Calculate arm vectors
    forearm = wrist - elbow  # Vector from elbow to wrist
    upper_arm = elbow - shoulder  # Vector from shoulder to elbow
    
    # Create orthonormal coordinate system
    y_axis = forearm / np.linalg.norm(forearm)  # Forward direction along forearm

    # Ensure z_axis points inward (towards the body)
    z_axis = np.cross(upper_arm, forearm)
    z_axis = z_axis / np.linalg.norm(z_axis)  # Normalized vector perpendicular to forearm and upper arm

    # If this is a left hand, flip the x-axis to maintain proper handedness
    if side == 'right':
        z_axis = -z_axis

        # x_axis should be perpendicular to y_axis and lie in the arm plane
        x_axis = np.cross(y_axis, z_axis)       

    else:
        # x_axis should be perpendicular to y_axis and lie in the arm plane
        x_axis = - np.cross(y_axis, z_axis)

    x_axis = x_axis / np.linalg.norm(x_axis)
        
    # Create rotation matrix
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=0)
    
    # Rotate canonical hand
    oriented_hand = canonical_hand @ rotation_matrix
    
    # Translate to wrist position
    oriented_hand = oriented_hand + wrist
    
    return oriented_hand

def update_wrist_position(body_pose_3d, optimized_hand_pose, side='left'):
    """
    Update wrist position in body pose based on optimized hand pose.
    """
    # Create a copy of the body pose
    updated_body_pose = body_pose_3d.copy()
    
    # Get indices for arm keypoints based on side
    if side == 'right':
        wrist_idx = 10  # Right arm indices
    else:
        wrist_idx =  9   # Left arm indices
    
    updated_body_pose[wrist_idx] = optimized_hand_pose[0]  # Update wrist position
    
    return updated_body_pose

def initialize_hand_pose_for_next_frame(current_hand_pose, next_body_pose, prev_body_pose, side='left'):
    """
    Initialize hand pose for the next frame by moving it to the new wrist position
    while preserving the hand orientation relative to the forearm.
    """
    # Create a copy of the current hand pose
    initialized_hand = current_hand_pose.copy()
    
    # Get indices based on side
    if side == 'right':
        wrist_idx = 10  # Right wrist index
        elbow_idx = 8   # Right elbow index
    else:
        wrist_idx = 9   # Left wrist index
        elbow_idx = 7   # Left elbow index
    
    # Get wrist and elbow positions
    prev_wrist = prev_body_pose[wrist_idx]
    prev_elbow = prev_body_pose[elbow_idx]
    next_wrist = next_body_pose[wrist_idx]
    next_elbow = next_body_pose[elbow_idx]

    wrist_diff = next_wrist - prev_wrist
    
    # Calculate previous and next forearm directions
    prev_forearm = prev_wrist - prev_elbow
    prev_forearm = prev_forearm / np.linalg.norm(prev_forearm)
    
    next_forearm = next_wrist - next_elbow
    next_forearm = next_forearm / np.linalg.norm(next_forearm)
    
    # Calculate rotation from previous to next forearm direction
    rotation = rotation_matrix_from_vectors(prev_forearm, next_forearm)
    
    # Move hand to origin (relative to wrist)
    hand_relative = initialized_hand - current_hand_pose[0]
    
    # Rotate hand
    hand_rotated = np.dot(hand_relative, rotation.T)
    
    # Move hand to new wrist position
    initialized_hand = hand_rotated + current_hand_pose[0] + wrist_diff

    if np.isnan(initialized_hand).any():
        print("Warning: NaN detected in initialized hand pose. Returning current hand pose.")
        return current_hand_pose
    
    return initialized_hand

def rotation_matrix_from_vectors(vec1, vec2):
    """Calculate rotation matrix that rotates vec1 to vec2"""
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(v)
    
    if s == 0:
        return np.eye(3)
        
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
