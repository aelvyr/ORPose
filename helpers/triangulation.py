import os
import numpy as np
import cv2
import json
from helpers.hand_transform import orient_canonical_hand
from helpers.camera import project_points
from helpers.pose_construction import match_hand_poses

def triangulate_hand_keypoints(hand_poses_2d, camera_intrinsics, camera_extrinsics, camera_matrices, distortion_coeffs, frame_idx=0, hand_id_mapping=None, num_keypoints_hands=21, multi_person=False, person_id=None):
    """
    Triangulate 3D hand keypoints from 2D observations across multiple cameras using weighted triangulation.
    
    Args:
        hand_poses_2d: Dictionary of 2D hand poses for each camera
        camera_intrinsics: Camera intrinsic matrices
        camera_extrinsics: Camera extrinsic parameters
        camera_matrices: Camera projection matrices
        distortion_coeffs: Camera distortion coefficients
        frame_idx: Current frame index
        hand_id_mapping: Optional dictionary mapping camera name to {obj_id: hand_side}, where hand_side is 0 for left and 1 for right
        num_keypoints_hands: Number of keypoints per hand
        multi_person: Whether using multi-person mode
        person_id: Person ID for multi-person mode
        
    Returns:
        Triangulated 3D hand keypoints
    """
    # Match hands to left/right wrists based on initial hand poses
    left_hand_2d = {}  # camera_name -> 2D keypoints for left hand
    right_hand_2d = {}  # camera_name -> 2D keypoints for right hand
    
    # Use provided mapping to assign hands directly
    for cam_name, hand_pose_2d in hand_poses_2d.items():
        if frame_idx >= len(hand_pose_2d) or not hand_pose_2d[frame_idx]:
            continue
            
        # Get the mapping for this camera
        cam_mapping = hand_id_mapping.get(cam_name, {})
        
        # Process each detected hand
        for obj_id, hand_data in hand_pose_2d[frame_idx].items():
            # Check if we have a valid hand detection
            
            # Check if this object ID is in the mapping
            if obj_id in cam_mapping:
                # 0 = left hand, 1 = right hand
                if cam_mapping[obj_id] == 0:
                    left_hand_2d[cam_name] = {
                        'keypoints': hand_data.keypoints.squeeze(),
                        'scores': hand_data.keypoint_scores.squeeze()
                    }
                elif cam_mapping[obj_id] == 1:
                    right_hand_2d[cam_name] = {
                        'keypoints': hand_data.keypoints.squeeze(),
                        'scores': hand_data.keypoint_scores.squeeze()
                    }
    
    # Triangulate each keypoint for left and right hand
    left_hand_3d = np.zeros((num_keypoints_hands, 3))
    right_hand_3d = np.zeros((num_keypoints_hands, 3))
    
    # Triangulate each keypoint independently
    for kp_idx in range(num_keypoints_hands):
        # Process left hand keypoints
        if len(left_hand_2d) >= 2:  # Need at least 2 cameras for triangulation
            points_2d = []
            cam_matrices = []
            weights = []
            
            for cam_name, hand_data in left_hand_2d.items():
                # Only use keypoints with sufficient confidence
                if hand_data['scores'][kp_idx] > 0.1:
                    kp = hand_data['keypoints'][kp_idx]
                    
                    # Undistort the point
                    K = camera_intrinsics[cam_name].reshape(3, 3)
                    dist = distortion_coeffs[cam_name]
                    kp_undistorted = cv2.undistortPoints(np.array([[kp]]), K, dist, None, K).reshape(2)
                    
                    points_2d.append(kp_undistorted)
                    cam_matrices.append(camera_matrices[cam_name])
                    weights.append(hand_data['scores'][kp_idx])
            
            if len(points_2d) >= 2:
                # Perform weighted triangulation
                points_2d = np.array(points_2d)
                cam_matrices = np.array(cam_matrices)
                weights = np.array(weights)
                
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Create weighted system of equations A*X = 0
                A = np.zeros((2 * len(points_2d), 4))
                for i, (point, P, w) in enumerate(zip(points_2d, cam_matrices, weights)):
                    x, y = point
                    A[2*i] = w * (x * P[2] - P[0])
                    A[2*i+1] = w * (y * P[2] - P[1])
                
                # Solve using SVD
                _, _, V = np.linalg.svd(A)
                point_3d_h = V[-1]
                point_3d = point_3d_h[:3] / point_3d_h[3]
                left_hand_3d[kp_idx] = point_3d

            else:
                left_hand_3d[kp_idx] = [np.nan, np.nan, np.nan]
        
        # Process right hand keypoints (similar approach)
        if len(right_hand_2d) >= 2:
            points_2d = []
            cam_matrices = []
            weights = []
            
            for cam_name, hand_data in right_hand_2d.items():
                if hand_data['scores'][kp_idx] > 0.3:
                    kp = hand_data['keypoints'][kp_idx]
                    
                    # Undistort the point
                    K = camera_intrinsics[cam_name].reshape(3, 3)
                    dist = distortion_coeffs[cam_name]
                    kp_undistorted = cv2.undistortPoints(np.array([[kp]]), K, dist, None, K).reshape(2)
                    
                    points_2d.append(kp_undistorted)
                    cam_matrices.append(camera_matrices[cam_name])
                    weights.append(hand_data['scores'][kp_idx])
            
            if len(points_2d) >= 2:
                # Perform weighted triangulation
                points_2d = np.array(points_2d)
                cam_matrices = np.array(cam_matrices)
                weights = np.array(weights)
                
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Create weighted system of equations A*X = 0
                A = np.zeros((2 * len(points_2d), 4))
                for i, (point, P, w) in enumerate(zip(points_2d, cam_matrices, weights)):
                    x, y = point
                    A[2*i] = w * (x * P[2] - P[0])
                    A[2*i+1] = w * (y * P[2] - P[1])
                
                # Solve using SVD
                _, _, V = np.linalg.svd(A)
                point_3d_h = V[-1]
                point_3d = point_3d_h[:3] / point_3d_h[3]
                right_hand_3d[kp_idx] = point_3d

            else:
                right_hand_3d[kp_idx] = [np.nan, np.nan, np.nan]
    
    # # Enforce wrist positions from body model if they exist
    # # Left wrist (index 9 in body model, index 0 in hand model)
    # left_hand_3d[0] = poses_3d_body[frame_idx][9]
    
    # # Right wrist (index 10 in body model, index 0 in hand model)
    # left_hand_3d[0] = poses_3d_body[frame_idx][10]
    
    # Combine left and right hand
    hand_3d = np.concatenate([left_hand_3d, right_hand_3d])
    if np.isnan(hand_3d).any():
        print(f"Warning: NaN values found in triangulated hand keypoints for frame {frame_idx}.")
        # Replace NaN with zeros to avoid issues
        hand_3d = np.nan_to_num(hand_3d, nan=0.0)
    return hand_3d

def identify_hand_ids(hand_poses_2d, video_paths, camera_intrinsics, camera_extrinsics, distortion_coeffs, poses_3d_body, matching_threshold=50, num_frames=None, visualize=False, pose_output_dir=None, multi_person=False, person_id=None):
    """
    Determine which object ID corresponds to which hand (left/right) by comparing tracked hands
    with wholebody pose estimation or canonical hand projections.
    
    Args:
        hand_poses_2d: Dictionary mapping camera names to 2D hand detections per frame
        video_paths: Dictionary mapping camera names to video file paths
        camera_intrinsics: Dictionary of camera intrinsic matrices
        camera_extrinsics: Dictionary of camera extrinsic parameters
        distortion_coeffs: Dictionary of camera distortion coefficients
        poses_3d_body: 3D body poses for each frame (dict for multi-person, array for single-person)
        matching_threshold: Maximum distance for a match to be considered valid
        num_frames: Maximum number of frames to process (optional)
        visualize: Whether to visualize the matching (optional)
        pose_output_dir: Directory containing precomputed wholebody JSON files (optional)
        multi_person: Whether using multi-person mode
        person_id: Person ID for multi-person mode
    
    Returns:
        Dictionary mapping camera names to dictionaries mapping object IDs to hand sides (0=left, 1=right)
    """    
    # Get the appropriate body poses for processing
    if multi_person:
        if person_id is None:
            raise ValueError("person_id must be provided in multi-person mode")
        if person_id not in poses_3d_body:
            raise ValueError(f"Person ID {person_id} not found in poses_3d_body")
        current_body_poses = poses_3d_body[person_id]
    else:
        current_body_poses = poses_3d_body
    
    # Initialize variables to store voting results per camera
    results = {}  # {cam_name: {obj_id: hand_side}}
    
    # Determine the number of frames to process
    if num_frames is None:
        # Find the maximum number of frames in hand poses
        max_frames = 0
        for cam_name, cam_poses in hand_poses_2d.items():
            max_frames = max(max_frames, len(cam_poses))
        # Also consider body poses length
        max_frames = min(max_frames, len(current_body_poses))
        num_frames = max_frames
    
    print(f"Processing {num_frames} frames for hand ID matching...")
    
    # Process each camera
    for cam_name, cam_poses in hand_poses_2d.items():
        # Initialize votes for this camera
        hand_id_votes = {}  # {obj_id: {0: left_votes, 1: right_votes}}
        
        if cam_name not in video_paths:
            print(f"Warning: No video path found for camera {cam_name}")
            continue

        # Get video path
        video_path = video_paths[cam_name]
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
        
        # Get camera parameters
        cam_intrinsics = camera_intrinsics[cam_name].reshape(3, 3)
        cam_rotation = camera_extrinsics[cam_name][0]
        cam_translation = camera_extrinsics[cam_name][1]
        cam_distortion = distortion_coeffs[cam_name]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if we have precomputed wholebody predictions
        wholebody_data = None
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if pose_output_dir:
            wholebody_path = os.path.join(pose_output_dir, f"{video_name}_wholebody.json")
            if os.path.exists(wholebody_path):
                print(f"Using precomputed wholebody predictions from {wholebody_path}")
                with open(wholebody_path, 'r') as f:
                    wholebody_data = json.load(f)
            else:
                print(f"Warning: Precomputed wholebody data not found at {wholebody_path}")
        
        # If we don't have precomputed data, use body bounding boxes for on-the-fly estimation
        if wholebody_data is None:
            from helpers.predictors import detect_body
            bboxes_body = detect_body(video_path)
          # Process frames
        for frame_idx in range(min(num_frames, len(cam_poses))):
            # Skip frames with no hand detections
            if not cam_poses[frame_idx]:
                continue
                
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue
                
            # Get wholebody data for this frame
            body_keypoints = None
            body_scores = None
            left_hand_keypoints = None
            left_hand_scores = None 
            right_hand_keypoints = None
            right_hand_scores = None
            
            if wholebody_data:
                # Use precomputed wholebody predictions
                if frame_idx < len(wholebody_data['body_instances']):
                    # Extract body keypoints
                    body_instances = wholebody_data['body_instances'][frame_idx]['instances']
                    if body_instances and len(body_instances) > 0:
                        body_keypoints = np.array(body_instances[0]['keypoints'])
                        body_scores = np.array(body_instances[0]['keypoint_scores'])
                    
                    # Extract left hand keypoints
                    left_hand_instances = wholebody_data['left_hand_instances'][frame_idx]['instances']
                    if left_hand_instances and len(left_hand_instances) > 0:
                        left_hand_keypoints = np.array(left_hand_instances[0]['keypoints'])
                        left_hand_scores = np.array(left_hand_instances[0]['keypoint_scores'])
                    
                    # Extract right hand keypoints
                    right_hand_instances = wholebody_data['right_hand_instances'][frame_idx]['instances']
                    if right_hand_instances and len(right_hand_instances) > 0:
                        right_hand_keypoints = np.array(right_hand_instances[0]['keypoints'])
                        right_hand_scores = np.array(right_hand_instances[0]['keypoint_scores'])
            else:
                # Estimate wholebody pose on the fly
                bbox = bboxes_body[frame_idx]
                if len(bbox) == 0:
                    continue
                
                from helpers.predictors import estimate_pose    
                body_instances, left_hand_instances, right_hand_instances = estimate_pose(frame, bbox, pose_type='wholebody', show=False)
                if body_instances is None:
                    continue
                    
                # Extract body keypoints
                body_keypoints = body_instances.keypoints  # First 17 keypoints are body keypoints
                body_scores = body_instances.keypoint_scores
                
                # Extract left and right hand keypoints from wholebody pose
                if left_hand_instances is not None:
                    left_hand_keypoints = left_hand_instances.keypoints.squeeze()
                    left_hand_scores = left_hand_instances.keypoint_scores.squeeze()
                
                if right_hand_instances is not None:
                    right_hand_keypoints = right_hand_instances.keypoints.squeeze()
                    right_hand_scores = right_hand_instances.keypoint_scores.squeeze()
            
            # Check if we have valid hand detections in wholebody
            has_left_hand = left_hand_keypoints is not None and np.mean(left_hand_scores) > 0.3
            has_right_hand = right_hand_keypoints is not None and np.mean(right_hand_scores) > 0.3

            if left_hand_scores.shape[0] != 21:
                if left_hand_scores.shape[1] == 21:
                    left_hand_scores = left_hand_scores[0]
                else:
                    has_left_hand = False
            if right_hand_scores.shape[0] != 21:
                if right_hand_scores.shape[1] == 21:
                    right_hand_scores = right_hand_scores[0]
                else:
                    has_right_hand = False
            if has_left_hand:
                if left_hand_keypoints.shape[0] != 21:
                    if left_hand_keypoints.shape[1] == 21:
                        left_hand_keypoints = left_hand_keypoints[0]
                    else:
                        has_left_hand = False

            if has_right_hand:
                if right_hand_keypoints.shape[0] != 21:
                    if right_hand_keypoints.shape[1] == 21:
                        right_hand_keypoints = right_hand_keypoints[0]
                    else:
                        has_right_hand = False
            
            # If we don't have valid hand detections, use canonical hands
            if not has_left_hand:
                # Project canonical hands
                if frame_idx < len(current_body_poses):
                    # Create canonical hands based on 3D body pose
                    if not has_left_hand:
                        from helpers.definitions import CANONICAL_HAND_POSE_3D
                        left_canonical = orient_canonical_hand(CANONICAL_HAND_POSE_3D, current_body_poses[frame_idx], side='left')
                        left_hand_keypoints = project_points(left_canonical, cam_intrinsics, cam_rotation, cam_translation, cam_distortion).squeeze()
                        has_left_hand = True
                    print("Using canonical left hand")
            if not has_right_hand:
                # Project canonical hands
                if frame_idx < len(current_body_poses):
                    # Create canonical hands based on 3D body pose
                    if not has_right_hand:
                        from helpers.definitions import CANONICAL_HAND_POSE_3D
                        right_canonical = orient_canonical_hand(CANONICAL_HAND_POSE_3D, current_body_poses[frame_idx], side='right')
                        right_hand_keypoints = project_points(right_canonical, cam_intrinsics, cam_rotation, cam_translation, cam_distortion).squeeze()
                        has_right_hand = True
                    print("Using canonical right hand")
            
            # Compare tracked hands with wholebody hands
            for obj_id, tracked_hand in cam_poses[frame_idx].items():
                if tracked_hand is None:
                    continue
                    
                tracked_keypoints = tracked_hand.keypoints.squeeze()
                tracked_scores = tracked_hand.keypoint_scores.squeeze()
                
                # Check if this is a valid hand (sufficient keypoints with good confidence)
                if np.mean(tracked_scores) < 0.2:
                    continue
                    
                # Initialize votes for this object ID if not already done
                if obj_id not in hand_id_votes:
                    hand_id_votes[obj_id] = {0: 0, 1: 0}  # {0: left_votes, 1: right_votes}
                
                # Calculate distance to left and right hand
                left_distance = np.inf
                right_distance = np.inf
                
                if has_left_hand:
                    # Calculate distance to left hand
                    valid_mask = tracked_scores > 0.3
                    if np.any(valid_mask):
                        distances = np.linalg.norm(
                            tracked_keypoints[valid_mask, :] - left_hand_keypoints[valid_mask, :],
                            axis=1
                        )
                        left_distance = np.mean(distances)
                
                if has_right_hand:
                    # Calculate distance to right hand
                    valid_mask = tracked_scores > 0.3
                    if np.any(valid_mask):
                        distances = np.linalg.norm(
                            tracked_keypoints[valid_mask, :] - right_hand_keypoints[valid_mask, :],
                            axis=1
                        )
                        right_distance = np.mean(distances)
                
                # Vote for the closest hand
                if left_distance < right_distance:
                    if left_distance < matching_threshold:
                        hand_id_votes[obj_id][0] += 1  # Vote for left hand
                elif right_distance < matching_threshold:
                    hand_id_votes[obj_id][1] += 1  # Vote for right hand
                    
                # Visualize if requested
                if visualize:
                    from helpers.pose_visualization import draw_pose
                    vis_frame = frame.copy()
                    
                    # Draw tracked hand
                    draw_pose(vis_frame, tracked_hand, pose_type='hand', IS_HAND_THRESHOLD=0.1)
                    
                    # Draw wholebody hands
                    if has_left_hand:
                        for i, (pt, score) in enumerate(zip(left_hand_keypoints, left_hand_scores if left_hand_scores is not None else np.ones(left_hand_keypoints.shape[0]))):
                            if score > 0.3 or left_hand_scores is None:
                                x, y = map(int, pt)
                                cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)  # Left hand in red
                                
                    if has_right_hand:
                        for i, (pt, score) in enumerate(zip(right_hand_keypoints, right_hand_scores if right_hand_scores is not None else np.ones(right_hand_keypoints.shape[0]))):
                            if score > 0.3 or right_hand_scores is None:
                                x, y = map(int, pt)
                                cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)  # Right hand in green
                    
                    # Add text showing the object ID and current vote counts
                    votes = hand_id_votes[obj_id]
                    cv2.putText(vis_frame, f"Camera: {cam_name}, ID: {obj_id}, Left: {votes[0]}, Right: {votes[1]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display the frame
                    cv2.namedWindow("Hand ID Matching", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Hand ID Matching", 1280, 720)
                    cv2.imshow("Hand ID Matching", vis_frame)
                    key = cv2.waitKey(100)  # Short delay to see visualization
                    if key == 27:  # ESC to exit
                        visualize = False
                        cv2.destroyAllWindows()
        
        # Release video
        cap.release()
        
        # Determine final hand ID mapping for this camera based on voting
        camera_mapping = {}
        for obj_id, votes in hand_id_votes.items():
            if votes[0] + votes[1] < num_frames // 2:
                continue
            # Assign hand with more votes
            camera_mapping[obj_id] = 0 if votes[0] > votes[1] else 1
            print(f"Camera {cam_name}, Object ID {obj_id}: {votes[0]} votes for left hand, {votes[1]} votes for right hand -> Assigned to {'left' if camera_mapping[obj_id] == 0 else 'right'} hand")
        
        # Store results for this camera
        results[cam_name] = camera_mapping
    
    if visualize:
        cv2.destroyAllWindows()
        
    return results
