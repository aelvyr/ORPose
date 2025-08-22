import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers.camera import project_points, project_points_safe

def draw_pose(frame, pred_instance, pose_type='body', IS_HAND_THRESHOLD=0.3, POSE_KPT_THRESHOLD=0.3, 
             POSE_VIS_LINE_WIDTH=2, POSE_VIS_RADIUS=5, hide_legs=False, return_frame=False):
    """Draw pose keypoints and connections on frame"""
    height, width = frame.shape[:2]
    
    # Extract keypoints and scores
    keypoints = pred_instance.keypoints.squeeze()
    scores = pred_instance.keypoint_scores.squeeze()
    
    # Safety check for NaN values
    if keypoints is None or scores is None or np.isnan(keypoints).any():
        return
    
    if pose_type == 'hand':
        # Hand keypoint colors
        colors = [(255, 0, 0),   # thumb - red
                 (0, 255, 0),    # index - green
                 (0, 0, 255),    # middle - blue
                 (255, 255, 0),  # ring - yellow
                 (255, 0, 255)]  # pinky - magenta
        
        # Hand connections (finger joints)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]        # Draw keypoints and connections
        keypoints = pred_instance.keypoints.squeeze()
        scores = pred_instance.keypoint_scores.squeeze()
        
        # Draw connections first
        for idx, conn in enumerate(connections):
            if scores[conn[0]] > IS_HAND_THRESHOLD and scores[conn[1]] > IS_HAND_THRESHOLD:
                pt1 = tuple(map(int, keypoints[conn[0]]))
                pt2 = tuple(map(int, keypoints[conn[1]]))
                finger_idx = idx // 4  # Determine which finger
                
                # Check if points are within image bounds
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    try:
                        cv2.line(frame, pt1, pt2, colors[finger_idx], POSE_VIS_LINE_WIDTH)
                    except:
                        print(f"Error drawing line for connection {conn}: {pt1} to {pt2}")
                        continue

        # Draw keypoints
        for i, (point, score) in enumerate(zip(keypoints, scores)):
            if score > IS_HAND_THRESHOLD:  # Only draw high-confidence keypoints
                x, y = map(int, point)
                # Check if point is within image bounds
                if 0 <= x < width and 0 <= y < height:
                    finger_idx = (i - 1) // 4 if i > 0 else 0  # Determine which finger the point belongs to
                    cv2.circle(frame, (x, y), POSE_VIS_RADIUS, colors[finger_idx], -1)

    elif pose_type == 'body':
        # Body keypoint colors - using consistent colors for different body parts
        colors = {
            'torso': (255, 0, 0),      # red
            'head': (0, 255, 0),       # green
            'arms': (0, 0, 255),       # blue
            'legs': (255, 255, 0)      # yellow
        }        # Body connections based on MMPose skeleton
        connections = [
            # Torso
            ([5, 6], colors['torso']),      # shoulders
            ([5, 11], colors['torso']),     # left shoulder to hip
            ([6, 12], colors['torso']),     # right shoulder to hip
            ([11, 12], colors['torso']),    # hips
            
            # Head
            ([5, 3], colors['head']),       # left shoulder to neck
            ([6, 3], colors['head']),       # right shoulder to neck
            ([3, 1], colors['head']),       # neck to head
            
            # Arms
            ([5, 7], colors['arms']),       # left arm
            ([7, 9], colors['arms']),       # left forearm
            ([6, 8], colors['arms']),       # right arm
            ([8, 10], colors['arms']),      # right forearm
        ]
        
        # Add leg connections only if not hidden
        if not hide_legs:
            leg_connections = [
                # Legs
                ([11, 13], colors['legs']),     # left thigh
                ([13, 15], colors['legs']),     # left calf
                ([12, 14], colors['legs']),     # right thigh
                ([14, 16], colors['legs']),     # right calf
            ]
            connections.extend(leg_connections)
            
        # Define leg keypoint indices
        leg_keypoints = [11, 12, 13, 14, 15, 16] if hide_legs else []# Draw keypoints and connections
        keypoints = pred_instance.keypoints.squeeze()
        scores = pred_instance.keypoint_scores.squeeze()        # Draw connections first
        for (conn, color) in connections:
            if scores[conn[0]] > POSE_KPT_THRESHOLD and scores[conn[1]] > POSE_KPT_THRESHOLD:
                pt1 = tuple(map(int, keypoints[conn[0]]))
                pt2 = tuple(map(int, keypoints[conn[1]]))
                
                # Check if points are within image bounds
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    try:
                        cv2.line(frame, pt1, pt2, color, POSE_VIS_LINE_WIDTH)
                    except Exception as e:
                        print(f"Error drawing body line for connection {conn}: {str(e)}")
                        continue

        # Draw keypoints
        for i, (point, score) in enumerate(zip(keypoints, scores)):
            # Skip leg keypoints if hide_legs is True
            if hide_legs and i in leg_keypoints:
                continue
                
            if score > POSE_KPT_THRESHOLD:  # Only draw high-confidence keypoints
                x, y = map(int, point)
                # Check if point is within image bounds
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), POSE_VIS_RADIUS, (255, 255, 255), -1)

    if return_frame:
        return frame

def visualize_poses(frame, body_pose_3d, initial_hands_3d, hand_poses_2d, 
                    camera_intrinsics, rotation, translation, distortion, 
                    frame_idx, cam_name, draw_reprojected_only=True, 
                    draw_body=True, draw_hands=True, hide_legs=False,
                    hand_id_mapping=None):
    """
    Visualize body poses, 3D hand poses, and 2D hand detections on a frame.
    
    Args:
        frame: Input image/frame
        body_pose_3d: 3D body pose (num_keypoints, 3)
        initial_hands_3d: 3D hand poses (num_hands * num_keypoints_hands, 3)
        hand_poses_2d: Dictionary of 2D hand poses for current frame
        camera_intrinsics: Camera intrinsic matrix (3x3)
        rotation: Camera rotation matrix
        translation: Camera translation vector
        distortion: Distortion coefficients
        frame_idx: Current frame index
        cam_name: Camera name
        draw_reprojected_only: Whether to draw only reprojected poses (not detected ones)
        draw_body: Whether to draw body poses
        draw_hands: Whether to draw hand poses
        hide_legs: Whether to hide leg keypoints
        hand_id_mapping: Optional mapping between hand object IDs and hand sides (left/right)
    """
    # Create copy of frame for drawing
    vis_frame = frame.copy()
    
    # Project and draw 3D body pose
    if body_pose_3d is not None and draw_body:
        body_2d = project_points_safe(body_pose_3d, camera_intrinsics, 
                               rotation, translation, distortion)
        draw_pose(vis_frame, type('', (), {
            'keypoints': body_2d.reshape(1, -1, 2),
            'keypoint_scores': np.ones((1, len(body_2d)))
        }), pose_type='body', hide_legs=hide_legs)
    
    # Project and draw 3D hand poses
    if initial_hands_3d is not None and draw_hands:
        # Left hand (first 21 points)
        left_hand_2d = project_points_safe(initial_hands_3d[:21], camera_intrinsics,
                                    rotation, translation, distortion)
        # Create dummy instance with projected points
        left_hand_instance = type('', (), {
            'keypoints': np.array(left_hand_2d).reshape(1, -1, 2),
            'keypoint_scores': np.ones((1, len(left_hand_2d)))
        })
        draw_pose(vis_frame, left_hand_instance, pose_type='hand')
        
        # Right hand (last 21 points)
        right_hand_2d = project_points_safe(initial_hands_3d[21:], camera_intrinsics,
                                     rotation, translation, distortion)
        right_hand_instance = type('', (), {
            'keypoints': np.array(right_hand_2d).reshape(1, -1, 2),
            'keypoint_scores': np.ones((1, len(right_hand_2d)))
        })
        draw_pose(vis_frame, right_hand_instance, pose_type='hand')
    
    # Draw 2D detected hand poses if not drawing reprojected poses only
    if not draw_reprojected_only and draw_hands:
        if cam_name in hand_poses_2d and len(hand_poses_2d[cam_name]) > frame_idx:
            poses_dict = hand_poses_2d[cam_name][frame_idx]
            for hand_id, pose_2d in poses_dict.items():
                if pose_2d is not None:
                    draw_pose(vis_frame, pose_2d, pose_type='hand')
    
    # Display viewpoint counts if hand_id_mapping is provided
    if hand_id_mapping is not None and draw_hands:
        # Count how many viewpoints see each hand
        left_hand_count = 0
        right_hand_count = 0
        
        for cam in hand_poses_2d.keys():
            if len(hand_poses_2d[cam]) > frame_idx and hand_poses_2d[cam][frame_idx]:
                for obj_id, pose in hand_poses_2d[cam][frame_idx].items():
                    # Check if this camera and object ID has a mapping
                    if cam in hand_id_mapping and obj_id in hand_id_mapping[cam]:
                        hand_side = hand_id_mapping[cam][obj_id]
                        if hand_side == 0:  # Left hand
                            left_hand_count += 1
                        elif hand_side == 1:  # Right hand
                            right_hand_count += 1
        
        # Display viewpoint counts in bottom corners
        height, width = vis_frame.shape[:2]
        
        # Left hand count (bottom left)
        cv2.putText(vis_frame, f"Left Hand: {left_hand_count} views", 
                   (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 255), 2)
        
        # Right hand count (bottom right)
        right_text = f"Right Hand: {right_hand_count} views"
        right_text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        cv2.putText(vis_frame, right_text, 
                   (width - right_text_size[0] - 10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
    
    return vis_frame

def display_frame(frame):
    """Helper function to display frame with proper window settings"""
    cv2.namedWindow('Poses', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Poses', 1920, 1080)
    cv2.imshow('Poses', frame)
    if cv2.waitKey(1) == 113:
        cv2.destroyAllWindows()
        return True

def process_and_visualize_poses(video_path, tracks, output_dir, poses_3d_body=None, camera_matrices=None, 
                             camera_intrinsics=None, camera_extrinsics=None, distortion_coeffs=None,
                             hide_legs=False, create_label_poses=False, multi_person=False, person_id=None):
    from helpers.predictors import estimate_pose
    """Process and visualize body and hand poses"""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_name}_poses.mp4")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    hand_poses_2d = []
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Project 3D poses to 2D for this camera and frame
        if poses_3d_body is not None:
            # Handle multi-person mode
            if multi_person:
                if person_id is None:
                    raise ValueError("person_id must be provided in multi-person mode")
                if person_id in poses_3d_body and frame_idx < len(poses_3d_body[person_id]):
                    pose_3d = poses_3d_body[person_id][frame_idx]
                else:
                    pose_3d = None
            else:
                # Single-person mode
                if frame_idx < len(poses_3d_body):
                    pose_3d = poses_3d_body[frame_idx]
                else:
                    pose_3d = None
            
            if pose_3d is not None:
                # Get camera parameters for this video
                cam_name = video_name.split('_')[0]  # Extract camera name from video name
                if cam_name in camera_matrices:                
                    cam_intrinsics = camera_intrinsics[cam_name].reshape(3,3)
                    cam_extrinsics = camera_extrinsics[cam_name]
                    cam_distortion = distortion_coeffs[cam_name]
                    
                    points_2d = project_points_safe(pose_3d, cam_intrinsics, cam_extrinsics[0], cam_extrinsics[1], cam_distortion)
                    
                    # Create a body pose instance with the projected points
                    body_pose = type('', (), {})()  # Create empty object
                    body_pose.keypoints = np.array(points_2d).reshape(-1, 2)
                    body_pose.keypoint_scores = np.ones(len(points_2d))  # Assuming all points are valid
                    
                else:
                    body_pose = None
            else:
                body_pose = None
        else:
            body_pose = None

        hand_pose_2d = {}
        # Draw bounding boxes and estimate hand poses for tracked hands
        if frame_idx in tracks:
            for hand_id, box in tracks[frame_idx].items():
                # Draw bounding box
                if len(box) == 4:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add hand ID text above the box
                    cv2.putText(frame, f'Hand {hand_id}', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                    # Estimate and draw hand pose
                    hand_pose = estimate_pose(frame, box, pose_type='hand', show=False)
                    if hand_pose is not None:
                        draw_pose(frame, hand_pose, pose_type='hand')
                        hand_pose_2d[hand_id] = hand_pose

            if create_label_poses and 1 not in hand_pose_2d.keys():
                # If no hand poses were detected, use a placeholder
                box = np.array([0, 0, 0, 0])
                hand_pose = estimate_pose(frame, box, pose_type='hand', show=False)
                if hand_pose is not None:
                    draw_pose(frame, hand_pose, pose_type='hand')
                    hand_pose_2d[1] = hand_pose

            if create_label_poses and 0 not in hand_pose_2d.keys():
                # If no hand poses were detected, use a placeholder
                box = np.array([0, 0, 0, 0])
                hand_pose = estimate_pose(frame, box, pose_type='hand', show=False)
                if hand_pose is not None:
                    draw_pose(frame, hand_pose, pose_type='hand')
                    hand_pose_2d[0] = hand_pose

        hand_poses_2d.append(hand_pose_2d)
                    
        # Draw body keypoints
        if body_pose is not None:
            draw_pose(frame, body_pose, pose_type='body', hide_legs=hide_legs)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return hand_poses_2d

def create_reprojection_videos(poses_3d_body, hand_poses_3d_dict, camera_intrinsics, 
                     camera_extrinsics, distortion_coeffs, data_dir, output_dir, 
                     camera_angles=['gopro10', 'gopro5'], hide_legs=False,
                     compare_viewpoints=False, hand_id_mapping=None, multi_person=False, person_id=None):
    """
    Create videos with 3D hand poses reprojected into 2D camera views from specified angles.
    
    Args:
        poses_3d_body: Array of 3D body poses for each frame (or dict for multi-person)
        hand_poses_3d_dict: Dictionary mapping method names to arrays of 3D hand poses
        camera_intrinsics: Dictionary of camera intrinsic matrices
        camera_extrinsics: Dictionary of camera extrinsic parameters
        distortion_coeffs: Dictionary of camera distortion coefficients
        data_dir: Directory containing input videos
        output_dir: Directory to save output videos
        camera_angles: List of camera names to use for video creation
        hide_legs: Whether to hide leg keypoints in the visualization
        compare_viewpoints: If True, create videos comparing the same method across different viewpoints
                           If False (default), compare different methods for the same viewpoint
        hand_id_mapping: Optional mapping between hand object IDs and hand sides for viewpoint counting
        multi_person: Whether using multi-person mode
        person_id: Person ID for multi-person mode
    """
    # Check if specified camera angles exist
    available_cameras = []
    for camera in camera_angles:
        video_path = os.path.join(data_dir, f"{camera}_synced_cut.MP4")
        if os.path.exists(video_path):
            available_cameras.append(camera)
        else:
            print(f"Warning: Video for camera {camera} not found at {video_path}")
    
    if not available_cameras:
        print("No available cameras found for reprojection videos")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if compare_viewpoints:
        # Create one video per method, comparing different viewpoints
        for method_name, hand_poses_3d in hand_poses_3d_dict.items():
            if len(hand_poses_3d) == 0:
                print(f"Warning: No hand poses for method {method_name}")
                continue
                
            # Create a combined video showing all viewpoints for this method
            output_path = os.path.join(output_dir, f'{method_name}_viewpoint_comparison.mp4')
            
            # Get dimensions of the first available camera
            sample_camera = available_cameras[0]
            video_path = os.path.join(data_dir, f"{sample_camera}_synced_cut.MP4")
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = min(len(poses_3d_body), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            
            # Initialize video writer for combined viewpoints
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            combined_writer = cv2.VideoWriter(
                output_path, fourcc, fps, 
                (frame_width * len(available_cameras), frame_height)
            )
            
            # Dictionary to hold frame buffers for each camera
            camera_frames = {camera: [] for camera in available_cameras}
            
            # Process each camera's video
            for camera in available_cameras:
                video_path = os.path.join(data_dir, f"{camera}_synced_cut.MP4")
                cap = cv2.VideoCapture(video_path)
                
                # Get camera parameters
                cam_intrinsics = camera_intrinsics[camera].reshape(3, 3)
                cam_extrinsics = camera_extrinsics[camera]
                cam_rotation = cam_extrinsics[0]
                cam_translation = cam_extrinsics[1]
                cam_distortion = distortion_coeffs[camera]
                
                # Process each frame
                frame_idx = 0
                while frame_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        # If we run out of frames, add blank frames
                        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, f"{camera} (No data)", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        camera_frames[camera].append(blank_frame)
                    else:
                        try:
                            # Create a frame for this camera
                            method_frame = frame.copy()
                            
                            # Add camera name as text label
                            cv2.putText(method_frame, f"{camera}", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            # Add method name too
                            cv2.putText(method_frame, f"{method_name}", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            
                            # Project 3D body pose to 2D
                            body_pose_frame = None
                            if multi_person:
                                if person_id is not None and person_id in poses_3d_body and frame_idx < len(poses_3d_body[person_id]):
                                    body_pose_frame = poses_3d_body[person_id][frame_idx]
                            else:
                                if frame_idx < len(poses_3d_body):
                                    body_pose_frame = poses_3d_body[frame_idx]
                            
                            if body_pose_frame is not None and not np.isnan(body_pose_frame).any():
                                body_2d = project_points_safe(body_pose_frame, cam_intrinsics, 
                                                        cam_rotation, cam_translation, cam_distortion)
                                
                                # Create a dummy instance with the projected points for drawing
                                body_instance = type('', (), {
                                    'keypoints': np.array(body_2d).reshape(1, -1, 2),
                                    'keypoint_scores': np.ones((1, len(body_2d)))
                                })
                                
                                # Draw body pose
                                draw_pose(method_frame, body_instance, pose_type='body', hide_legs=hide_legs)
                            
                            # Project 3D hand pose to 2D
                            if frame_idx < len(hand_poses_3d) and hand_poses_3d[frame_idx] is not None:
                                hand_3d = hand_poses_3d[frame_idx]
                                # Check for NaN values in hand poses
                                if not np.isnan(hand_3d).any():
                                    # Visualize the poses with viewpoint counts if hand_id_mapping is provided
                                    method_frame = visualize_poses(
                                        method_frame, 
                                        None,  # Don't redraw body
                                        hand_3d, 
                                        hand_poses_2d= None,
                                        camera_intrinsics=cam_intrinsics,
                                        rotation=cam_rotation,
                                        translation=cam_translation,
                                        distortion=cam_distortion,
                                        frame_idx=frame_idx,
                                        cam_name=camera,
                                        draw_body=False,
                                        hand_id_mapping=None
                                    )
                            
                            camera_frames[camera].append(method_frame)
                        except Exception as e:
                            print(f"Error processing frame {frame_idx} for camera {camera}: {str(e)}")
                            # Use a copy of the original frame with just the camera name as a fallback
                            fallback_frame = frame.copy()
                            cv2.putText(fallback_frame, f"{camera} (Error)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            camera_frames[camera].append(fallback_frame)
                    
                    frame_idx += 1
                    print(f"\rProcessing frame {frame_idx}/{total_frames} for camera {camera} (Method: {method_name})", end='')
                
                cap.release()
                print(f"\nFinished processing {camera} video for method {method_name}")
            
            # Combine frames from all cameras and write to video
            for frame_idx in range(min(len(camera_frames[cam]) for cam in available_cameras)):
                frames_to_combine = [camera_frames[cam][frame_idx] for cam in available_cameras]
                try:
                    combined_frame = np.hstack(frames_to_combine)
                    combined_writer.write(combined_frame)
                except Exception as e:
                    print(f"Error creating combined frame at frame {frame_idx}: {str(e)}")
            
            combined_writer.release()
            
            print(f"Created viewpoint comparison video for method {method_name}")
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                print(f"  - {output_path}: {file_size:.2f} MB")
            else:
                print(f"  - Warning: {output_path} was not created successfully")
                
    else:
        # Original implementation: Compare different methods for the same viewpoint
        for camera in available_cameras:
            # Load video
            video_path = os.path.join(data_dir, f"{camera}_synced_cut.MP4")
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = min(len(poses_3d_body), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            
            # Get camera parameters
            cam_intrinsics = camera_intrinsics[camera].reshape(3, 3)
            cam_extrinsics = camera_extrinsics[camera]
            cam_rotation = cam_extrinsics[0]
            cam_translation = cam_extrinsics[1]
            cam_distortion = distortion_coeffs[camera]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Create video writer for each method
            video_writers = {}
            for method_name in hand_poses_3d_dict.keys():
                output_path = os.path.join(output_dir, f'{camera}_reprojection_{method_name}.mp4')
                video_writers[method_name] = cv2.VideoWriter(
                    output_path, fourcc, fps, (frame_width, frame_height)
                )
            
            # Create a combined video with all methods side by side
            combined_output_path = os.path.join(output_dir, f'{camera}_reprojection_comparison.mp4')
            
            # Read the first frame to get dimensions
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read first frame from {camera} video")
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Make sure all methods have valid 3D poses for the comparison video
            valid_methods = []
            for method_name, hand_poses_3d in hand_poses_3d_dict.items():
                if len(hand_poses_3d) > 0:
                    valid_methods.append(method_name)
                    
            if not valid_methods:
                print(f"Warning: No valid methods with 3D poses for {camera} video")
                continue
            
            # Initialize the video writer with the correct dimensions
            combined_writer = cv2.VideoWriter(
                combined_output_path, fourcc, fps, 
                (frame_width * len(valid_methods), frame_height)
            )
            
            # Process each frame
            frame_idx = 0
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create visualizations for each method
                method_frames = []
                for method_name in valid_methods:
                    hand_poses_3d = hand_poses_3d_dict[method_name]
                    if frame_idx < len(hand_poses_3d):
                        try:
                            # Create a copy of the frame for this method
                            method_frame = frame.copy()
                            
                            # Add method name as text label
                            cv2.putText(method_frame, method_name, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            # Project 3D body pose to 2D
                            body_pose_frame = None
                            if multi_person:
                                if person_id is not None and person_id in poses_3d_body and frame_idx < len(poses_3d_body[person_id]):
                                    body_pose_frame = poses_3d_body[person_id][frame_idx]
                            else:
                                if frame_idx < len(poses_3d_body):
                                    body_pose_frame = poses_3d_body[frame_idx]
                            
                            if body_pose_frame is not None and not np.isnan(body_pose_frame).any():
                                body_2d = project_points_safe(body_pose_frame, cam_intrinsics, 
                                                        cam_rotation, cam_translation, cam_distortion)
                                
                                # Create a dummy instance with the projected points for drawing
                                body_instance = type('', (), {
                                    'keypoints': np.array(body_2d).reshape(1, -1, 2),
                                    'keypoint_scores': np.ones((1, len(body_2d)))
                                })
                                
                                # Draw body pose
                                draw_pose(method_frame, body_instance, pose_type='body', hide_legs=hide_legs)
                            
                            # Project 3D hand pose to 2D
                            hand_3d = hand_poses_3d[frame_idx]
                            # Check for NaN values in hand poses
                            if not np.isnan(hand_3d).any():
                                # Visualize the poses with viewpoint counts if hand_id_mapping is provided
                                method_frame = visualize_poses(
                                    method_frame, 
                                    None,  # Don't redraw body
                                    hand_3d, 
                                    hand_poses_2d=None,
                                    camera_intrinsics=cam_intrinsics,
                                    rotation=cam_rotation,
                                    translation=cam_translation,
                                    distortion=cam_distortion,
                                    frame_idx=frame_idx,
                                    cam_name=camera,
                                    draw_body=False,
                                    hand_id_mapping=None
                                )
                            
                            # Write frame to method-specific video
                            video_writers[method_name].write(method_frame)
                            
                            # Add to list for combined video
                            method_frames.append(method_frame)
                        except Exception as e:
                            print(f"Error processing frame {frame_idx} for method {method_name}: {str(e)}")
                            # Use a copy of the original frame with just the method name as a fallback
                            fallback_frame = frame.copy()
                            cv2.putText(fallback_frame, f"{method_name} (Error)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            method_frames.append(fallback_frame)
                    else:
                        # If this method doesn't have data for this frame, use a blank frame
                        blank_frame = frame.copy()
                        cv2.putText(blank_frame, f"{method_name} (No data)", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        method_frames.append(blank_frame)
                
                # Create combined frame with all methods side by side
                if method_frames:
                    try:
                        # Check all frames have the same dimensions before stacking
                        heights = [f.shape[0] for f in method_frames]
                        widths = [f.shape[1] for f in method_frames]
                        channels = [f.shape[2] for f in method_frames]
                        
                        if len(set(heights)) > 1 or len(set(widths)) > 1 or len(set(channels)) > 1:
                            print(f"Warning: Inconsistent frame dimensions at frame {frame_idx}: {list(zip(heights, widths, channels))}")
                            # Resize all frames to match the expected dimensions
                            for i in range(len(method_frames)):
                                if method_frames[i].shape[:2] != (frame_height, frame_width):
                                    method_frames[i] = cv2.resize(method_frames[i], (frame_width, frame_height))
                        
                        combined_frame = np.hstack(method_frames)
                        
                        # Verify the combined frame has the expected dimensions
                        expected_width = frame_width * len(method_frames)
                        expected_height = frame_height
                        
                        if combined_frame.shape[1] != expected_width or combined_frame.shape[0] != expected_height:
                            print(f"Warning: Combined frame has unexpected dimensions: {combined_frame.shape}")
                            combined_frame = cv2.resize(combined_frame, (expected_width, expected_height))
                        
                        combined_writer.write(combined_frame)
                    except Exception as e:
                        print(f"Error creating combined frame at frame {frame_idx}: {str(e)}")
                
                # Increment frame index
                frame_idx += 1
                print(f"\rProcessing frame {frame_idx}/{total_frames} for camera {camera}", end='')
            
            print(f"\nFinished processing {camera} video")
            
            # Release resources
            cap.release()
            for writer in video_writers.values():
                writer.release()
            combined_writer.release()
            
            # Verify the output files exist and have appropriate size
            for method_name in valid_methods:
                output_path = os.path.join(output_dir, f'{camera}_reprojection_{method_name}.mp4')
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                    print(f"  - {output_path}: {file_size:.2f} MB")
                else:
                    print(f"  - Warning: {output_path} was not created successfully")
                    
            if os.path.exists(combined_output_path):
                file_size = os.path.getsize(combined_output_path) / (1024 * 1024)  # Size in MB
                print(f"  - {combined_output_path}: {file_size:.2f} MB")
            else:
                print(f"  - Warning: {combined_output_path} was not created successfully")
    
    print("All reprojection videos created successfully")
