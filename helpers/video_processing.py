import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from mmpose.structures import split_instances
from helpers.predictors import estimate_pose, detect_body

# This module provides functions for processing videos to extract 2D poses.
# It includes functions for:
# - process_single_video: Process a single video to extract body poses
# - process_single_video_wholebody: Process a single video to extract wholebody poses (body + hands + face)
# - prepare_video_frames: Prepare video frames for further processing

def process_single_video(video_name, data_dir, pose_output_dir):
    """Process a single video to extract 2D poses"""
    video_path = os.path.join(data_dir, f"{video_name}.MP4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False

    print(f"Processing {video_name}...")

    # Process frames and estimate poses
    pred_instances_body = []
    cap = cv2.VideoCapture(video_path)

    if os.path.exists(os.path.join(pose_output_dir, f"{video_name}_body.json")):
        print(f"Skipping {video_name} - already processed")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detect body poses
    bboxes_body = detect_body(video_path)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Estimate body pose
        bbox = np.array(bboxes_body[frame_idx])
        if len(bbox.shape) == 1:
            bbox = bbox.reshape((1, 4))
            
        pred_instances = estimate_pose(frame, bbox, pose_type='body', show=False)
        
        # Convert instances to JSON-serializable format
        instances_json = []
        if pred_instances is not None:
            for instance in split_instances(pred_instances):
                instance_json = {
                    'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                    'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                }
                if 'bbox' in instance:
                    instance_json['bbox'] = np.array(instance['bbox']).astype(float).tolist()
                    if 'bbox_score' in instance:
                        instance_json['bbox_score'] = float(instance['bbox_score'])
                instances_json.append(instance_json)

        pred_instances_body.append(dict(
            frame_id=frame_idx,
            instances=instances_json
        ))
        
        frame_idx += 1
        print(f"\rProcessing frame {frame_idx}", end='')

    print(". Done!")
        
    cap.release()
    
    # Save predictions
    save_path = os.path.join(pose_output_dir, f"{video_name}_body.json")
    with open(save_path, 'w') as f:
        json.dump({
            'instance_info': pred_instances_body
        }, f, indent=2)
        
    return frame_idx

def process_single_video_wholebody(video_name, data_dir, pose_output_dir, show=False, vis_output_dir=None):
    """Process a single video to extract wholebody poses (body + hands + face)
    
    Args:
        video_name: Name of the video file (without extension)
        data_dir: Directory containing the video file
        pose_output_dir: Directory to save pose results
        show: Whether to visualize the poses during processing
        vis_output_dir: Directory to save visualization video (if None, defaults to 'vis_dir')
        
    Returns:
        Number of processed frames or False if video not found
    """
    video_path = os.path.join(data_dir, f"{video_name}.MP4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False

    print(f"Processing {video_name} with wholebody model...")

    # Setup visualization output directory if needed
    if show and vis_output_dir is None:
        vis_output_dir = 'vis_dir'
    
    if show and not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)

    # Check if already processed
    if os.path.exists(os.path.join(pose_output_dir, f"{video_name}_wholebody.json")):
        print(f"Skipping {video_name} - already processed")
        cap = cv2.VideoCapture(video_path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detect body poses
    bboxes_body = detect_body(video_path)
      # Process frames and estimate poses
    body_instances_list = []
    left_hand_instances_list = []
    right_hand_instances_list = []
    
    # Setup video writer for visualization if needed
    video_writer = None
    if show:
        output_video_path = os.path.join(vis_output_dir, f"{video_name}_wholebody.mp4")
        cap_temp = cv2.VideoCapture(video_path)
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        cap_temp.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )
        print(f"Visualizations will be saved to {output_video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Estimate wholebody pose
        bbox = np.array(bboxes_body[frame_idx])
        if len(bbox.shape) == 1:
            bbox = bbox.reshape((1, 4))
            
        body_instances, left_hand_instances, right_hand_instances = estimate_pose(
            frame, bbox, pose_type='wholebody'
        )
        
        # Convert body instances to JSON-serializable format
        body_instances_json = []
        if body_instances is not None:
            for instance in split_instances(body_instances):
                instance_json = {
                    'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                    'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                }
                if 'bbox' in instance:
                    instance_json['bbox'] = np.array(instance['bbox']).astype(float).tolist()
                    if 'bbox_score' in instance:
                        instance_json['bbox_score'] = float(instance['bbox_score'])
                body_instances_json.append(instance_json)

        # Convert left hand instances to JSON-serializable format
        left_hand_instances_json = []
        if left_hand_instances is not None:
            for instance in split_instances(left_hand_instances):
                instance_json = {
                    'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                    'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                }
                left_hand_instances_json.append(instance_json)
                
        # Convert right hand instances to JSON-serializable format
        right_hand_instances_json = []
        if right_hand_instances is not None:
            for instance in split_instances(right_hand_instances):
                instance_json = {
                    'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                    'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                }
                right_hand_instances_json.append(instance_json)
                
        # Add to lists
        body_instances_list.append(dict(
            frame_id=frame_idx,
            instances=body_instances_json
        ))
        
        left_hand_instances_list.append(dict(
            frame_id=frame_idx,
            instances=left_hand_instances_json
        ))
        right_hand_instances_list.append(dict(
            frame_id=frame_idx,
            instances=right_hand_instances_json
        ))
        
        # Visualize if needed
        if show and video_writer is not None:
            # Create a frame for visualization
            vis_frame = frame.copy()
            
            # Draw body keypoints
            if body_instances is not None:
                for i, kpts in enumerate(body_instances.keypoints):
                    scores = body_instances.keypoint_scores[i]
                    for j, (kpt, score) in enumerate(zip(kpts, scores)):
                        if score > 0.3:  # Confidence threshold
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)  # Green for body
                    
                    # Draw lines between connected body keypoints using WHOLEBODY_KEYPOINT_PAIRS
                    from helpers.definitions import WHOLEBODY_KEYPOINT_PAIRS
                    for pair in WHOLEBODY_KEYPOINT_PAIRS:
                        idx1, idx2 = pair
                        if scores[idx1] > 0.3 and scores[idx2] > 0.3:
                            pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                            pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                            cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)
            
            # Draw left hand keypoints
            if left_hand_instances is not None:
                for i, kpts in enumerate(left_hand_instances.keypoints):
                    scores = left_hand_instances.keypoint_scores[i]
                    for j, (kpt, score) in enumerate(zip(kpts, scores)):
                        if score > 0.3:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)  # Blue for left hand
                    
                    # Draw connections between hand keypoints
                    from helpers.definitions import KEYPOINT_PAIRS_HANDS
                    for pair in KEYPOINT_PAIRS_HANDS:
                        idx1, idx2 = pair
                        if scores[idx1] > 0.3 and scores[idx2] > 0.3:
                            pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                            pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                            cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
            
            # Draw right hand keypoints
            if right_hand_instances is not None:
                for i, kpts in enumerate(right_hand_instances.keypoints):
                    scores = right_hand_instances.keypoint_scores[i]
                    for j, (kpt, score) in enumerate(zip(kpts, scores)):
                        if score > 0.3:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)  # Red for right hand
                    
                    # Draw connections between hand keypoints
                    for pair in KEYPOINT_PAIRS_HANDS:
                        idx1, idx2 = pair
                        if scores[idx1] > 0.3 and scores[idx2] > 0.3:
                            pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                            pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                            cv2.line(vis_frame, pt1, pt2, (0, 0, 255), 1)
            
            # Write the frame to the video
            video_writer.write(vis_frame)
            
            # Display the frame
            if show:
                cv2.namedWindow('Wholebody Pose', cv2.WINDOW_NORMAL)
                cv2.imshow('Wholebody Pose', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
        print(f"\rProcessing frame {frame_idx}", end='')

    print(". Done!")
        
    # Release resources
    cap.release()
    if show and video_writer is not None:
        video_writer.release()
        cv2.destroyAllWindows()
    
    # Save predictions
    save_path = os.path.join(pose_output_dir, f"{video_name}_wholebody.json")
    with open(save_path, 'w') as f:
        json.dump({
            'body_instances': body_instances_list,
            'left_hand_instances': left_hand_instances_list,
            'right_hand_instances': right_hand_instances_list
        }, f, indent=2)
        
    return frame_idx

def prepare_video_frames(video_path, resample_to=None):
    """Create frames directory if video file exists"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(os.path.dirname(video_path), video_name)
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resample_to is not None:
                frame = cv2.resize(frame, resample_to)
            cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), frame)
            frame_idx += 1
            
        cap.release()
        
    return frames_dir
