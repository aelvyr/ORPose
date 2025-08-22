import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from mmpose.structures import split_instances
from helpers.predictors import estimate_pose, detect_body
from helpers.hand_detection import adjust_tracks

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

def process_image_sequence_wholebody(image_dir, data_dir, pose_output_dir, show=False, vis_output_dir=None, sequence_name=None, resample_to=(640, 480)):
    """Process a sequence of images to extract wholebody poses (body + hands + face)
    
    First tracks each detected person using EfficientTAM or SAM, then performs wholebody
    pose estimation on each tracked person. This ensures consistent person IDs across frames.
    
    Args:
        image_dir: Directory containing the image sequence
        data_dir: Base directory containing the image_dir
        pose_output_dir: Directory to save pose results
        show: Whether to visualize the poses during processing
        vis_output_dir: Directory to save visualization video (if None, defaults to 'vis_dir')
        sequence_name: Name of the sequence (if None, uses the image_dir basename)
        resample_to: Tuple for resampling images (width, height)
        
    Returns:
        Number of processed frames or False if image directory not found
    """

    def box_tracked(tracks, frame_idx, box):
        if len(box.shape) == 2:
            box = box.flatten()
        """Check if box is already tracked in the tracks"""
        box_mean_x = (box[0] + box[2]) / 2
        box_mean_y = (box[1] + box[3]) / 2
        if frame_idx not in tracks:
            return False

        for tracked_box in tracks[frame_idx].values():
            if len(tracked_box) == 0:
                continue
            tracked_box = tracked_box.flatten()
            tracked_box_mean_x = (tracked_box[0] + tracked_box[2]) / 2
            tracked_box_mean_y = (tracked_box[1] + tracked_box[3]) / 2
            tracked_in_box = (tracked_box_mean_x >= box[0] and tracked_box_mean_x <= box[2] and 
                             tracked_box_mean_y >= box[1] and tracked_box_mean_y <= box[3])
            box_in_tracked = (box_mean_x >= tracked_box[0] and box_mean_x <= tracked_box[2] and 
                             box_mean_y >= tracked_box[1] and box_mean_y <= tracked_box[3])
            if tracked_in_box and box_in_tracked:
                return True
        return False
    from helpers.predictors import add_object, track_object

    
    full_image_dir = os.path.join(data_dir, image_dir)
    if not os.path.exists(full_image_dir):
        print(f"Image directory not found: {full_image_dir}")
        return False
    
    jpg_image_dir = os.path.join(data_dir, image_dir + '_jpg')
    if not os.path.exists(jpg_image_dir):
        print(f"JPG image directory not found: {jpg_image_dir}")
        jpg_image_dir = prepare_video_frames(full_image_dir, resample_to=resample_to, from_imgs=True, out_jpg_dir=jpg_image_dir)

    # Get sequence name from directory if not provided
    if sequence_name is None:
        sequence_name = os.path.basename(os.path.normpath(image_dir))
    
    print(f"Processing image sequence '{sequence_name}' with wholebody model and person tracking...")
    
    # Setup visualization output directory if needed
    if show and vis_output_dir is None:
        vis_output_dir = 'vis_dir'
    
    if show and not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
    
    # Check if already processed
    if os.path.exists(os.path.join(pose_output_dir, f"{sequence_name}_wholebody.json")):
        print(f"Skipping {sequence_name} - already processed")
        image_files = [f for f in os.listdir(full_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return len(image_files)
    
    # Get all image files and sort them
    image_files = [f for f in os.listdir(full_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files_jpg = [f for f in os.listdir(jpg_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Sort files by numeric part of filename, assuming frame index is in the filename
    try:
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image_files_jpg.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    except ValueError:
        # Fallback to regular sorting if numeric sorting fails
        image_files.sort()
        image_files_jpg.sort()
    
    if not image_files:
        print(f"No image files found in {full_image_dir}")
        return False
        
    # Process frames and estimate poses
    body_instances_list = []
    left_hand_instances_list = []
    right_hand_instances_list = []
    bbox_instances = []  # To store detected bounding boxes for each frame
    
    # Setup video writer for visualization if needed
    video_writer = None

    # Get sample image dimensions
    sample_img_path = os.path.join(full_image_dir, image_files[0])
    sample_img = cv2.imread(sample_img_path)
    height, width = sample_img.shape[:2]
    print(f"Original image resolution: {width}x{height}")
    original_resolution = (width, height)
    if show:
        output_video_path = os.path.join(vis_output_dir, f"{sequence_name}_wholebody.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Assuming 30 fps for visualization
        video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            30,
            (width, height)
        )
        print(f"Visualizations will be saved to {output_video_path}")
    
    # Step 1: Detect people in all frames
    for i, img_file in enumerate(image_files_jpg):
        img_path = os.path.join(jpg_image_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        # Detect people in the current frame
        bboxes = detect_body(img_path)
        
        # Convert bboxes to JSON-serializable format
        bboxes_json = []
        for bbox in bboxes:
            bbox_json = {
                'bbox': np.array(bbox).astype(float).tolist()
            }
            bboxes_json.append(bbox_json)
        
        bbox_instances.append(dict(
            frame_id=i,
            instances=bboxes_json
        ))
        
        print(f"\rDetected {len(bboxes_json)} people in frame {i+1}/{len(image_files)}", end='')

    # Detect people in the first frame
    initial_bboxes = map(lambda x: x['bbox'], bbox_instances[0]['instances'])

    # Make sure bboxes are in correct format
    initial_bboxes = np.array(list(initial_bboxes))
    if len(initial_bboxes.shape) == 1:
        initial_bboxes = initial_bboxes.reshape((1, 4))
    
    print(initial_bboxes)
    
    print(f"Detected {len(initial_bboxes)} persons in the first frame")
    
    # Step 2: Initialize tracking for each detected person
    predictor = None
    inference_state = None
    all_tracks = None 
    
    # Track each detected person
    for obj_id, bbox in enumerate(initial_bboxes):
        print(f"Initializing tracking for person {obj_id}")
        
        # Initialize tracker with first frame's box
        inference_state, predictor, frame_names = add_object(
            jpg_image_dir,
            input_box=bbox,
            frame_idx=0,
            obj_id=obj_id,
            show=False,
            inference_state=inference_state,
            predictor_in=predictor
        )
        
        # Track the person through all frames
        _, person_tracks = track_object(
            jpg_image_dir,
            inference_state,
            predictor,
            frame_names,
            show=False,
            prev_bboxes=all_tracks,
            whole_mask=True  # Get single bbox per person
        )
        
        # Merge tracking results
        all_tracks = person_tracks
    
    print(f"Tracking completed for initial bboxes.")

    obj_id = len(initial_bboxes)
    
    for frame_idx in list(map(lambda x: x['frame_id'], bbox_instances)):
        if frame_idx not in all_tracks:
            all_tracks[frame_idx] = {}
        
        for bbox in bbox_instances[frame_idx]['instances']:
            # Convert bbox to numpy array
            bbox_array = np.array(bbox['bbox'])
            if len(bbox_array.shape) == 1:
                bbox_array = bbox_array.reshape((1, 4))
            
            # Check if bbox is already tracked
            if bbox_array.shape[0] == 0:
                continue
            # If bbox matches an existing track, skip adding it again
            if not box_tracked(all_tracks, frame_idx, bbox_array):
                print(f"Initializing tracking for person {obj_id}")
    
                # Initialize tracker with first frame's box
                inference_state, predictor, frame_names = add_object(
                    jpg_image_dir,
                    input_box=bbox_array,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    show=False,
                    inference_state=inference_state,
                    predictor_in=predictor
                )
                
                # Track the person through all frames
                _, person_tracks = track_object(
                    jpg_image_dir,
                    inference_state,
                    predictor,
                    frame_names,
                    show=False,
                    prev_bboxes=all_tracks,
                    whole_mask=True  # Get single bbox per person
                )
                
                # Merge tracking results
                all_tracks = person_tracks
                obj_id += 1

    print(f"Tracking completed for all detected persons.")
    print(f"Total tracked persons: {obj_id}")

    all_tracks = adjust_tracks(all_tracks, resample_to, original_resolution)


    # Process each image now with tracking information
    for frame_idx, img_file in enumerate(image_files):
        # Load image
        img_path = os.path.join(full_image_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        # Initialize empty lists for this frame
        frame_body_instances = []
        frame_left_hand_instances = []
        frame_right_hand_instances = []
        
        # Get tracked bboxes for this frame
        if frame_idx not in all_tracks:
            print(f"No tracking data for frame {frame_idx}, skipping")
            # Add empty instances for this frame
            body_instances_list.append(dict(
                frame_id=frame_idx,
                instances=[]
            ))
            left_hand_instances_list.append(dict(
                frame_id=frame_idx,
                instances=[]
            ))
            right_hand_instances_list.append(dict(
                frame_id=frame_idx,
                instances=[]
            ))
            continue
            
        body_instances_json = []
        left_hand_instances_json = []
        right_hand_instances_json = []

        # Process each tracked person in this frame
        for obj_id, bbox_array in all_tracks[frame_idx].items():
            # Skip if no valid bounding boxes
            if len(bbox_array) == 0:
                continue
                
            # Get the first bounding box for this person
            # (usually only one box per person with whole_mask=True)
            bbox = bbox_array[0] if len(bbox_array.shape) > 1 else bbox_array
            
            # Estimate wholebody pose for this person
            body_instances, left_hand_instances, right_hand_instances = estimate_pose(
                frame, bbox.reshape(1, 4), pose_type='wholebody'
            )
        
            # Convert body instances to JSON-serializable format
            if body_instances is not None:
                for instance in split_instances(body_instances):
                    instance_json = {
                        'person_id': obj_id,
                        'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                        'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                    }
                    if 'bbox' in instance:
                        instance_json['bbox'] = np.array(instance['bbox']).astype(float).tolist()
                        if 'bbox_score' in instance:
                            instance_json['bbox_score'] = float(instance['bbox_score'])
                    body_instances_json.append(instance_json)
            
            # Convert left hand instances to JSON-serializable format
            if left_hand_instances is not None:
                for instance in split_instances(left_hand_instances):
                    instance_json = {
                        'person_id': obj_id,
                        'keypoints': np.array(instance['keypoints']).astype(float).tolist(),
                        'keypoint_scores': np.array(instance['keypoint_scores']).astype(float).tolist()
                    }
                    left_hand_instances_json.append(instance_json)
            
            # Convert right hand instances to JSON-serializable format
            if right_hand_instances is not None:
                for instance in split_instances(right_hand_instances):
                    instance_json = {
                        'person_id': obj_id,
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
            
            for body_instances in body_instances_json:
                obj_id = body_instances['person_id']
                # Draw bounding box
                if 'bbox' in body_instances:
                    bbox = body_instances['bbox']
                    if len(bbox) != 4 and len(bbox) > 0:
                        bbox = bbox[0]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # write person ID on the box
                        cv2.putText(vis_frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw body keypoints
                if body_instances is not None:
                    kpts = body_instances['keypoints']
                    scores = body_instances['keypoint_scores']
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
            
            for left_hand_instances in left_hand_instances_json:
                obj_id = body_instances['person_id']
                # Draw left hand keypoints
                if left_hand_instances is not None:
                    kpts = left_hand_instances['keypoints']
                    scores = left_hand_instances['keypoint_scores']
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
            
            for right_hand_instances in right_hand_instances_json:
                obj_id = body_instances['person_id']
                # Draw right hand keypoints
                if right_hand_instances is not None:
                    kpts = right_hand_instances['keypoints']
                    scores = right_hand_instances['keypoint_scores']
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
        
        print(f"\rProcessing frame {frame_idx+1}/{len(image_files)}", end='')
    
    print(". Done!")
    
    # Release resources
    if show and video_writer is not None:
        video_writer.release()
        cv2.destroyAllWindows()
    
    # Save predictions
    save_path = os.path.join(pose_output_dir, f"{sequence_name}_wholebody.json")
    with open(save_path, 'w') as f:
        json.dump({
            'body_instances': body_instances_list,
            'left_hand_instances': left_hand_instances_list,
            'right_hand_instances': right_hand_instances_list
        }, f, indent=2)
    
    return len(image_files)

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

def prepare_video_frames(video_path, resample_to=None, from_imgs=False, out_jpg_dir=None):
    """Create frames directory if video file exists"""

    if from_imgs:
        # If from_imgs is True, assume video_path is a directory of images
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Image directory not found: {video_path}")
        if out_jpg_dir is not None:
            frames_dir = out_jpg_dir
        else:
            frames_dir = video_path + '_jpg'
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
            # Copy images to frames_dir
            for img_file in os.listdir(video_path):
                if img_file.lower().endswith(('.png')):
                    jpg_file = img_file.replace('.png', '.jpg')
                    src_path = os.path.join(video_path, img_file)
                    dst_path = os.path.join(frames_dir, jpg_file)
                    if resample_to is not None:
                        img = cv2.imread(src_path)
                        img = cv2.resize(img, resample_to)
                        cv2.imwrite(dst_path, img)
                    else:
                        os.rename(src_path, dst_path)
        return frames_dir
    
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
