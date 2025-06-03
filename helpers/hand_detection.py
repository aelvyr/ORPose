import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import json

def crop_img(img, box):
    """Crop image based on bounding box"""
    x1, y1, x2, y2 = box
    return img[int(y1):int(y2), int(x1):int(x2)]

def detect_hands_in_frame(frame, model_person, model_hand):
    """Detect hands in a single frame using YOLO model"""
    # Process frame with YOLO
    results = model_person(frame, classes=0)
    # print(results)
    boxes = []
    confidence = []
    
    if len(results) == 0:
        results_hands = model_hand(frame)
        for r in results_hands:
            boxes_tensor = r.boxes.xyxy.cpu()
            confs = r.boxes.conf.cpu()
            for box, conf in zip(boxes_tensor, confs):
                if conf > model_hand.conf:
                    boxes.append(box.flatten())
                    confidence.append(conf)
    # Process detections
    for r in results:
        boxes_tensor = r.boxes.xyxy.cpu()
        confs = r.boxes.conf.cpu()
        for box1, conf in zip(boxes_tensor, confs):
            cropped = crop_img(frame, box1)
            results_hands = model_hand(cropped)
            for r in results_hands:
                boxes_tensor = r.boxes.xyxy.cpu()
                confs = r.boxes.conf.cpu()
                for box2, conf in zip(boxes_tensor, confs):
                    if conf > model_hand.conf:
                        # print(box2)
                        adjusted_box = np.add(np.array(box2).reshape(2, 2), box1[:2])
                        boxes.append(adjusted_box.flatten())
                        confidence.append(conf)
                       
                
    return np.array(boxes) if boxes else np.array([]), np.array(confidence) if confidence else np.array([])

def process_video_rohan(input_file, model_person, model_hand, show=True):
    """Process video file for hand detection"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_file}")
    
    frame_idx = 0
    output_boxes = []
    
    success = True
    while cap.isOpened() and success:   
        success, frame = cap.read()
        if not success:
            break
            
        # Detect hands
        boxes, confidence = detect_hands_in_frame(frame, model_person, model_hand)
        
        if len(boxes) > 0:
            output_boxes.append((boxes, confidence, frame_idx))
            
        if show:
            # Visualize detections
            for box, conf in zip(boxes, confidence):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Hand, conf: {round(100*conf, 1)}%", (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detections', 1920, 1080)
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) == 113:
                success = False

                
        frame_idx += 1
        print(f"Processing frame {frame_idx}", end='\r')
    
    cap.release()
    if show:
        cv2.destroyAllWindows()
        
    return output_boxes

def save_to_file(boxes, output_file):
    # Save output to file
    with open(output_file, 'w+') as f:
        for boxes, conf, frame_idx in boxes:
            f.write(f"{frame_idx} ")
            for i, box in enumerate(boxes):
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {conf[i]} ")
            f.write("\n")

def read_detection_file(filepath, resolution=None, original_resolution=None, with_confidence=False):
    """Read detection file containing frame index and bounding boxes"""
    detections = {}
    stride = 5 if with_confidence else 4
    with open(filepath, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            frame_idx = int(values[0])
            boxes = []
            # Each box has 4 coordinates
            for i in range(1, len(values), stride):
                boxes.append(values[i:i+4])

            boxes = np.array(boxes)

            if resolution is not None and original_resolution is not None:
                # Rescale bounding boxes to original resolution
                boxes[:, 0] *= resolution[0] / original_resolution[0]
                boxes[:, 1] *= resolution[1] / original_resolution[1]
                boxes[:, 2] *= resolution[0] / original_resolution[0]
                boxes[:, 3] *= resolution[1] / original_resolution[1]
            detections[frame_idx] = boxes
    return detections

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

def process_video_hands(video_name, data_dir, detection_dir, resolution, original_resolution, max_hands, 
                       show=True, predictor_module=None, add_object_fn=None, track_object_fn=None):
    """Process a single video file and track all boxes as one object"""

    def box_tracked(tracks, frame_idx, box):
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

    # Get paths
    video_path = os.path.join(data_dir, f"{video_name}.MP4")
    detection_path = os.path.join(detection_dir, f"{video_name}.txt")

    if not os.path.exists(detection_path):
        print(f"No detection file found for {video_name}")
        return None, None

    # Prepare video frames
    frames_dir = prepare_video_frames(video_path, resolution)

    # Read detections
    detections = read_detection_file(detection_path, resolution, original_resolution, with_confidence=True)

    # Initialize tracker for all boxes under single ID
    obj_id = 0
    all_tracks = None
    predictor = None
    inference_state = None
    frame_names = None

    # Process each frame with detections
    for frame_idx, boxes in detections.items():
        for box in boxes:
            if (all_tracks is None or not box_tracked(all_tracks, frame_idx, box)) and obj_id < max_hands:
                print(f"Frame {frame_idx}: {box}")
                print(f"Object ID: {obj_id}")
                # Initialize tracker with first frame's boxes if not done yet
                inference_state, predictor, frame_names = add_object_fn(
                    frames_dir,
                    input_box=box,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    show=show,
                    inference_state=inference_state,
                    predictor_in=predictor)

                # Track the object through video
                _, track_boxes = track_object_fn(
                    frames_dir,
                    inference_state,
                    predictor,
                    frame_names,
                    show=show,
                    prev_bboxes=None,
                    whole_mask=True)

                obj_id += 1
                all_tracks = track_boxes

    if frame_names is None:
        return None, None

    return all_tracks, frame_names

def adjust_tracks(tracks, adj_resolution, orig_resolution):
    for frame_idx, frame_tracks in tracks.items():
        for obj_id, box in frame_tracks.items():
            box = box.flatten()
            if len(box) == 0:
                continue
            box[0] *= orig_resolution[0] / adj_resolution[0]
            box[1] *= orig_resolution[1] / adj_resolution[1]
            box[2] *= orig_resolution[0] / adj_resolution[0]
            box[3] *= orig_resolution[1] / adj_resolution[1]
            tracks[frame_idx][obj_id] = box
    return tracks

def process_video_wholebody_hands(video_name, data_dir, pose_dir, save_dir=None, show=False, margin_ratio=0.15):
    """
    Process a video using wholebody model outputs to detect hands.
    
    This function reads the wholebody JSON output files and extracts hand keypoints
    to create bounding boxes around hands.
    
    Args:
        video_name: Name of the video file (without extension)
        data_dir: Directory containing the video
        pose_dir: Directory containing the JSON files
        save_dir: Directory to save the detection file (if None, will save to data_dir)
        show: Whether to display the detections (default: False)
        margin_ratio: Ratio of width/height to add as margin around bounding boxes (default: 0.15)
    
    Returns:
        output_boxes: List of tuples (boxes, confidence, frame_idx) containing hand detections
    """
    # Define paths
    video_path = os.path.join(data_dir, f"{video_name}.MP4")
    json_path = os.path.join(pose_dir, f"{video_name}_wholebody.json")
    
    if save_dir is None:
        save_dir = data_dir
    
    if not os.path.exists(json_path):
        print(f"No wholebody JSON file found for {video_name}")
        return []
    
    # Read wholebody JSON file
    with open(json_path, 'r') as f:
        wholebody_data = json.load(f)
    
    # Initialize video capture for visualization
    cap = None
    if show:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
    
    output_boxes = []
    
    # Process each frame
    for frame_idx in range(len(wholebody_data['left_hand_instances'])):
        left_hand_instances = wholebody_data['left_hand_instances'][frame_idx]['instances']
        right_hand_instances = wholebody_data['right_hand_instances'][frame_idx]['instances']
        
        boxes = []
        confidence = []
        
        # Process left hand
        if left_hand_instances and len(left_hand_instances) > 0:
            left_keypoints = left_hand_instances[0]['keypoints']
            left_confidence = left_hand_instances[0]['keypoint_scores']
              # Check if bounding box exists, otherwise create one from keypoints
            if 'bbox' in left_hand_instances[0]:
                # Use existing bounding box
                left_bbox = left_hand_instances[0]['bbox']  # [x, y, w, h]
                # Add margin to the bounding box
                margin_w = left_bbox[2] * margin_ratio
                margin_h = left_bbox[3] * margin_ratio
                boxes.append([
                    max(0, left_bbox[0] - margin_w), 
                    max(0, left_bbox[1] - margin_h),
                    left_bbox[0] + left_bbox[2] + margin_w, 
                    left_bbox[1] + left_bbox[3] + margin_h
                ])
                # Use average confidence as the detection confidence
                confidence.append(np.mean(left_confidence))
            else:                # Create bounding box from keypoints
                valid_keypoints = [(kp[0], kp[1]) for kp, conf in zip(left_keypoints, left_confidence) if conf > 0.2]
                
                if valid_keypoints:
                    keypoints_array = np.array(valid_keypoints)
                    x_min = np.min(keypoints_array[:, 0])
                    y_min = np.min(keypoints_array[:, 1])
                    x_max = np.max(keypoints_array[:, 0])
                    y_max = np.max(keypoints_array[:, 1])
                    
                    # Add padding to the box
                    width = x_max - x_min
                    height = y_max - y_min
                    padding_x = width * margin_ratio
                    padding_y = height * margin_ratio
                    
                    boxes.append([
                        max(0, x_min - padding_x),
                        max(0, y_min - padding_y),
                        x_max + padding_x,
                        y_max + padding_y
                    ])
                    # Use average confidence as the detection confidence
                    confidence.append(np.mean(left_confidence))
        
        # Process right hand
        if right_hand_instances and len(right_hand_instances) > 0:
            right_keypoints = right_hand_instances[0]['keypoints']
            right_confidence = right_hand_instances[0]['keypoint_scores']
              # Check if bounding box exists, otherwise create one from keypoints
            if 'bbox' in right_hand_instances[0]:
                # Use existing bounding box
                right_bbox = right_hand_instances[0]['bbox']  # [x, y, w, h]
                # Add margin to the bounding box
                margin_w = right_bbox[2] * margin_ratio
                margin_h = right_bbox[3] * margin_ratio
                boxes.append([
                    max(0, right_bbox[0] - margin_w), 
                    max(0, right_bbox[1] - margin_h),
                    right_bbox[0] + right_bbox[2] + margin_w, 
                    right_bbox[1] + right_bbox[3] + margin_h
                ])
                # Use average confidence as the detection confidence
                confidence.append(np.mean(right_confidence))
            else:                # Create bounding box from keypoints
                valid_keypoints = [(kp[0], kp[1]) for kp, conf in zip(right_keypoints, right_confidence) if conf > 0.2]
                
                if valid_keypoints:
                    keypoints_array = np.array(valid_keypoints)
                    x_min = np.min(keypoints_array[:, 0])
                    y_min = np.min(keypoints_array[:, 1])
                    x_max = np.max(keypoints_array[:, 0])
                    y_max = np.max(keypoints_array[:, 1])
                    
                    # Add padding to the box
                    width = x_max - x_min
                    height = y_max - y_min
                    padding_x = width * margin_ratio
                    padding_y = height * margin_ratio
                    
                    boxes.append([
                        max(0, x_min - padding_x),
                        max(0, y_min - padding_y),
                        x_max + padding_x,
                        y_max + padding_y
                    ])
                    # Use average confidence as the detection confidence
                    confidence.append(np.mean(right_confidence))
        
        # Convert lists to numpy arrays
        boxes_array = np.array(boxes)
        confidence_array = np.array(confidence)
        
        if len(boxes_array) > 0:
            output_boxes.append((boxes_array, confidence_array, frame_idx))
        
        # Visualize detections if show=True
        if show and cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw detections
            for box, conf in zip(boxes, confidence):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Hand, conf: {round(100*conf, 1)}%", 
                            (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            cv2.namedWindow('Wholebody Hand Detections', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Wholebody Hand Detections', 1920, 1080)
            cv2.imshow('Wholebody Hand Detections', frame)
            if cv2.waitKey(1) == 113:  # 'q' key
                break
                
        print(f"Processing frame {frame_idx}", end='\r')
    
    # Save detection results to file
    if save_dir:
        output_file = os.path.join(save_dir, f"{video_name}_wholebody_hands.txt")
        with open(output_file, 'w+') as f:
            for boxes, conf, frame_idx in output_boxes:
                f.write(f"{frame_idx} ")
                for i, box in enumerate(boxes):
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {conf[i]} ")
                f.write("\n")
        print(f"Detection results saved to {output_file}")
    
    # Clean up
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    
    return output_boxes

def process_video_wholebody_hands_for_tracking(video_name, data_dir, detection_dir, resolution, original_resolution, max_hands, 
                                              show=True, add_object_fn=None, track_object_fn=None, margin_ratio=0.15):
    """
    Process a single video file using wholebody hand detections and track all boxes.
    This is analogous to the process_video_hands function but uses wholebody model outputs.
    
    Args:
        video_name: Name of the video file (without extension)
        data_dir: Directory containing the video file
        detection_dir: Directory containing the detection files
        resolution: Resolution to resize frames to
        original_resolution: Original resolution of the video
        max_hands: Maximum number of hands to track
        show: Whether to display the tracking results
        add_object_fn: Function to add a new object for tracking
        track_object_fn: Function to track an object
        margin_ratio: Ratio of width/height to add as margin around bounding boxes (default: 0.15)
        
    Returns:
        all_tracks: Dictionary containing track information
        frame_names: List of frame names
    """
    def box_tracked(tracks, frame_idx, box):
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

    # Get paths
    video_path = os.path.join(data_dir, f"{video_name}.MP4")
    detection_path = os.path.join(detection_dir, f"{video_name}_wholebody_hands.txt")

    if not os.path.exists(detection_path):
        print(f"No wholebody hand detection file found for {video_name}")        # If the detection file doesn't exist, try to generate it
        print(f"Generating wholebody hand detection file for {video_name}...")
        process_video_wholebody_hands(video_name, data_dir, detection_dir, detection_dir, show=False, margin_ratio=margin_ratio)
        
        if not os.path.exists(detection_path):
            print(f"Failed to generate wholebody hand detection file for {video_name}")
            return None, None

    # Prepare video frames
    frames_dir = prepare_video_frames(video_path, resolution)

    # Read detections
    detections = read_detection_file(detection_path, resolution, original_resolution, with_confidence=True)

    # Initialize tracker for all boxes under single ID
    obj_id = 0
    all_tracks = None
    predictor = None
    inference_state = None
    frame_names = None

    # Process each frame with detections
    for frame_idx, boxes in detections.items():
        for box in boxes:
            if (all_tracks is None or not box_tracked(all_tracks, frame_idx, box)) and obj_id < max_hands:
                print(f"Frame {frame_idx}: {box}")
                print(f"Object ID: {obj_id}")
                # Initialize tracker with first frame's boxes if not done yet
                inference_state, predictor, frame_names = add_object_fn(
                    frames_dir,
                    input_box=box,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    show=show,
                    inference_state=inference_state,
                    predictor_in=predictor)

                # Track the object through video
                _, track_boxes = track_object_fn(
                    frames_dir,
                    inference_state,
                    predictor,
                    frame_names,
                    show=show,
                    prev_bboxes=None,
                    whole_mask=True)

                obj_id += 1
                all_tracks = track_boxes

    if frame_names is None:
        return None, None

    return all_tracks, frame_names
