import os
import json
import numpy as np

def read_keypoints_from_json(folder_path, num_keypoints, cam_names, num_frames, start_frame=0, suffix='_keypoints', identifier=None, remove_features=None, body=True, hands=False):
    """
    Reads 2D keypoints from JSON files and organizes them into the required format.
    
    Args:
        folder_path: Path to the folder containing JSON files for all frames and cameras.
        num_keypoints: Number of keypoints per frame.
        cam_names: List of camera names.
        num_frames: Number of frames in the dataset.
    
    Returns:
        poses_2d: A list of 2D poses where each entry is in the format 
                  poses_2d[frame][camera][keypoint] -> (x, y, confidence).
    """
    # Dictionary to hold the 2D poses for each frame and camera
    poses_2d = []

    # Get a sorted list of all JSON files in the folder
    if identifier is not None:
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f'{suffix}.json') and identifier in f])
    else:
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f'{suffix}.json') and 'selected_0' in f]) # and f.split("_")[2] == "hands"]

    # Organize JSON files by frame and camera (assuming the filenames are frame_camera.json)
    # This will depend on how filenames are structured. Adjust according to your naming convention.
    
    frames_data = {}

    for file_name in json_files:
        # Assuming filename format: 'camera_[cam_suffix]_frame_keypoints.json', adjust accordingly
        parts = file_name.replace(f"{suffix}.json", "").split("_")
        cam_name = parts[0]
        if cam_name in cam_names:
            if suffix == '_test':
                frame_idx = 150
                camera_idx = cam_names.index(cam_name)

                # Load the JSON file
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    keypoints_data = json.load(f)
            
                keypoints_flat = [p['pose_keypoints_2d'] for p in keypoints_data['people']][0]  # This is a flat array: [x1, y1, c1, x2, y2, c2, ...]
                
                if len(keypoints_flat) > 0:

                    if remove_features is not None:
                        indexes_to_remove = []
                        for feat in remove_features:
                            indexes_to_remove.extend([feat*3, feat*3+1, feat*3+2])
                        for i in sorted(indexes_to_remove, reverse=True):
                            del keypoints_flat[i]

                    # Ensure that we have the expected number of keypoints (3 * num_keypoints)
                    assert len(keypoints_flat) == 3 * num_keypoints, f"Unexpected number of keypoints in {file_name}: {len(keypoints_flat)}"

                    # Reshape the flat array into (x, y, confidence) tuples for each keypoint
                    keypoints = [(keypoints_flat[i], keypoints_flat[i + 1], keypoints_flat[i + 2]) 
                                for i in range(0, len(keypoints_flat), 3)]
                    
                else:
                    keypoints = [(0, 0, 0)]*num_keypoints
                    
                # Store in frames_data by frame and camera
                if frame_idx not in frames_data:
                    frames_data[frame_idx] = [[(0, 0, 0)]*num_keypoints for _ in range(len(cam_names))]
                
                frames_data[frame_idx][camera_idx] = keypoints
            else:
                frame_idx = int(parts[-1])
                if (hands and parts[2] == 'hands') or (not hands and len(parts) == 3) and frame_idx < (num_frames + start_frame) and frame_idx >= start_frame:
                    camera_idx = cam_names.index(cam_name)

                    # Load the JSON file
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as f:
                        keypoints_data = json.load(f)
                
                    keypoints_flat_body = [p['pose_keypoints_2d'] for p in keypoints_data['people']]  # This is a flat array: [x1, y1, c1, x2, y2, c2, ...]
                    keypoints_flat_hand_left = [p['hand_left_keypoints_2d'] for p in keypoints_data['people']]
                    keypoints_flat_hand_right = [p['hand_right_keypoints_2d'] for p in keypoints_data['people']]

                    if len(keypoints_flat_body) > 0:
                        keypoints_flat_body = keypoints_flat_body[0]

                    if len(keypoints_flat_hand_left) > 0:
                        keypoints_flat_hand_left = keypoints_flat_hand_left[0]
                        
                    if len(keypoints_flat_hand_right) > 0:
                        keypoints_flat_hand_right = keypoints_flat_hand_right[0]

                    if body:
                        keypoints_flat = keypoints_flat_body
                    else:
                        keypoints_flat = []

                    if hands:
                        keypoints_flat.extend(keypoints_flat_hand_left)
                        keypoints_flat.extend(keypoints_flat_hand_right)
                    
                    if len(keypoints_flat) > 0:

                        if remove_features is not None:
                            indexes_to_remove = []
                            for feat in remove_features:
                                indexes_to_remove.extend([feat*3, feat*3+1, feat*3+2])
                            for i in sorted(indexes_to_remove, reverse=True):
                                del keypoints_flat[i]

                        # Ensure that we have the expected number of keypoints (3 * num_keypoints)
                        assert len(keypoints_flat) == 3 * num_keypoints, f"Unexpected number of keypoints in {file_name}: {len(keypoints_flat)/3}, should be {num_keypoints}"

                        # Reshape the flat array into (x, y, confidence) tuples for each keypoint
                        keypoints = [(keypoints_flat[i], keypoints_flat[i + 1], keypoints_flat[i + 2]) 
                                    for i in range(0, len(keypoints_flat), 3)]
                        
                    else:
                        keypoints = [(0, 0, 0)]*num_keypoints
                        
                    # Store in frames_data by frame and camera
                    if frame_idx not in frames_data:
                        frames_data[frame_idx] = [[(0, 0, 0)]*num_keypoints for _ in range(len(cam_names))]
                    
                    frames_data[frame_idx][camera_idx] = keypoints
        
        

    # Convert frames_data to a list of frames
    num_frames = len(frames_data)
    poses_2d = [frames_data[frame_idx] for frame_idx in range(start_frame, start_frame + num_frames)]

    return poses_2d

def read_keypoints_mmpose(folder_path, cam_names, num_frames, suffix, num_body_keypoints=26, num_hand_keypoints=21, use_wholebody=False):
    """
    Reads 2D keypoints from JSON files and organizes them into the required format.
    
    Args:
        folder_path: Path to the folder containing JSON files.
        cam_names: List of camera names.
        num_frames: Number of frames in the dataset.
        suffix: Suffix of the json files.
        num_body_keypoints: (Optional) Number of keypoints in the estimated pose [default: 26]
        num_hand_keypoints: (Optional) Number of keypoints per hand in the estimated pose [default:21]
        use_wholebody: (Optional) Whether to read from wholebody model output files [default: False]
    
    Returns:
        poses_2d_body: A list of 2D poses where each entry is in the format 
                  poses_2d_body[frame][camera][keypoint] -> (x, y, confidence).

        poses_2d_hands: A list of 2D poses where each entry is in the format 
                  poses_2d_hands[frame][camera][keypoint] -> (x, y, confidence).
        
    """
    # Dictionary to hold the 2D poses for each frame and camera
    frames_data = {"body":{}, "hands": {}}

    # Get a sorted list of all JSON files in the folder
    for camera_idx, cam_name in enumerate(cam_names):
        body = True
        hands = True
        
        if use_wholebody:
            try:
                json_file = os.path.join(f'{folder_path}', f'{cam_name}{suffix}_wholebody.json')
                with open(json_file, 'r') as f:
                    wholebody_data = json.load(f)
                    keypoints_data_body = wholebody_data['body_instances']
                    left_hand_data = wholebody_data['left_hand_instances']
                    right_hand_data = wholebody_data['right_hand_instances']
            except Exception as e:
                print(f"Wholebody keypoints not available for cam: {cam_name}")
                print(f"Error: {str(e)}")
                body = False
                hands = False
        else:
            try:
                json_file = os.path.join(f'{folder_path}', f'{cam_name}{suffix}_body.json')
                with open(json_file, 'r') as f:
                    keypoints_data_body = json.load(f)['instance_info']
            except:
                print(f"Body keypoints not available for cam: {cam_name}")
                body = False

            try:
                json_file = os.path.join(f'{folder_path}', f'{cam_name}{suffix}_hands.json')
                with open(json_file, 'r') as f:
                    keypoints_data_hands = json.load(f)['instance_info']
            except:
                print(f"Hand keypoints not available for cam: {cam_name}")
                hands = False        
        
        for frame_idx in range(num_frames):
            
            if frame_idx not in frames_data['body']:
                frames_data['body'][frame_idx] = [[(0, 0, 0)]*num_body_keypoints for _ in range(len(cam_names))]

            if frame_idx not in frames_data['hands']:
                frames_data['hands'][frame_idx] = [[(0, 0, 0)]*(2*num_hand_keypoints) for _ in range(len(cam_names))]

            if body:
                if use_wholebody:
                    body_instances = keypoints_data_body[frame_idx]['instances']
                else:
                    body_instances = keypoints_data_body[frame_idx]['instances']
                
                if len(body_instances) > 1:
                    print("More than one body detected")    

                try:
                    body_keypoints = body_instances[0]['keypoints']
                    body_confidence = body_instances[0]['keypoint_scores']
                    keypoints_body = [(body_keypoints[i][0], body_keypoints[i][1], body_confidence[i]) for i in range(num_body_keypoints)]
                    
                    frames_data['body'][frame_idx][camera_idx] = keypoints_body

                except Exception as e:
                    print(f"body keypoints could not be accessed for {frame_idx}")
                    print(e)
                
            if hands:
                if use_wholebody:
                    # For wholebody model, we need to combine left and right hand data
                    left_hand_instances = left_hand_data[frame_idx]['instances']
                    right_hand_instances = right_hand_data[frame_idx]['instances']
                    
                    hand_keypoints = []
                    hand_confidence = []
                    
                    # Process left hand
                    if left_hand_instances and len(left_hand_instances) > 0:
                        left_keypoints = left_hand_instances[0]['keypoints']
                        left_confidence = left_hand_instances[0]['keypoint_scores']
                        hand_keypoints.extend(left_keypoints)
                        hand_confidence.extend(left_confidence)
                    else:
                        hand_keypoints.extend([[0, 0]]*num_hand_keypoints)
                        hand_confidence.extend([0]*num_hand_keypoints)
                    
                    # Process right hand
                    if right_hand_instances and len(right_hand_instances) > 0:
                        right_keypoints = right_hand_instances[0]['keypoints']
                        right_confidence = right_hand_instances[0]['keypoint_scores']
                        hand_keypoints.extend(right_keypoints)
                        hand_confidence.extend(right_confidence)
                    else:
                        hand_keypoints.extend([[0, 0]]*num_hand_keypoints)
                        hand_confidence.extend([0]*num_hand_keypoints)
                else:
                    hand_instances = keypoints_data_hands[frame_idx]['instances']
                    if len(hand_instances) > 2:
                        print("More than 2 hands detected")

                    try:
                        hand_keypoints = hand_instances[0]['keypoints']
                        hand_confidence = hand_instances[0]['keypoint_scores']
                        if len(hand_instances) > 1:
                            hand_keypoints.extend(hand_instances[1]['keypoints'])
                            hand_confidence.extend(hand_instances[1]['keypoint_scores'])
                        else: 
                            hand_keypoints.extend([[0, 0]]*num_hand_keypoints)
                            hand_confidence.extend([0]*num_hand_keypoints)

                    except Exception as e:
                        print(f"hand keypoints could not be accessed for {frame_idx}")
                        print(e)
                
                
                keypoints_hands = [(hand_keypoints[i][0], hand_keypoints[i][1], hand_confidence[i]) for i in range(len(hand_keypoints))]
                frames_data['hands'][frame_idx][camera_idx] = keypoints_hands
                

    poses_2d_body = [frames_data['body'][frame_idx] for frame_idx in range(num_frames)]
    poses_2d_hands = [frames_data['hands'][frame_idx] for frame_idx in range(num_frames)]

    return poses_2d_body, poses_2d_hands

def read_detection_file(filepath, with_confidence=False, resolution=None, original_resolution=None):
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

