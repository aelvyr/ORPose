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

def read_keypoints_mmpose(folder_path, cam_names, num_frames, suffix, num_body_keypoints=17, num_hand_keypoints=21, use_wholebody=False, multi_person=False, matched_poses_path=None):
    """
    Reads 2D keypoints from JSON files and organizes them into the required format.
    
    Args:
        folder_path: Path to the folder containing JSON files.
        cam_names: List of camera names.
        num_frames: Number of frames in the dataset.
        suffix: Suffix of the json files.
        num_body_keypoints: (Optional) Number of keypoints in the estimated pose [default: 17]
        num_hand_keypoints: (Optional) Number of keypoints per hand in the estimated pose [default:21]
        use_wholebody: (Optional) Whether to read from wholebody model output files [default: False]
        multi_person: (Optional) Whether to support multi-person scenarios [default: False]
        matched_poses_path: (Optional) Path to matched poses JSON file for multi-person mode
    
    Returns:
        For single person (multi_person=False):
            poses_2d_body: A list of 2D poses where each entry is in the format 
                      poses_2d_body[frame][camera][keypoint] -> (x, y, confidence).
            poses_2d_hands: A list of 2D poses where each entry is in the format 
                      poses_2d_hands[frame][camera][keypoint] -> (x, y, confidence).
        
        For multi-person (multi_person=True):
            poses_2d_body: A dict of lists where each key is person_id and value is 
                      poses_2d_body[person_id][frame][camera][keypoint] -> (x, y, confidence).
            poses_2d_hands: A dict of lists where each key is person_id and value is
                      poses_2d_hands[person_id][frame][camera][keypoint] -> (x, y, confidence).
    """
    import json
    # Dictionary to hold the 2D poses for each frame and camera
    frames_data = {"body":{}, "hands": {}}

    if not multi_person:

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
                        print(f"More than one body detected for frame {frame_idx}")

                    try:
                        body_keypoints = body_instances[0]['keypoints']
                        body_confidence = body_instances[0]['keypoint_scores']
                        keypoints_body = [(body_keypoints[i][0], body_keypoints[i][1], body_confidence[i]) for i in range(num_body_keypoints)]
                        
                        frames_data['body'][frame_idx][camera_idx] = keypoints_body

                    except Exception as e:
                        print(f"body keypoints could not be accessed for frame {frame_idx}")
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
                

    if multi_person and matched_poses_path:
        # Load matched poses and organize by person_id
        
        with open(matched_poses_path, 'r') as f:
            matched_poses = json.load(f)
        
        # Initialize data structures for multi-person
        poses_2d_body_multi = {}
        poses_2d_hands_multi = {}
        
        # Find all unique person IDs
        person_ids = set()
        for frame_data in matched_poses:
            for matched_instance in frame_data.get('matched_instances', []):
                person_ids.add(matched_instance['person_id'])
        
        # Initialize data for each person
        for person_id in person_ids:
            poses_2d_body_multi[person_id] = [[[None for _ in range(num_body_keypoints)] for _ in range(len(cam_names))] for _ in range(num_frames)]
            poses_2d_hands_multi[person_id] = [[[None for _ in range(2*num_hand_keypoints)] for _ in range(len(cam_names))] for _ in range(num_frames)]
        
        # Process matched poses
        for frame_idx, frame_data in enumerate(matched_poses):
            if frame_idx >= num_frames:
                break
                
            for matched_instance in frame_data.get('matched_instances', []):
                person_id = matched_instance['person_id']
                
                for instance in matched_instance.get('instances', []):
                    camera_name = instance['camera_name']
                    
                    # Find camera index
                    try:
                        cam_idx = cam_names.index(camera_name)
                    except ValueError:
                        continue
                    
                    # Extract keypoints and confidence scores
                    keypoints = instance.get('keypoints', [])
                    confidence_scores = instance.get('confidence_scores', [])
                    
                    if len(keypoints) >= num_body_keypoints and len(confidence_scores) >= num_body_keypoints:
                        # Process body keypoints
                        body_keypoints = [(keypoints[i][0], keypoints[i][1], confidence_scores[i]) 
                                        for i in range(num_body_keypoints)]
                        poses_2d_body_multi[person_id][frame_idx][cam_idx] = body_keypoints
                        
                        # For hands, if wholebody data includes hand keypoints
                        if len(keypoints) > num_body_keypoints:
                            # Extract hand keypoints (assuming they follow body keypoints)
                            hand_start_idx = num_body_keypoints
                            hand_keypoints = [(keypoints[i][0], keypoints[i][1], confidence_scores[i]) 
                                            for i in range(hand_start_idx, min(len(keypoints), hand_start_idx + 2*num_hand_keypoints))]
                            
                            # Pad with zeros if not enough hand keypoints
                            while len(hand_keypoints) < 2*num_hand_keypoints:
                                hand_keypoints.append((0, 0, 0))
                            
                            poses_2d_hands_multi[person_id][frame_idx][cam_idx] = hand_keypoints
        
        # Fill None values with (0, 0, 0)
        for person_id in person_ids:
            for frame_idx in range(num_frames):
                for cam_idx in range(len(cam_names)):
                    if poses_2d_body_multi[person_id][frame_idx][cam_idx] is None or poses_2d_body_multi[person_id][frame_idx][cam_idx][0] is None:
                        poses_2d_body_multi[person_id][frame_idx][cam_idx] = [(0, 0, 0)] * num_body_keypoints
                    if poses_2d_hands_multi[person_id][frame_idx][cam_idx] is None or poses_2d_hands_multi[person_id][frame_idx][cam_idx][0] is None:
                        poses_2d_hands_multi[person_id][frame_idx][cam_idx] = [(0, 0, 0)] * (2*num_hand_keypoints)
        
        return poses_2d_body_multi, poses_2d_hands_multi
    
    # Single person mode (original behavior)
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

def extract_specific_frame(dataset_file):
    """
    Load poses_2d from the NPZ, detect the only camera, and find the only frame
    where at least ONE JOINT on at least ONE HAND has keypoint_scores == 1.
    Returns both hands from that frame.

    Returns:
        {
          "camera": <camera_name>,
          "frame_index": <int>,
          "hands": [
              {"keypoints": np.ndarray(shape=(21, 2/3)), "keypoint_scores": np.ndarray(shape=(21,))},
              {"keypoints": np.ndarray(shape=(21, 2/3)), "keypoint_scores": np.ndarray(shape=(21,))}
          ]
        }

    Raises:
        ValueError if no such frame exists or if multiple frames match.
    """
    data = np.load(dataset_file, allow_pickle=True)
    poses_2d = data["poses_2d"]
    if isinstance(poses_2d, np.ndarray) and poses_2d.dtype == object and poses_2d.shape == ():
        poses_2d = poses_2d.item()

    # --- detect single camera ---
    cams = list(poses_2d.keys())
    if len(cams) != 1:
        raise ValueError(f"Expected exactly one camera, found {len(cams)}: {cams}")
    cam = cams[0]
    frames = poses_2d[cam]

    def _extract_kp_and_scores(inst):
        """Return (keypoints, keypoint_scores) with leading singleton dim squeezed if present."""
        if isinstance(inst, dict):
            kp = inst.get("keypoints", None)
            sc = inst.get("keypoint_scores", inst.get("scores", None))
        else:
            kp = getattr(inst, "keypoints", None)
            sc = getattr(inst, "keypoint_scores", getattr(inst, "scores", None))

        if kp is None or sc is None:
            return None, None

        kp = np.asarray(kp)
        sc = np.asarray(sc)

        # squeeze leading singleton person-dim if present (e.g., (1,21,2)->(21,2))
        if kp.ndim >= 3 and kp.shape[0] == 1:
            kp = kp[0]
        if sc.ndim >= 2 and sc.shape[0] == 1:
            sc = sc[0]
        return kp, sc

    def _looks_like_instance(obj):
        """Heuristic: does this object look like an instance with pose fields?"""
        if obj is None:
            return False
        # OpenMMLab InstanceData: attributes
        if hasattr(obj, "keypoints") or hasattr(obj, "keypoint_scores"):
            return True
        # Plain dict instance
        if isinstance(obj, dict) and ("keypoints" in obj or "keypoint_scores" in obj):
            return True
        return False
    
    def _iter_instances(entry):
        """Yield instance objects/dicts for a frame entry across common layouts."""
        if entry is None:
            return

        # Case 1: dict with an "instances" list
        if isinstance(entry, dict) and "instances" in entry:
            insts = entry.get("instances", [])
            if isinstance(insts, (list, tuple)):
                for inst in insts:
                    if inst is not None:
                        yield inst
                return

        # Case 2: dict mapping (e.g., {0: InstanceData, 1: InstanceData, ...})
        if isinstance(entry, dict):
            vals = list(entry.values())
            if vals and all(_looks_like_instance(v) for v in vals):
                for inst in vals:
                    if inst is not None:
                        yield inst
                return

        # Case 3: already a list/tuple of instances
        if isinstance(entry, (list, tuple)):
            for inst in entry:
                if inst is not None:
                    yield inst
            return

        # Case 4: object with `.instances` attribute (list-like)
        if hasattr(entry, "instances"):
            try:
                for inst in entry.instances:
                    if inst is not None:
                        yield inst
                return
            except TypeError:
                # .instances isn't iterable; fall through to yield entry
                pass

        # Fallback: assume it's a single instance
        yield entry

    matching = []
    for fi in range(len(frames)):
        entry = frames[fi]
        hands = []
        for inst in _iter_instances(entry):
            kp, sc = _extract_kp_and_scores(inst)
            if kp is None or sc is None:
                continue
            hands.append((kp, sc))
           

        # Expect exactly two hands in the frame
        if len(hands) != 2:
            continue

        # Accept if at least ONE JOINT on at least ONE hand has score == 1
        if any(np.any(sc == 1) for _, sc in hands):
            matching.append((fi, hands))

    if len(matching) == 0:
        raise ValueError("No frame found where any joint on any hand has keypoint_scores == 1.")
    if len(matching) > 1:
        raise ValueError(f"Multiple frames match the criterion (>=1 joint with score 1): {[fi for fi, _ in matching]}")

    frame_idx, hands = matching[0]
    return {
        "camera": cam,
        "frame_index": frame_idx,
        "hands": [
            {"keypoints": hands[0][0], "keypoint_scores": hands[0][1]},
            {"keypoints": hands[1][0], "keypoint_scores": hands[1][1]},
        ],
    }

def _field(obj, name):
    """Get 'keypoints' / 'keypoint_scores' from dict or object."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

def _to_xy(arr):
    """Return array shaped (N, 2) from (N,2) or (1,N,2) or list-like."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[-1] != 2:
        # best effort: flatten pairs
        a = a.reshape(-1, 2)
    return a

def _to_1d(arr):
    """Return 1D float array (N,) from (N,) or (1,N) etc."""
    a = np.asarray(arr).squeeze()
    if a.ndim != 1:
        a = a.reshape(-1)
    return a.astype(float, copy=False)

def per_joint_distances(det_hand, gt_hand, threshold=0.3):
    """
    Euclidean distance per joint between det_hand and gt_hand.
    Places NaN if either score <= threshold at that joint.
    """
    det_xy = _to_xy(_field(det_hand, 'keypoints'))
    det_sc = _to_1d(_field(det_hand, 'keypoint_scores'))

    gt_xy  = _to_xy(_field(gt_hand,  'keypoints'))
    gt_sc  = _to_1d(_field(gt_hand,  'keypoint_scores'))

    n = min(det_xy.shape[0], gt_xy.shape[0], det_sc.size, gt_sc.size)
    det_xy, gt_xy, det_sc, gt_sc = det_xy[:n], gt_xy[:n], det_sc[:n], gt_sc[:n]

    dists = np.linalg.norm(det_xy - gt_xy, axis=1)
    valid = (det_sc > threshold) & (gt_sc > threshold)
    dists[~valid] = np.nan
    return dists  # shape (n,)