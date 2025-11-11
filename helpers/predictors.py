import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import filetype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from configs.yolo_hands.yolo import YOLO as YOLO_HANDS
from mmpose.apis.inference import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances
from mmpose.registry import VISUALIZERS
import mmcv
from ultralytics import YOLO

# Import SapiensPose for wholebody pose estimation - uses MMPose API
SAPIENS_AVAILABLE = True  # We'll use the MMPose interface with SapiensPose config

# Remove the direct imports of SAM and TAM
# Global variables for models
predictor = None
detector = None
detector_person = None
yolo_detector = None

'''      CONFIGURATION       '''

# Detection thresholds
NMS_THRESHOLD = 0.3
BBOX_THRESHOLD = 0.4
BODY_BBOX_THRESHOLD = 0.4
MIN_BOX_THRESHOLD = 5

# YOLO configuration
YOLO_SIZE = 1024
YOLO_CONFIDENCE = 0.75

# Model paths and configurations
DETECTOR_CONFIG = 'configs/mmdet/cascade_rcnn_x101_64x4d_fpn_1class.py'
DETECTOR_WEIGHTS = 'checkpoints/hands/detection/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'

DETECTOR_PERSON_CONFIG = 'configs/mmdet/rtmdet_m_640-8xb32_coco-person.py'
DETECTOR_PERSON_WEIGHTS = 'checkpoints/body/detection/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

YOLO_CONFIG = "configs/yolo_hands/cross-hands-yolov4-tiny.cfg"
YOLO_WEIGHTS = "checkpoints/hands/detection/cross-hands-yolov4-tiny.weights"

SAM_CHECKPOINT = "./checkpoints/sam/sam2.1_hiera_small.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

EFFICIENTTAM_CHECKPOINT = "./checkpoints/efficienttam/efficienttam_ti_512x512.pt"
EFFICIENTTAM_CONFIG = "configs/efficienttam/efficienttam_ti_512x512.yaml"

# YOLO Rohan model paths
YOLO_PERSON_WEIGHTS = 'checkpoints/yolo/yolo11l.pt'
#YOLO_HAND_WEIGHTS = 'checkpoints/yolo/rohan_pretrained.pt'

# Pose estimation model paths
HAND_POSE_CONFIG = 'configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'
HAND_POSE_WEIGHTS = 'checkpoints/hands/pose/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'

BODY_POSE_CONFIG = 'configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py'
BODY_POSE_WEIGHTS = 'checkpoints/body/pose/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth'

WHOLEBODY_POSE_CONFIG = 'configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
WHOLEBODY_POSE_WEIGHTS = 'checkpoints/wholebody/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth'

# SapiensPose model paths - using 1B parameter model
SAPIENS_POSE_CHECKPOINT = 'checkpoints/sapiens/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth'
SAPIENS_POSE_CONFIG = 'configs/wholebody_2d_keypoint/sapiens/sapiens_1b-210e_coco_wholebody-1024x768.py'

# Global variables for pose estimators
pose_estimator_hand = None
pose_estimator_body = None
visualizer_hand = None 
visualizer_body = None
pose_estimator_wholebody = None
visualizer_wholebody = None
sapiens_pose_estimator = None  # SapiensPose model

# Additional global variables
yolo_person = None
yolo_hand = None

# Visualization settings for pose estimation
POSE_VIS_RADIUS = 3
POSE_VIS_ALPHA = 0.8
POSE_VIS_LINE_WIDTH = 2
POSE_KPT_THRESHOLD = 0.3
IS_HAND_THRESHOLD = 0.2

# Device and compilation configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Check GPU compatibility for torch compile
    gpu_capability = torch.cuda.get_device_capability()
    enable_torch_compile = gpu_capability[0] >= 7  # Only enable for Volta (7.0) and newer GPUs
else:
    device = torch.device("cpu")
    enable_torch_compile = False

print(f"Device: {device}")
print(f"Torch compile enabled: {enable_torch_compile}")

# GPU optimization settings
if device.type == "cuda":

    # Enable TF32 on Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


'''      MODEL INSTANTIATIONS       '''

def initialize_models(use_tam: bool, use_wholebody:bool=False, use_sapiens:bool=False):
    """Initialize all models based on the tracker choice.
    
    Args:
        use_tam (bool): If True, use EfficientTAM tracker. If False, use SAM tracker.
        use_wholebody (bool): If True, initialize wholebody pose estimator.
        use_sapiens (bool): If True, use SapiensPose instead of default wholebody model.
    """
    global predictor, detector, detector_person, yolo_detector
    global pose_estimator_hand, pose_estimator_body, visualizer_hand, visualizer_body
    global yolo_person, yolo_hand, sapiens_pose_estimator
    if use_wholebody:
        global pose_estimator_wholebody, visualizer_wholebody    
    # Initialize detectors
    detector = init_detector(DETECTOR_CONFIG, DETECTOR_WEIGHTS, device='cuda')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    detector_person = init_detector(DETECTOR_PERSON_CONFIG, DETECTOR_PERSON_WEIGHTS, device='cuda')
    detector_person.cfg = adapt_mmdet_pipeline(detector_person.cfg)

    yolo_detector = YOLO_HANDS(YOLO_CONFIG, YOLO_WEIGHTS, ["hand"])
    yolo_detector.size = YOLO_SIZE
    yolo_detector.confidence = YOLO_CONFIDENCE

    # Initialize only the selected tracker
    if use_tam:
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        predictor = build_efficienttam_video_predictor(
            EFFICIENTTAM_CONFIG, 
            EFFICIENTTAM_CHECKPOINT, 
            device=device
        )
    else:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(
            SAM_CONFIG, 
            SAM_CHECKPOINT, 
            device=device
        )
    
    # Initialize pose estimators
    pose_estimator_hand = init_pose_estimator(
        HAND_POSE_CONFIG,
        HAND_POSE_WEIGHTS,
        device='cuda',
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    pose_estimator_body = init_pose_estimator(
        BODY_POSE_CONFIG,
        BODY_POSE_WEIGHTS,
        device='cuda',
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    if use_wholebody:   
        if use_sapiens and SAPIENS_AVAILABLE:
            # Initialize SapiensPose for wholebody pose estimation
            try:
                print("Initializing SapiensPose 1B model...")
                sapiens_pose_estimator = init_pose_estimator(
                    SAPIENS_POSE_CONFIG,
                    SAPIENS_POSE_CHECKPOINT,
                    device='cuda',
                    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
                )
                print("SapiensPose model initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize SapiensPose: {e}")
                sapiens_pose_estimator = None
        else:
            # Fall back to RTMPose wholebody model
            pose_estimator_wholebody = init_pose_estimator(
                WHOLEBODY_POSE_CONFIG,
                WHOLEBODY_POSE_WEIGHTS,
                device='cuda',
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
            print("RTMPose wholebody model initialized")
            sapiens_pose_estimator = None

    # Configure hand pose visualizer
    pose_estimator_hand.cfg.visualizer.radius = POSE_VIS_RADIUS
    pose_estimator_hand.cfg.visualizer.alpha = POSE_VIS_ALPHA
    pose_estimator_hand.cfg.visualizer.line_width = POSE_VIS_LINE_WIDTH
    visualizer_hand = VISUALIZERS.build(pose_estimator_hand.cfg.visualizer)
    visualizer_hand.set_dataset_meta(pose_estimator_hand.dataset_meta, skeleton_style='mmpose')

    # Configure body pose visualizer
    pose_estimator_body.cfg.visualizer.radius = POSE_VIS_RADIUS
    pose_estimator_body.cfg.visualizer.alpha = POSE_VIS_ALPHA
    pose_estimator_body.cfg.visualizer.line_width = POSE_VIS_LINE_WIDTH
    visualizer_body = VISUALIZERS.build(pose_estimator_body.cfg.visualizer)
    visualizer_body.set_dataset_meta(pose_estimator_body.dataset_meta, skeleton_style='mmpose')
    
    # Configure wholebody pose visualizer (only for RTMPose, not SapiensPose)
    if use_wholebody and not (use_sapiens and SAPIENS_AVAILABLE):
        pose_estimator_wholebody.cfg.visualizer.radius = POSE_VIS_RADIUS
        pose_estimator_wholebody.cfg.visualizer.alpha = POSE_VIS_ALPHA
        pose_estimator_wholebody.cfg.visualizer.line_width = POSE_VIS_LINE_WIDTH
        visualizer_wholebody = VISUALIZERS.build(pose_estimator_wholebody.cfg.visualizer)
        visualizer_wholebody.set_dataset_meta(pose_estimator_wholebody.dataset_meta, skeleton_style='mmpose')

    # Initialize YOLO models
    yolo_person = YOLO(YOLO_PERSON_WEIGHTS)
    # yolo_hand = YOLO(YOLO_HAND_WEIGHTS)
    
    # # Configure YOLO hand model settings
    # yolo_hand.conf = 0.5  # NMS confidence threshold
    # yolo_hand.iou = 0.5   # NMS IoU threshold
    
    print(f"Models initialized with {'EfficientTAM' if use_tam else 'SAM'} tracker")
    print("Pose estimation models initialized")
    if use_wholebody and use_sapiens and SAPIENS_AVAILABLE:
        print("SapiensPose 1B wholebody model ready")
    elif use_wholebody:
        print("RTMPose wholebody model ready")
    print("YOLO models initialized")

def check_models_initialized():
    """Check if models have been initialized."""
    if predictor is None:
        raise RuntimeError("Models not initialized. Call initialize_models(use_tam: bool) first.")

'''      FUNCTION DEFINITIONS       '''

def mask_to_boxes(mask, min_box_thr=5, whole_mask=False):
    """Convert a boolean mask into bounding box(es).
    
    Args:
        mask: Boolean mask array
        min_box_thr: Minimum box size threshold
        whole_mask: If True, return single bbox around entire mask.
                   If False, return boxes for each disconnected component.
    
    Returns:
        np.ndarray of bounding boxes
    """
    if whole_mask:
        # Find coordinates of True values
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return np.array([])
            
        # Get min/max coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Create single bounding box
        bbox = np.array([[x_min, y_min, x_max + 1, y_max + 1]])
        
        # Apply minimum size threshold
        if ((bbox[0,2] - bbox[0,0]) > min_box_thr and 
            (bbox[0,3] - bbox[0,1]) > min_box_thr):
            return bbox
        return np.array([])
        
    # Original disconnected components logic
    max_ix = max(s+1 for s in mask.shape)
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # These arrays will be used to carry the "box start" indices down and to the right.
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i-1, j] if i>0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    # Propagate the earliest y-index in each segment to the bottom-right corner of the segment
    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j-1] if j>0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    # Find the bottom-right corners of each segment
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False)==-1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False)==-1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    # Extract the boxes, getting the top-right corners from the index arrays
    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None]+1, y_stops[:, None]+1])
    outboxes = []
    for box in ltrb_boxes:
        if ((box[2] - box[0]) > min_box_thr) and ((box[3] - box[1]) > min_box_thr):
            outboxes.append(box)
    return np.array(outboxes)


def show_mask(mask, ax, obj_id=None, random_color=False):
    print(obj_id)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_autocast_settings(model_name):
    """Define autocast settings per model"""
    settings = {
        'sam': {'enabled': True, 'dtype': torch.bfloat16},
        'tam': {'enabled': True, 'dtype': torch.bfloat16},
        'yolo': {'enabled': False, 'dtype': None},  # YOLO might not support bf16
        'pose': {'enabled': True, 'dtype': torch.float16}  # Pose estimation often works better with fp16
    }
    return settings.get(model_name, {'enabled': False, 'dtype': None})

# First, modify the decorator function to work with or without arguments
def with_autocast(arg):
    """Decorator to handle autocast context
    Can be used as @with_autocast or @with_autocast('model_name')
    """
    if callable(arg):  # Used as @with_autocast without parameters
        func = arg
        model_name = 'default'
        def wrapper(*args, **kwargs):
            settings = get_autocast_settings(model_name)
            if device.type == "cuda" and settings['enabled']:
                if settings['dtype'] == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    settings['dtype'] = torch.float16
                with torch.autocast("cuda", dtype=settings['dtype']):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    else:  # Used as @with_autocast('model_name')
        model_name = arg
        def decorator(func):
            def wrapper(*args, **kwargs):
                settings = get_autocast_settings(model_name)
                if device.type == "cuda" and settings['enabled']:
                    if settings['dtype'] == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                        settings['dtype'] = torch.float16
                    with torch.autocast("cuda", dtype=settings['dtype']):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Then use it in either way:
@with_autocast('tam')
def add_object(video_dir, input_box=None, point_coords=None, point_labels=None, frame_idx=0, obj_id=0, show=True, inference_state=None, predictor_in=None):
    """Add a new object to track using either box or points for both SAM and TAM.
    
    If both box and points are provided, box takes precedence.
    If neither is provided, raises ValueError.
    
    Args:
        video_dir: Directory containing video frames
        input_box: Bounding box coordinates [x1,y1,x2,y2]
        point_coords: Point coordinates for prompting
        point_labels: Labels for points
        frame_idx: Index of frame to add object
        obj_id: ID to assign to new object
        show: Whether to display visualization
        inference_state: Optional existing inference state
        predictor_in: Optional existing predictor
    """
    check_models_initialized()
    
    # Get frame names
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Use existing predictor/state if provided, otherwise initialize new ones
    predictor_use = predictor_in if predictor_in is not None else predictor
    if inference_state is None:
        inference_state = predictor_use.init_state(video_path=video_dir)
        predictor_use.reset_state(inference_state)

    # If box is provided, use it regardless of tracker type
    if input_box is not None:
        _, out_obj_ids, out_mask_logits = predictor_use.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=input_box
        )
        vis_type = 'box'
    # Otherwise use points if provided
    elif point_coords is not None and point_labels is not None:
        _, out_obj_ids, out_mask_logits = predictor_use.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_coords,
            labels=point_labels
        )
        vis_type = 'points'
    else:
        raise ValueError("Either bounding box or points (with labels) must be provided")
    
    if show:
        plt.figure(figsize=(16, 9))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
        if vis_type == 'points':
            show_points(point_coords, point_labels, plt.gca())
        else:
            show_box(input_box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    return inference_state, predictor_use, frame_names

# Then use it in either way:
@with_autocast('tam')
def track_object(video_dir, inference_state, predictor, frame_names, show=True, prev_bboxes=None, whole_mask=False):
    """Track object using the selected tracker (SAM or TAM)
    
    Args:
        video_dir: Directory containing video frames
        inference_state: Tracker inference state
        predictor: Tracker predictor instance
        frame_names: List of frame filenames
        show: Whether to show visualization
        prev_bboxes: Optional previous bounding boxes
        whole_mask: If True, create single bbox around entire mask
    """
    check_models_initialized()
    
    video_segments = {}
    if prev_bboxes is None:
        bboxes = {}
    else:
        bboxes = prev_bboxes

    # Forward propagation
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if prev_bboxes is None:
            bboxes[out_frame_idx] = {
                out_obj_id: mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy(), 
                                        whole_mask=whole_mask)
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        else:
            for i, out_obj_id in enumerate(out_obj_ids):
                created_boxes = mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy(), 
                                        whole_mask=whole_mask)
                if len(created_boxes) > 0:
                    bboxes[out_frame_idx][out_obj_id] = created_boxes

    # Backward propagation if needed
    if len(video_segments) < len(frame_names):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if prev_bboxes is None:
                bboxes[out_frame_idx] = {
                    out_obj_id: mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy(), 
                                            whole_mask=whole_mask)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            else:
                for i, out_obj_id in enumerate(out_obj_ids):
                    created_boxes = mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy(), 
                                            whole_mask=whole_mask)
                    if len(created_boxes) > 0:
                        bboxes[out_frame_idx][out_obj_id] = created_boxes

    if show:
        # Visualization code
        vis_frame_stride = 30
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(16, 9))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.close("all")

    return video_segments, bboxes

    


def detect_bbox_yolo(file: str, exclude_box = None, prev_bboxes=None, write_folder=False, show=True):
    """Detect hands in an image or video using YOLO model.

    Args:
        file (str): Path to image or video file
        exclude_box (list, optional): Boxes to exclude from detection. Defaults to None.
        prev_bboxes (dict, optional): Previous bounding boxes. Defaults to None.
        write_folder (bool, optional): Whether to save frames. Defaults to False.
        show (bool, optional): Whether to show visualization. Defaults to True.

    Returns:
        For images: np.ndarray of bounding boxes
        For videos: tuple of (np.ndarray of boxes, frame index)
    """
    check_models_initialized()
    
    if filetype.is_image(file):
        print(file)
        mat = cv2.imread(file)

        width, height, inference_time, results = yolo_detector.inference(mat)

        print("%s in %s seconds: %s classes found!" %
            (os.path.basename(file), round(inference_time, 2), len(results)))

        output = []

        if show:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1920, 1080)

        for detection in results:
            id, name, confidence, x, y, w, h = detection

            if show:
                # draw a bounding box rectangle and label on the image
                color = (255, 0, 255)
                cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
                text = "%s (%s)" % (name, round(confidence, 2))
                cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, color, 1)

            print("%s with %s confidence" % (name, round(confidence, 2)))

            # cv2.imwrite("export.jpg", mat)
            output.append([x,y,x+w,y+h])

        if show:
            # show the output image
            cv2.imshow('image', mat)
            cv2.waitKey(0)
        return np.array(output)
    elif filetype.is_video(file):
        print(file)
        cap = cv2.VideoCapture(file)  # Replace with your image path
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {file}")
        
        hands_found = False
        frame_idx = 0
        output = []
        out_frame_idx = None

        parentdir = os.path.abspath(os.path.join(file, os.pardir))
        base_name = os.path.splitext(os.path.basename(file))[0]
        try:
            if write_folder:
                os.makedirs(os.path.join(parentdir, base_name))
        except:
            write_folder = False

        while cap.isOpened() and (not hands_found or write_folder):

            print(frame_idx, end='\r')
            success, frame = cap.read()

            if not success:
                print("Never found hands")
                break

            if write_folder:
                # Only extract frames at the desired frame rate
                output_file = os.path.join(parentdir, base_name, f"{frame_idx:05d}.jpg")
                cv2.imwrite(output_file, frame)
                print(f"Frame {frame_idx} has been extracted and saved as {output_file}")
            

            if not hands_found:
                width, height, inference_time, results = yolo_detector.inference(frame)

                if exclude_box is not None:
                    for box in exclude_box:
                        for result in results:
                            id, name, confidence, x, y, w, h = result
                            cx = x + (w / 2)
                            cy = y + (h / 2)
                            if cx >= box[0] and cx <= box[2] and cy >= box[1] and cy <= box[3]:
                                results.remove(result)

                if prev_bboxes is not None:
                    exclude_box_i = prev_bboxes[frame_idx]
                    for obj, boxes in exclude_box_i.items():
                        for box in boxes:
                            for result in results:
                                id, name, confidence, x, y, w, h = result
                                cx = x + (w / 2)
                                cy = y + (h / 2)
                                if cx >= box[0] and cx <= box[2] and cy >= box[1] and cy <= box[3]:
                                    results.remove(result)


                if len(results) > 0:
                    hands_found = True
                    out_frame_idx = frame_idx

                    if show:
                        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('image', 1920, 1080)

                    for detection in results:
                        id, name, confidence, x, y, w, h = detection

                        if show:
                            # draw a bounding box rectangle and label on the image
                            color = (255, 0, 255)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                            text = "%s (%s)" % (name, round(confidence, 2))
                            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.25, color, 1)

                        print("%s with %s confidence" % (name, round(confidence, 2)))

                        # cv2.imwrite("export.jpg", mat)
                        output.append([x,y,x+w,y+h])

                    if show:
                        # show the output image
                        cv2.imshow('image', frame)
                        cv2.waitKey(0)

            frame_idx += 1
        
        print("End of video")
            
            
        cap.release()

        return np.array(output), out_frame_idx

def detect_bbox(file, show=False):    
    """Detect hands using Cascade R-CNN model.

    Args:
        file (str): Path to image or video file
        show (bool, optional): Whether to show visualization. Defaults to False.

    Returns:
        For images: np.ndarray of bounding boxes 
        For videos: tuple of (bounding boxes, frame index)
    """
    check_models_initialized()
    
    if filetype.is_image(file):
        print(file)
        det_result = inference_detector(detector, file)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                        pred_instance.scores > BBOX_THRESHOLD)]
        bboxes = bboxes[nms(bboxes, NMS_THRESHOLD), :4]

        if show:
            mat = cv2.imread(file)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1920, 1080)
            for bbox in bboxes:
                print(bbox)
                # draw a bounding box rectangle and label on the image
                color = (255, 0, 255)
                cv2.rectangle(mat, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
            # show the output image
            cv2.imshow('image', mat)
            cv2.waitKey(0)
            
        return bboxes
    elif filetype.is_video(file):
        print(file)
        cap = cv2.VideoCapture(file)  # Replace with your image path
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {file}")
        
        hands_found = False
        frame_idx = 0
        out_frame_idx = None
        bboxes = []

        while cap.isOpened() and (not hands_found):

            print(frame_idx, end='\r')
            success, frame = cap.read()

            if not success:
                print("Never found hands")
                break

            det_result = inference_detector(detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > BBOX_THRESHOLD)]
            bboxes = bboxes[nms(bboxes, NMS_THRESHOLD), :4]


            if len(bboxes) > 0:
                print("Hand bbox detected")
                hands_found = True
                out_frame_idx = frame_idx

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 1920, 1080)
                for bbox in bboxes:
                    print(bbox)
                    # draw a bounding box rectangle and label on the image
                    color = (255, 0, 255)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                # show the output image
                cv2.imshow('image', frame)
                cv2.waitKey(0)

            frame_idx += 1

                
        print("End of video")
            
        cap.release()

        return bboxes, out_frame_idx
    

def detect_body(file):
    """Detect person bounding boxes using RTMDet model.

    Args:
        file (str): Path to image or video file

    Returns:
        For images: np.ndarray of bounding boxes
        For videos: list of bounding boxes per frame
    """
    check_models_initialized()
    
    if filetype.is_image(file):
        det_result = inference_detector(detector_person, file)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                        pred_instance.scores > BBOX_THRESHOLD)]
        bboxes = bboxes[nms(bboxes, NMS_THRESHOLD), :4]

        return bboxes
    elif filetype.is_video(file):
        print(file)
        cap = cv2.VideoCapture(file)  # Replace with your image path
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {file}")
        
        frame_idx = 0
        out_frame_idx = None
        out_bboxes = []

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            det_result = inference_detector(detector_person, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > BODY_BBOX_THRESHOLD)]
            bboxes = bboxes[nms(bboxes, NMS_THRESHOLD), :4]

            if len(bboxes) > 0:
                out_bboxes.append(bboxes)
            else:
                out_bboxes.append([[0, 0, 0, 0]])

            frame_idx += 1

            
        cap.release()

        return out_bboxes


def render_bbox_video(output_file, video_dir, bboxes, frame_names):
    """Render video with bounding box visualizations.

    Args:
        output_file (str): Output directory path
        video_dir (str): Directory containing video frames
        bboxes (dict): Bounding boxes per frame
        frame_names (list): List of frame filenames
    """
    # render the segmentation results every few frames
    plt.close("all")

    os.makedirs(output_file, exist_ok=True)

    for out_frame_idx in tqdm(range(len(frame_names))):
        plt.figure(figsize=(16, 9))
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        for out_obj_id, out_box in bboxes[out_frame_idx].items():
            for box in out_box:
                show_box(box, plt.gca())        
            
        plt.savefig(os.path.join(output_file, f'{out_frame_idx:05d}.jpg'))
        plt.close()


    plt.close("all")

    video_name = output_file+'.mp4'

    images = [img for img in os.listdir(output_file) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(output_file, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(output_file, image)))

    cv2.destroyAllWindows()
    video.release()

'''      POSE ESTIMATION FUNCTIONS       '''

@with_autocast('pose')
def estimate_pose(img, bbox, pose_type='hand', show=False, write_img=None):
    """Estimate pose for a given bbox in an image.
    
    Args:
        img: Input image (path or numpy array)
        bbox: Bounding box coordinates [[x1,y1,x2,y2]]
        pose_type: Type of pose to estimate ('hand', 'body', or 'sapiens')
        show: Whether to show visualization
        write_img: Optional image to draw visualization on
    
    Returns:
        For 'hand' or 'body': Predicted pose instances
        For 'sapiens': Tuple of (body_instances, left_hand_instances, right_hand_instances)
    """
    check_models_initialized()

    # Ensure bbox is in correct format (N,4)
    if isinstance(bbox, np.ndarray):
        bbox = bbox.reshape(1,-1) if bbox.size == 4 else bbox

    # Select appropriate models
    if pose_type == 'hand':
        # Predict keypoints
        pose_results = inference_topdown(pose_estimator_hand, img, bbox)
        data_samples = merge_data_samples(pose_results)

        # Visualize if requested
        if visualizer_hand is not None and show:
            visualizer_hand.set_dataset_meta(pose_estimator_hand.dataset_meta, skeleton_style='mmpose')
            if write_img is None:
                write_img = img
            if isinstance(write_img, str):
                write_img = mmcv.imread(write_img, channel_order='rgb')
            elif isinstance(write_img, np.ndarray):
                write_img = mmcv.bgr2rgb(write_img)

            visualizer_hand.add_datasample(
                'result',
                write_img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show_kpt_idx=False,
                skeleton_style='mmpose',
                show=show,
                wait_time=0.1,
                kpt_thr=POSE_KPT_THRESHOLD)

        return data_samples.get('pred_instances', None)

    elif pose_type == 'body':
        # Predict keypoints
        pose_results = inference_topdown(pose_estimator_body, img, bbox)
        data_samples = merge_data_samples(pose_results)

        # Visualize if requested
        if visualizer_body is not None and show:
            if write_img is None:
                write_img = img
            if isinstance(write_img, str):
                write_img = mmcv.imread(write_img, channel_order='rgb')
            elif isinstance(write_img, np.ndarray):
                write_img = mmcv.bgr2rgb(write_img)

            visualizer_body.add_datasample(
                'result',
                write_img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show_kpt_idx=False,
                skeleton_style='mmpose',
                show=show,
                wait_time=0.1,
                kpt_thr=POSE_KPT_THRESHOLD)

        return data_samples.get('pred_instances', None)
    
    elif pose_type == 'wholebody':
        # Check if SapiensPose is available and initialized
        if sapiens_pose_estimator is not None:
            # Use SapiensPose for wholebody estimation
            return estimate_pose_sapiens(img, bbox, show, write_img)
        else:
            # Fall back to RTMPose wholebody model
            pose_results = inference_topdown(pose_estimator_wholebody, img, bbox)
            data_samples = merge_data_samples(pose_results)
            
        
            # Visualize if requested
            if visualizer_wholebody is not None and show:
                if write_img is None:
                    write_img = img
                if isinstance(write_img, str):
                    write_img = mmcv.imread(write_img, channel_order='rgb')
                elif isinstance(write_img, np.ndarray):
                    write_img = mmcv.bgr2rgb(write_img)

                visualizer_wholebody.add_datasample(
                    'result',
                    write_img,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show_kpt_idx=False,
                    skeleton_style='mmpose',
                    show=show,
                    wait_time=0.1,
                    kpt_thr=POSE_KPT_THRESHOLD)
            
            # Get the predicted instances from the data sample
            pred_instances = data_samples.get('pred_instances', None)
            
            if pred_instances is None:
                return None, None, None
            
            # Extract body, left hand, and right hand keypoints
            # Based on the COCO-WholeBody dataset format used by SAPIENS
            # Body keypoints are the first 17 points (0-16)
            # Left hand starts at index 91 and has 21 keypoints (91-111)
            # Right hand starts at index 112 and has 21 keypoints (112-132)
            
            # Create a copy of the instances to avoid modifying the original
            body_instances = pred_instances.clone()
            left_hand_instances = pred_instances.clone()
            right_hand_instances = pred_instances.clone()

            # Extract body keypoints (first 17 keypoints)
            body_keypoints = pred_instances.keypoints[:, :17, :]
            body_instances.keypoints = body_keypoints
            body_instances.keypoint_scores = pred_instances.keypoint_scores[:, :17]
            
            # Extract left hand keypoints (indices 91-111)
            if pred_instances.keypoints.shape[1] > 111:
                left_hand_keypoints = pred_instances.keypoints[:, 91:112, :]
                left_hand_instances.keypoints = left_hand_keypoints
                left_hand_instances.keypoint_scores = pred_instances.keypoint_scores[:, 91:112]
            else:
                left_hand_instances = None
                
            # Extract right hand keypoints (indices 112-132)
            if pred_instances.keypoints.shape[1] > 132:
                right_hand_keypoints = pred_instances.keypoints[:, 112:133, :]
                right_hand_instances.keypoints = right_hand_keypoints
                right_hand_instances.keypoint_scores = pred_instances.keypoint_scores[:, 112:133]       
            else:
                right_hand_instances = None
                
            return body_instances, left_hand_instances, right_hand_instances
        
    else:
        raise ValueError(f"Invalid pose_type: {pose_type}. Must be 'hand', 'body', or 'wholebody'.")

def estimate_pose_sapiens(img, bbox, show=False, write_img=None):
    """Estimate wholebody pose using SapiensPose model.
    
    Args:
        img: Input image (path or numpy array)
        bbox: Bounding box coordinates [[x1,y1,x2,y2]]
        show: Whether to show visualization
        write_img: Optional image to draw visualization on
        
    Returns:
        Tuple of (body_instances, left_hand_instances, right_hand_instances) 
        Each containing keypoints and confidence scores
    """
    if sapiens_pose_estimator is None:
        raise RuntimeError("SapiensPose model not initialized. Call initialize_models with use_sapiens=True.")
    
    # Ensure bbox is in correct format (N,4)
    if isinstance(bbox, np.ndarray):
        bbox = bbox.reshape(1, -1) if bbox.size == 4 else bbox
    elif isinstance(bbox, list) and len(bbox) > 0:
        # Convert list to numpy array
        bbox = np.array(bbox)
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)

    # Predict keypoints using SAPIENS wholebody model
    with torch.inference_mode():
        pose_results = inference_topdown(sapiens_pose_estimator, img, bbox)
    
    data_samples = merge_data_samples(pose_results)
    
    # Get the predicted instances from the data sample
    pred_instances = data_samples.get('pred_instances', None)
    
    if pred_instances is None:
        return None, None, None
    
    # Extract body, left hand, and right hand keypoints
    # Based on the COCO-WholeBody dataset format used by SAPIENS
    # Body keypoints are the first 17 points (0-16)
    # Left hand starts at index 91 and has 21 keypoints (91-111)
    # Right hand starts at index 112 and has 21 keypoints (112-132)
    
    # Create a copy of the instances to avoid modifying the original
    body_instances = pred_instances.clone()
    left_hand_instances = pred_instances.clone()
    right_hand_instances = pred_instances.clone()
    
    # Extract body keypoints (first 17 keypoints)
    if pred_instances.keypoints.shape[1] > 17:
        body_keypoints = pred_instances.keypoints[:, :17, :]
        body_keypoint_scores = pred_instances.keypoint_scores[:, :17]
        body_instances.keypoints = body_keypoints
        body_instances.keypoint_scores = body_keypoint_scores
    else:
        body_instances = None
        
    # Extract left hand keypoints (indices 91-111)
    if pred_instances.keypoints.shape[1] > 111:
        left_hand_keypoints = pred_instances.keypoints[:, 91:112, :]
        left_hand_keypoint_scores = pred_instances.keypoint_scores[:, 91:112]
        left_hand_instances.keypoints = left_hand_keypoints
        left_hand_instances.keypoint_scores = left_hand_keypoint_scores
    else:
        left_hand_instances = None
        
    # Extract right hand keypoints (indices 112-132)
    if pred_instances.keypoints.shape[1] > 132:
        right_hand_keypoints = pred_instances.keypoints[:, 112:133, :]
        right_hand_keypoint_scores = pred_instances.keypoint_scores[:, 112:133]
        right_hand_instances.keypoints = right_hand_keypoints
        right_hand_instances.keypoint_scores = right_hand_keypoint_scores
    else:
        right_hand_instances = None
        
    return body_instances, left_hand_instances, right_hand_instances

'''      NEW ROHAN DETECTION FUNCTIONS       '''

def crop_img(img, box):
    """Crop image based on bounding box"""
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]

@with_autocast('yolo')
def detect_hands_in_frame(frame):
    """Detect hands in a single frame using two-stage YOLO detection.
    
    First detects person, then looks for hands within person bbox.
    
    Args:
        frame: Input frame/image
        
    Returns:
        boxes: Array of hand bounding boxes
        confidence: Array of confidence scores
    """
    check_models_initialized()
    
    # Process frame with person detector
    results = yolo_person(frame, classes=0)
    
    boxes = []
    confidence = []
    
    # Process each person detection
    for r in results:
        boxes_tensor = r.boxes.xyxy.cpu()
        confs = r.boxes.conf.cpu()
        
        for box1, conf in zip(boxes_tensor, confs):
            # Crop person region and detect hands
            cropped = crop_img(frame, box1)
            results_hands = yolo_hand(cropped)
            
            for r in results_hands:
                boxes_tensor = r.boxes.xyxy.cpu()
                confs = r.boxes.conf.cpu()
                
                for box2, conf in zip(boxes_tensor, confs):
                    if conf > yolo_hand.conf:
                        # Adjust hand bbox coordinates relative to full frame
                        adjusted_box = np.add(np.array(box2).reshape(2, 2), box1[:2])
                        boxes.append(adjusted_box.flatten())
                        confidence.append(conf)
                        
    return np.array(boxes) if boxes else np.array([]), np.array(confidence) if confidence else np.array([])

def process_video(input_file, write_folder=True, show=True):
    """Process video file for hand detection.
    
    Args:
        input_file: Path to video file
        write_folder: Whether to save extracted frames
        show: Whether to show visualization
    
    Returns:
        List of tuples (boxes, frame_idx) containing detections
    """
    check_models_initialized()
    
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_file}")
    
    frame_idx = 0
    output_boxes = []
    out_frame_idx = None
    
    # Setup output directory for frames
    if write_folder:
        parentdir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        frames_dir = os.path.join(parentdir, base_name)
        os.makedirs(frames_dir, exist_ok=True)
    
    success = True
    while cap.isOpened() and success:   
        success, frame = cap.read()
        if not success:
            break
            
        # Save frame if requested
        if write_folder:
            output_file = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(output_file, frame)
            
        # Detect hands
        boxes, confidence = detect_hands_in_frame(frame)
        
        if len(boxes) > 0:
            output_boxes.append((boxes, frame_idx))
            
        if show:
            # Visualize detections
            for box, conf in zip(boxes, confidence):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Hand ({conf:.1%})", (x1 - 30, y1 - 30), 
                          cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detections', 1920, 1080)
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) == 113:  # 'q' key
                success = False
                
        frame_idx += 1
        print(f"Processing frame {frame_idx}", end='\r')
    
    cap.release()
    if show:
        cv2.destroyAllWindows()
        
    return output_boxes

def save_detections(boxes, output_file):
    """Save detection results to file.
    
    Args:
        boxes: List of (boxes, frame_idx) tuples
        output_file: Path to output file
    """
    with open(output_file, 'w+') as f:
        for boxes, frame_idx in boxes:
            f.write(f"{frame_idx} ")
            for box in boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} ")
            f.write("\n")