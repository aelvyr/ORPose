import os
import cv2
from tqdm import tqdm

import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

from configs.yolo_hands.yolo import YOLO as YOLO_HANDS

import torch
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

import filetype

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


'''      DEFINITIONS       '''

nms_thr = 0.3
bbox_thr = 0.3
body_bbox_thr = 0.4
min_box_thr = 5

yolo_size = 1024
yolo_confidence = 0.85

'''      MODEL INSTANTIATIONS       '''

# build detectors
detector = init_detector(
    'configs/mmdet/cascade_rcnn_x101_64x4d_fpn_1class.py', 
    'checkpoints/hands/detection/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth',
    device='cuda')
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

detector_person = init_detector(
    'configs/mmdet/rtmdet_m_640-8xb32_coco-person.py',
    'checkpoints/body/detection/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
    device='cuda'
)
detector_person.cfg = adapt_mmdet_pipeline(detector_person.cfg)

yolo_detector = YOLO_HANDS("configs/yolo_hands/cross-hands-yolov4-tiny.cfg", "checkpoints/hands/detection/cross-hands-yolov4-tiny.weights", ["hand"])
yolo_detector.size = yolo_size
yolo_detector.confidence = yolo_confidence

checkpoint = "./checkpoints/sam/sam2.1_hiera_large.pt"
model_cfg = "sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda')


'''      FUNCTION DEFINITIONS       '''

def mask_to_boxes(mask, min_box_thr=5):
    """ Convert a boolean (Height x Width) mask into a (N x 4) array of NON-OVERLAPPING bounding boxes
    surrounding "islands of truth" in the mask.  Boxes indicate the (Left, Top, Right, Bottom) bounds
    of each island, with Right and Bottom being NON-INCLUSIVE (ie they point to the indices AFTER the island).

    This algorithm (Downright Boxing) does not necessarily put separate connected components into
    separate boxes.

    You can "cut out" the island-masks with
        boxes = mask_to_boxes(mask)
        island_masks = [mask[t:b, l:r] for l, t, r, b in boxes]
    """
    max_ix = max(s+1 for s in mask.shape)   # Use this to represent background
    # These arrays will be used to carry the "box start" indices down and to the right.
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # Propagate the earliest x-index in each segment to the bottom-right corner of the segment
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


def add_obj_to_sam(video_dir, input_box, frame_idx, obj_id, show=True):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=input_box,
    )

    if show:
        # show the results on the current (interacted) frame
        plt.figure(figsize=(16, 9))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
        show_box(input_box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    return inference_state, predictor, frame_names


def track_with_sam(video_dir, inference_state, predictor, frame_names, show=True, prev_bboxes=None):

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    if prev_bboxes is None:
        bboxes = {}
    else:
        bboxes = prev_bboxes
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if prev_bboxes is None:
            bboxes[out_frame_idx] = {
                out_obj_id: mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy())
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        else:
            for i, out_obj_id in enumerate(out_obj_ids):
                created_boxes = mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy())
                if len(created_boxes) > 0:
                    bboxes[out_frame_idx][out_obj_id] = created_boxes
    
    if len(video_segments) < len(frame_names):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if prev_bboxes is None:
                bboxes[out_frame_idx] = {
                    out_obj_id: mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy())
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            else:
                for i, out_obj_id in enumerate(out_obj_ids):
                    bboxes[out_frame_idx][out_obj_id] = mask_to_boxes((out_mask_logits[i] > 0.0).squeeze().cpu().numpy())

    if show:
        # render the segmentation results every few frames
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
            os.makedirs(os.path.join(parentdir, base_name))
        except:
            write_folder = False

        while cap.isOpened() and (not hands_found or write_folder):
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
            
            
        cap.release()

        return np.array(output), out_frame_idx

def detect_bbox(file, show=False):    
    if filetype.is_image(file):
        print(file)
        det_result = inference_detector(detector, file)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                        pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]

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
            success, frame = cap.read()

            if not success:
                print("Never found hands")
                break

            det_result = inference_detector(detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > bbox_thr)]
            bboxes = bboxes[nms(bboxes, nms_thr), :4]


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

            
                
            
            
        cap.release()

        return bboxes, out_frame_idx
    

def detect_body(file):
    if filetype.is_image(file):
        print(file)
        det_result = inference_detector(detector_person, file)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                        pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]

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
                                            pred_instance.scores > body_bbox_thr)]
            bboxes = bboxes[nms(bboxes, nms_thr), :4]

            if len(bboxes) > 0:
                out_bboxes.append(bboxes)
            else:
                out_bboxes.append([[0, 0, 0, 0]])

            frame_idx += 1

            
        cap.release()

        return out_bboxes


def render_bbox_video(output_file, video_dir, bboxes, frame_names):
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
