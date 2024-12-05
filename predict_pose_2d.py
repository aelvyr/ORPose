from helpers.definitions import *
from helpers.predictors import *
import pickle

from mmpose.apis.inference import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances
from mmpose.registry import VISUALIZERS

import cv2
import mmcv
import numpy as np
import os
import json_tricks as json
import time
from tqdm import tqdm

show = True
draw_bbox = True
kpt_thresh = 0.2
radius = 5
thickness = 3
exclude_box = None #[[610, 80, 795, 185]]

resolution = (1980, 1080)

def select_detector(prompt, error_msg, options):
    in_progress = True
    while in_progress:
        try:
            typed = str(input(prompt))
            if typed in options:
                in_progress = False
            else:
                print(error_msg)
        except:
            print(error_msg)
    return typed

def process_one_image(img,
                      bboxes,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      write_img = None,
                      show=show):
    """Visualize predicted keypoints of one image."""
    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if write_img is None:
        write_img = img
    # show the results
    if isinstance(write_img, str):
        write_img = mmcv.imread(write_img, channel_order='rgb')
    elif isinstance(write_img, np.ndarray):
        write_img = mmcv.bgr2rgb(write_img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            write_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=draw_bbox,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=show,
            wait_time=show_interval,
            kpt_thr=kpt_thresh)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

def main():
    term_size = os.get_terminal_size().columns
    for cam_name in cam_names:
        
        video_name = f'{cam_name}_selected_hands_2'
        print("-"*term_size)
        print(f"Performing full pose estimation on {video_name}")
        print("-"*term_size)

        in_file = f'inputs/{video_name}.mp4'
        intermediate_file = f'vis_dir/body_pose_{video_name}.mp4'
        video_dir = f'inputs/{video_name}'
        output_file = f'vis_dir/{video_name}'
        output_video = f'vis_dir/full_pose_{video_name}.mp4'
        save_path_hands = f'pose_outputs/{video_name}_hands.json'
        save_path_body = f'pose_outputs/{video_name}_body.json'


        curr_obj_id = 0

        print("-"*term_size)
        print("Body Pose Estimation Initiated...")
        print("-"*term_size)

        bboxes_body = detect_body(in_file)

        # build pose estimator
        pose_estimator_body = init_pose_estimator(
            'configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py',
            'checkpoints/body/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth',
            device='cuda',
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=False))))
        
        # build visualizer
        pose_estimator_body.cfg.visualizer.radius = radius
        pose_estimator_body.cfg.visualizer.alpha = 0.8
        pose_estimator_body.cfg.visualizer.line_width = thickness
        visualizer_body = VISUALIZERS.build(pose_estimator_body.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        visualizer_body.set_dataset_meta(
            pose_estimator_body.dataset_meta, skeleton_style='mmpose')

        cap = cv2.VideoCapture(in_file) 
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {in_file}")
        
        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            bbox = np.array(bboxes_body[frame_idx])
            bbox_shape = bbox.shape
            if len(bbox_shape) == 1:
                bbox = bbox.reshape((1, 4))
            
            if bbox_shape[-1] == 4:

                # topdown pose estimation
                pred_instances = process_one_image(frame, bbox, pose_estimator_body, visualizer_body, 0.001, show=False)

                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

                # output videos
                frame_vis = visualizer_body.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        intermediate_file,
                        fourcc,
                        30,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            frame_idx += 1

        if video_writer:
            video_writer.release()

        cap.release()

        with open(save_path_body, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator_body.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {save_path_body}')
            

        print("-"*term_size)
        print("Hand Pose Estimation Initiated...")
        print("-"*term_size)

        bboxes_rtm, frame_idx_rtm = detect_bbox(in_file, show=show)

        bboxes_yolo, frame_idx_yolo = detect_bbox_yolo(in_file, exclude_box, write_folder=True, show=show)

        if frame_idx_rtm is not None and frame_idx_yolo is not None:

            selection = select_detector("Which initial detection performs better? (Enter 1 or 2)", "Please enter a valid response: 1 or 2", ["1", "2"])
            
            if int(selection) == 1:
                bboxes_in = bboxes_rtm
                frame_idx = frame_idx_rtm
            else:
                bboxes_in = bboxes_yolo
                frame_idx = frame_idx_yolo

        elif frame_idx_rtm is not None:
            bboxes_in = bboxes_rtm
            frame_idx = frame_idx_rtm
        else:
            frame_idx = frame_idx_yolo
            bboxes_in = bboxes_yolo

        if frame_idx is None:
            print("No hands detected")

        else:

            inference_state, predictor, frame_names = add_obj_to_sam(video_dir, bboxes_in[0], frame_idx, curr_obj_id, show=False)
            video_segments, bboxes = track_with_sam(video_dir, inference_state, predictor, frame_names, show=False)
            render_bbox_video(output_file, video_dir, bboxes, frame_names)
            bboxes_yolo, frame_idx = detect_bbox_yolo(in_file, exclude_box, bboxes, write_folder=True, show=False)

            if frame_idx is not None:
                curr_obj_id += 1

                inference_state, predictor, frame_names = add_obj_to_sam(video_dir, bboxes_yolo[0], frame_idx, curr_obj_id, show=False)
                prev_bboxes = bboxes
                video_segments, bboxes = track_with_sam(video_dir, inference_state, predictor, frame_names, show=False, prev_bboxes=prev_bboxes)
                render_bbox_video(output_file, video_dir, bboxes, frame_names)

            for k,v in bboxes.items():
                for obj_id in range(2):
                    if obj_id not in v.keys() or len(v[obj_id]) == 0:
                        bboxes[k][obj_id] = np.array([[0, 0, 0, 0]])

            # build pose estimator
            pose_estimator = init_pose_estimator(
                'configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py',
                'checkpoints/hands/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth',
                device='cuda',
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))))

            # build visualizer
            pose_estimator.cfg.visualizer.radius = radius
            pose_estimator.cfg.visualizer.alpha = 0.8
            pose_estimator.cfg.visualizer.line_width = thickness
            visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
            # the dataset_meta is loaded from the checkpoint and
            # then pass to the model in init_pose_estimator
            visualizer.set_dataset_meta(
                pose_estimator.dataset_meta, skeleton_style='mmpose')

            # Load the given frame
            cap = cv2.VideoCapture(in_file) 
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video file: {in_file}")
            
            cap2 = cv2.VideoCapture(intermediate_file)
            if not cap2.isOpened():
                raise FileNotFoundError(f"Cannot open video file: {intermediate_file}")

            video_writer = None
            pred_instances_list = []
            frame_idx = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            while cap.isOpened():
                success, frame = cap.read()
                

                if not success:
                    break

                bbox = np.array(list(bboxes[frame_idx].values()))
                print(bbox)
                bbox_shape = bbox.shape
                
                if bbox_shape[-1] == 4:

                    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, write_frame = cap2.read()
                    if not success:
                        write_frame = None

                    # topdown pose estimation
                    pred_instances = process_one_image(frame, bbox.reshape(bbox_shape[0]*bbox_shape[1], bbox_shape[2]), pose_estimator, visualizer, 0.001, write_frame)

                    # save prediction results
                    pred_instances_list.append(
                        dict(
                            frame_id=frame_idx,
                            instances=split_instances(pred_instances)))

                    # output videos
                    frame_vis = visualizer.get_image()

                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # the size of the image with visualization may vary
                        # depending on the presence of heatmaps
                        video_writer = cv2.VideoWriter(
                            output_video,
                            fourcc,
                            30,  # saved fps
                            (frame_vis.shape[1], frame_vis.shape[0]))

                    video_writer.write(mmcv.rgb2bgr(frame_vis))

                    if show:
                        # press ESC to exit
                        if cv2.waitKey(5) & 0xFF == 27:
                            break

                        time.sleep(0)

                frame_idx += 1

            if video_writer:
                video_writer.release()

            cap.release()

            with open(save_path_hands, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {save_path_hands}')

        print("-"*term_size)
        print(f"Finished Pose Estimation for {in_file}")
        print("-"*term_size)
        


if __name__ == "__main__":
    main()


