import os
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import pickle
import torch

from helpers.predictors import *
from helpers.json_handling import read_keypoints_mmpose
from helpers.camera import *
from helpers.definitions import *
from helpers.plotting import *
from helpers.pose_construction import *
from helpers.hand_transform import *
from helpers.triangulation import *
from helpers.hand_optimization import *
from helpers.video_processing import *
from helpers.hand_detection import *
from helpers.pose_visualization import *
import helpers.definitions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full RocSync pipeline: body + hand reconstruction"
    )
    parser.add_argument(
        "--data_input", "--data-input",
        type=str,
        default="cha/cha1",
        help="Data collection identifier, e.g. 'cha/cha1'. "
             "Used as inputs/{data_input}, pose_outputs/{data_input}, etc."
    )

    # show: default True, can be turned off with --no-show
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Enable visualizations (default).",
    )
    group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable visualizations.",
    )
    parser.set_defaults(show=True)

    # force_retrain: default False, enabled with flag
    parser.add_argument(
        "--force_retrain", "--force-retrain",
        dest="force_retrain",
        action="store_true",
        help="Force recomputation of 3D poses / hand results even if outputs exist.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration from CLI
    data_input = args.data_input
    show = args.show
    FORCE_RETRAIN = args.force_retrain

    intrinsics_suffix = '_synced'  # _synced
    resolution = (960, 540)        # Processing resolution
    original_resolution = (3840, 2160)  # Original video resolution
    max_hands = 2                  # Maximum number of hands to track per person
    reproj_threshold = 7.5

    # Set up paths
    data_dir = f'inputs/{data_input}'              # Input video directory
    pose_output_dir = f'pose_outputs/{data_input}'  # 2D pose output directory
    output_3d_dir = f'output_3d/{data_input}'      # 3D pose output directory
    detection_dir = f'outputs/{data_input}'        # Directory containing detection txt files

    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(output_3d_dir, exist_ok=True)

    lambda_values = {
        'lambda_reproj': 1.0,
        'lambda_temp': 20.0,
        'lambda_shape': 50.0,
        'lambda_wrist': 0.0,
        'lambda_bmc': 5.0
    }

    # Optimization Configs
    use_canonical = True
    outlier_detection = False

    # Camera configuration
    far_field_cameras = []  # Specify which cameras to use
    near_field_cameras = []  # Specify which cameras to use
    for i in range(1, 13):
        if i < 5:
            far_field_cameras.append('gopro' + str(i))
        else:
            near_field_cameras.append('gopro' + str(i))

    # Ensure the correct cameras are set up
    for camera in far_field_cameras + near_field_cameras:
        if camera not in cam_names:
            print(f"Camera {camera} not found in cam_names, adding it.")
            helpers.definitions.cam_names.append(camera)

    for camera in cam_names[:]:  # copy to avoid modifying while iterating
        if camera not in far_field_cameras and camera not in near_field_cameras:
            print(f"Camera {camera} is not assigned to any category, remvoving it.")
            cam_names.remove(camera)

    # Initialize YOLO models for hand detection
    model_person = YOLO('checkpoints/yolo/yolo11l.pt')

    # Initialize models with TAM tracker
    USE_TAM = True
    initialize_models(use_tam=USE_TAM, use_wholebody=True)

    # ---------------------- Body pose extraction ----------------------

    total_frames = None

    # Process all videos
    for video_file in os.listdir(data_dir):
        if video_file.lower().endswith(('.mp4', '.mov')):
            video_name = os.path.splitext(video_file)[0]
            # Check if video is from an allowed camera
            if any(video_name.startswith(cam + '_') for cam in far_field_cameras + near_field_cameras):
                # Process the video
                total_frames = process_single_video_wholebody(
                    video_name,
                    data_dir,
                    pose_output_dir,
                    show=show,
                    vis_output_dir=pose_output_dir
                )
            else:
                print(f"Skipping {video_name} - not in far field cameras list")

    # Load camera parameters
    camera_params_dir = f'camera_poses/{data_input.split("/")[0]}'
    output_file = os.path.join(output_3d_dir, 'body_poses_3d.npz')

    camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices = load_camera_params(
        os.path.join(camera_params_dir, 'camera_intrinsics'),
        os.path.join(camera_params_dir, 'camera_extrinsics/aligned_poses.json'),
        suffix=intrinsics_suffix,
        named=True,
        import_extrinsics_matrix=True,
        camnames=cam_names
    )

    far_field_cam_intr = [camera_intrinsics[cam] for cam in far_field_cameras]
    far_field_cam_extr = [camera_extrinsics[cam] for cam in far_field_cameras]
    far_field_cam_dist = [distortion_coeffs[cam] for cam in far_field_cameras]
    far_field_cam_mat = [camera_matrices[cam] for cam in far_field_cameras]

    if not FORCE_RETRAIN and os.path.exists(os.path.join(output_3d_dir, 'body_poses_3d.npz')):
        print(f"Found existing body pose estimations, loading from {output_file}...")
        pose_file = np.load(output_file)
        poses_2d_body = pose_file['poses_2d']
        poses_3d_body = pose_file['poses_3d']
    else:
        # Read 2D keypoints from JSON files
        poses_2d_body, poses_2d_hands = read_keypoints_mmpose(
            pose_output_dir,
            cam_names,
            num_frames=total_frames,  # Use dynamically determined frame count
            num_body_keypoints=17,    # Number of keypoints for body
            num_hand_keypoints=21,    # Number of keypoints for hands
            suffix='_synced_cut',
            use_wholebody=True
        )

        # Mask keypoints based on confidence
        poses_2d_masked = mask_keypoints(
            poses_2d_body,
            confidence_threshold=0.3,
            min_cameras=2,
            num_keypoints=17
        )

        # Reconstruct 3D poses
        poses_3d_body = reconstruct_3d_pose(
            poses_2d_masked,
            far_field_cam_intr,
            far_field_cam_extr,
            far_field_cam_dist,
            far_field_cam_mat,
            regularization_weight=1.0,
            reprojection_weight=1.0,
            smoothing=True
        )

        # Save results
        np.savez(
            output_file,
            poses_2d=poses_2d_body,
            poses_3d=poses_3d_body
        )

    # Visualize results
    if show and not os.path.exists(os.path.join(output_3d_dir, 'visualization_body.mp4')):
        pose2video(
            poses_3d_body,
            output_3d_dir,
            'visualization_body',
            xlim=(1, -1),
            ylim=(1, -1),
            zlim=(0, 2.5),
            use_wholebody=True
        )
        print(f"Succesfully visualized Body at {os.path.join(output_3d_dir, 'visualization_body.mp4')}")

    # ---------------------- Hand Detection and Tracking --------------------------------

    for video_file in os.listdir(data_dir):
        if video_file.lower().endswith(('.mp4', '.mov')):
            video_name = os.path.splitext(video_file)[0]
            if any(video_name.startswith(cam) for cam in near_field_cameras):
                print(f"\nProcessing video: {video_name}")
                video_path = os.path.join(data_dir, video_file)

                if os.path.exists(f"outputs/{data_input}/{video_name}.txt"):
                    print("Output already exists, skipping")
                    continue

                # Process the video
                bboxes = process_video_wholebody_hands(
                    video_name,
                    data_dir=data_dir,
                    pose_dir=pose_output_dir,
                    save_dir=detection_dir,
                    show=show
                )

                if len(bboxes) == 0:
                    print("No hands detected")
                else:
                    (boxes, conf, frame_idxs) = zip(*bboxes)
                    print(f"\nFound hands at frame {frame_idxs}")
                    print(f"Bounding boxes: \n{boxes}")

                    os.makedirs(f"outputs/{data_input}", exist_ok=True)
                    save_to_file(bboxes, f"outputs/{data_input}/{video_name}.txt")

    # ---------------------- Hand Tracking with efficientTAM --------------------------------

    hand_poses_2d = {}
    output_file = os.path.join(output_3d_dir, 'hand_poses_2d.npz')
    if os.path.exists(output_file):
        print("Hand poses already found, skipping")
        hand_poses_2d = np.load(output_file, allow_pickle=True)['poses_2d'].item()
    else:
        output_dir_temp = os.path.join(output_3d_dir, 'hand_poses_2d_temp')
        os.makedirs(output_dir_temp, exist_ok=True)

        for video_file in sorted(
            os.listdir(data_dir),
            key=lambda x: int(x.split('gopro')[1].split('_')[0]) if 'gopro' in x else 0,
            reverse=True
        ):
            if video_file.lower().endswith(('.mp4', '.mov')):
                video_name = os.path.splitext(video_file)[0]
                if any(video_name.startswith(cam) for cam in near_field_cameras):
                    cam_name = video_name.split('_')[0]
                    print(f"\nProcessing {video_name} for hand detection and tracking...")
                    video_path = os.path.join(data_dir, video_file)

                    tracks = None
                    if os.path.exists(os.path.join(detection_dir, f"tracked_bboxes_{video_name}.pkl")):
                        print(f"Tracks already found for {video_name}")
                        with open(os.path.join(detection_dir, f"tracked_bboxes_{video_name}.pkl"), 'rb') as f:
                            tracks = pickle.load(f)
                    else:
                        # Track hands
                        tracks, frame_names = process_video_wholebody_hands_for_tracking(
                            video_name,
                            data_dir,
                            detection_dir,
                            resolution,
                            original_resolution,
                            max_hands,
                            show=show,
                            add_object_fn=add_object,
                            track_object_fn=track_object
                        )

                        if tracks is not None:
                            # Save tracked boxes
                            with open(os.path.join(detection_dir, f"tracked_bboxes_{video_name}.pkl"), 'wb') as f:
                                tracks = adjust_tracks(tracks, resolution, original_resolution)
                                pickle.dump(tracks, f)

                    if tracks is not None:
                        if os.path.exists(os.path.join(output_dir_temp, cam_name + '.npy')):
                            print(f"Hand poses already processed for {video_name}, skipping")
                            hand_poses_2d[cam_name] = np.load(
                                os.path.join(output_dir_temp, cam_name + '.npy'),
                                allow_pickle=True
                            )
                        else:
                            # Process and visualize poses
                            hand_poses_2d[cam_name] = process_and_visualize_poses(
                                video_path,
                                tracks,
                                pose_output_dir,
                                poses_3d_body,
                                camera_matrices,
                                camera_intrinsics,
                                camera_extrinsics,
                                distortion_coeffs,
                                hide_legs=True,
                                create_label_poses=True,
                            )

                            np.save(
                                os.path.join(output_dir_temp, cam_name),
                                hand_poses_2d[cam_name],
                                allow_pickle=True
                            )
                            print(f"Processed {video_name} for hand poses.")

        # Save hand poses to file
        np.savez(
            output_file,
            poses_2d=hand_poses_2d,
        )

    # ---------------- Hand-ID Matching using Wholebody Pose Estimation ----------------
    if not FORCE_RETRAIN and os.path.exists(os.path.join(output_3d_dir, 'hand_id_mapping.pkl')):
        print("Hand ID mapping already exists, loading...")
        with open(os.path.join(output_3d_dir, 'hand_id_mapping.pkl'), 'rb') as f:
            hand_id_mapping = pickle.load(f)
    else:
        print("Identifying hand IDs...")

        # Prepare video paths dictionary for the identify_hand_ids function
        video_paths = collect_videos(data_dir, hand_poses_2d.keys())

        # Run hand ID identification
        hand_id_mapping = identify_hand_ids(
            hand_poses_2d,
            video_paths,
            camera_intrinsics,
            camera_extrinsics,
            distortion_coeffs,
            poses_3d_body,
            num_frames=total_frames,  # Process first 30 frames (adjust as needed)
            visualize=True,           # Set to True to see visual matches
            pose_output_dir=pose_output_dir
        )

        # Save the hand ID mapping
        with open(os.path.join(output_3d_dir, 'hand_id_mapping.pkl'), 'wb') as f:
            pickle.dump(hand_id_mapping, f)

    # Update the hand tracking and optimization functions to use this mapping
    print("\nHand ID mapping:")
    for cam_name, mapping in hand_id_mapping.items():
        print(f"Camera {cam_name}:")
        for obj_id, hand_side in mapping.items():
            print(f"  Object ID {obj_id} -> {'Left' if hand_side == 0 else 'Right'} hand")

    # --------------------- No-BMC Optimization with hand ID mapping --------------------
    if not FORCE_RETRAIN and os.path.exists(os.path.join(output_3d_dir, 'hand_poses_3d.npz')):
        print("No-BMC optimized hand poses already found with mapping, skipping")
        nobmc_optimized_hand_poses_3d = np.load(
            os.path.join(output_3d_dir, 'hand_poses_3d.npz')
        )['poses_3d']

    else:
        nobmc_optimized_hand_poses_3d = []
        not_initialized = True
        initial_frame_idx = 0
        prev_optimized_hand = None

        while not_initialized and initial_frame_idx < len(poses_3d_body):
            # For first frame, use initial hand poses
            initial_pose = np.concatenate([
                orient_canonical_hand(CANONICAL_HAND_POSE_3D, poses_3d_body[initial_frame_idx], side='left'),
                orient_canonical_hand(CANONICAL_HAND_POSE_3D, poses_3d_body[initial_frame_idx], side='right')
            ])

            # Optimize without BMC but with hand ID mapping
            optimized_hand, loss = optimize_poses_no_bmc(
                initial_pose, hand_poses_2d,
                camera_intrinsics, camera_extrinsics, distortion_coeffs,
                frame_idx=initial_frame_idx, hand_id_mapping=hand_id_mapping,
                lambdas=[lambda_values['lambda_reproj'], 0.0, 0.0, 0.0],
                body_pose=poses_3d_body[initial_frame_idx]
            )
            if loss <= reproj_threshold:
                not_initialized = False
                nobmc_optimized_hand_poses_3d.append(optimized_hand)
                prev_optimized_hand = optimized_hand
                print(f"Found initializing frame: {initial_frame_idx}")
            else:
                initial_frame_idx += 1
                nobmc_optimized_hand_poses_3d.append(None)

        if initial_frame_idx > 0:
            for frame_idx in range(initial_frame_idx - 1, -1, -1):
                if use_canonical or prev_optimized_hand is None:
                    left_hand = orient_canonical_hand(
                        CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='left'
                    )
                    right_hand = orient_canonical_hand(
                        CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='right'
                    )
                else:
                    left_hand = prev_optimized_hand[:num_keypoints_hands]
                    right_hand = prev_optimized_hand[num_keypoints_hands:]

                # Concatenate left and right hand poses
                initial_pose = np.concatenate([left_hand, right_hand])

                # If the previous optimized hand is NaN, use the initial pose
                if (prev_optimized_hand is None) or np.isnan(prev_optimized_hand).any():
                    print(f"Previous optimized hand for frame {frame_idx} is NaN or undefined, using initial pose")
                    prev_optimized_hand = initial_pose

                # Optimize without BMC but with hand ID mapping
                optimized_hand, loss = optimize_poses_no_bmc(
                    initial_pose, hand_poses_2d,
                    camera_intrinsics, camera_extrinsics, distortion_coeffs,
                    frame_idx=frame_idx, previous_pose=prev_optimized_hand,
                    hand_id_mapping=hand_id_mapping,
                    lambdas=[
                        lambda_values['lambda_reproj'],
                        lambda_values['lambda_temp'],
                        lambda_values['lambda_shape'],
                        lambda_values['lambda_wrist']
                    ],
                    body_pose=poses_3d_body[frame_idx]
                )
                nobmc_optimized_hand_poses_3d[frame_idx] = optimized_hand
                prev_optimized_hand = optimized_hand

                print(f"No-BMC optimized hand pose with mapping for frame {frame_idx}: {optimized_hand.shape}")

        for frame_idx in range(initial_frame_idx + 1, len(poses_3d_body)):
            if use_canonical or prev_optimized_hand is None:
                left_hand = orient_canonical_hand(
                    CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='left'
                )
                right_hand = orient_canonical_hand(
                    CANONICAL_HAND_POSE_3D, poses_3d_body[frame_idx], side='right'
                )
            else:
                left_hand = prev_optimized_hand[:num_keypoints_hands]
                right_hand = prev_optimized_hand[num_keypoints_hands:]

            # Concatenate left and right hand poses
            initial_pose = np.concatenate([left_hand, right_hand])

            # If the previous optimized hand is NaN, use the initial pose
            if (prev_optimized_hand is None) or np.isnan(prev_optimized_hand).any():
                print(f"Previous optimized hand for frame {frame_idx} is NaN or undefined, using initial pose")
                prev_optimized_hand = initial_pose

            # Optimize without BMC but with hand ID mapping
            optimized_hand, loss = optimize_poses_no_bmc(
                initial_pose, hand_poses_2d,
                camera_intrinsics, camera_extrinsics, distortion_coeffs,
                frame_idx=frame_idx, previous_pose=prev_optimized_hand,
                hand_id_mapping=hand_id_mapping,
                lambdas=[
                    lambda_values['lambda_reproj'],
                    lambda_values['lambda_temp'],
                    lambda_values['lambda_shape'],
                    lambda_values['lambda_wrist']
                ],
                body_pose=poses_3d_body[frame_idx]
            )
            nobmc_optimized_hand_poses_3d.append(optimized_hand)
            prev_optimized_hand = optimized_hand

            print(f"No-BMC optimized hand pose with mapping for frame {frame_idx}: {optimized_hand.shape}")

        nobmc_optimized_hand_poses_3d = np.array(nobmc_optimized_hand_poses_3d)

        # Save results for No-BMC optimization with mapping
        output_file_nobmc_mapped = os.path.join(output_3d_dir, 'hand_poses_3d.npz')
        np.savez(
            output_file_nobmc_mapped,
            poses_3d=nobmc_optimized_hand_poses_3d,
        )

    if show and not os.path.exists(os.path.join(output_3d_dir, 'visualization_hands.mp4')):
        pose2video(
            poses_3d_body,
            output_3d_dir,
            'visualization_hands',
            nobmc_optimized_hand_poses_3d,
            render_body=False,
            dynamic_limits=True
        )
        print(f"Succesfully visualized Hands at {os.path.join(output_3d_dir, 'visualization_hands.mp4')}")


if __name__ == "__main__":
    main()
