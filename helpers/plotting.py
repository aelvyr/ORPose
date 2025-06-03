import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helpers.definitions import *

def draw_keypoints_and_lines(frame, keypoints, color_circles=(0, 255, 0), color_lines=(255, 0, 0), hide_legs=False, wholebody=False):
    """
    Draw keypoints and lines connecting them on the frame. Each keypoint is a pair of (x, y) coordinates.
    Lines are drawn between keypoint pairs if both keypoints meet the draw_condition
    
    Args:
        frame: The video frame to draw on
        keypoints: List of keypoints to draw
        color_circles: Color for keypoint circles (default: green)
        color_lines: Color for connecting lines (default: blue)
        hide_legs: If True, leg keypoints and connections will not be drawn (default: False)
        wholebody: Whether to use wholebody keypoints (default: False)
    """

    def draw_condition(x,y):
        return ~np.isnan(x) and ~np.isnan(y) and x >= 0 and x < 1980 and y >= 0 and y < 1080
    
    # Define leg keypoints to skip if hide_legs is True
    leg_keypoints = [11, 12, 13, 14, 15, 16] if hide_legs else []
    
    # Draw lines between keypoint pairs
    for pair in KEYPOINT_PAIRS:
        idx1, idx2 = pair
        # Skip leg connections if hide_legs is True
        if hide_legs and (idx1 in leg_keypoints or idx2 in leg_keypoints):
            continue
            
        if idx1 < num_keypoints and idx2 < num_keypoints and (keypoints[idx1] is not None) and (keypoints[idx2] is not None) and (idx1 not in hidden_keypoints) and (idx2 not in hidden_keypoints):
            x1, y1 = keypoints[idx1]
            x2, y2 = keypoints[idx2]
            
            if draw_condition(x1,y1) and draw_condition(x2,y2):  # Only draw if both points have sufficient confidence
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_lines, 2)  # Blue lines
    
    # Draw circles for keypoints
    for i in range(0, len(keypoints)):
        # Skip leg keypoints if hide_legs is True
        if hide_legs and i in leg_keypoints:
            continue
            
        if keypoints[i] is not None and i not in hidden_keypoints:
            x, y = keypoints[i]
            if draw_condition(x,y):
                cv2.circle(frame, (int(x), int(y)), 5, color_circles, thickness=-1)  # Green circles
    
    return frame

def overlay_keypoints_on_image(video_path, points, output_path, frame_idx, gt=None, hide_legs=False):
    """
    Overlays 2D points on an image and saves the result.

    Args:
        video_path: Path to the input video.
        points: List of 2D keypoints in (x, y) format to overlay on the image.
        output_path: Path to save the modified image.
        frame_idx: gives a frame index for the image to be overlayed
        gt: List of 2D ground truth keypoints in (x, y) format to overlay on the image.
        hide_legs: If True, leg keypoints and connections will not be drawn (default: False)
    """
    # Load the image in color mode
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame
    success, image = cap.read()
    if not success:
        print(f"Frame {frame_idx} does not exist in the video.")
        return
    
    # Overlay each point on the image
    image = draw_keypoints_and_lines(image, points, hide_legs=hide_legs)

    if gt is not None:
        image = draw_keypoints_and_lines(image, gt, color_circles=(255, 0, 0), color_lines=(255,0,255), hide_legs=hide_legs)

    # Save the modified image as a PNG
    cv2.imwrite(output_path, image)
    cap.release()

    print(f"Modified image saved as {output_path}")

def overlay_points_on_image(image_path, points, output_path, gt=None):
    """
    Overlays 2D points on an image and saves the result.

    Args:
        image_path: Path to the input PNG image.
        points: List of 2D keypoints in (x, y) format to overlay on the image.
        output_path: Path to save the modified image.
        gt: List of 2D ground truth keypoints in (x, y) format to overlay on the image.
    """
    def draw_condition(x,y):
        return ~np.isnan(x) and ~np.isnan(y) and x >= 0 and x < 1980 and y >= 0 and y < 1080
    # Load the image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Image at {image_path} not found.")
        return
    
    for i in range(0, len(points)):
        x, y = points[i]
        if draw_condition(x,y):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)  # Green circles
    
    for i in range(0, len(gt)):
        if gt[i] is not None:
            x, y = gt[i]
            if draw_condition(x,y):
                cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), thickness=-1)  # Green circles

    # Save the modified image as a PNG
    cv2.imwrite(output_path, image)

    print(f"Modified image saved as {output_path}")

def overlay_points_and_lines_on_frame_and_save_as_pdf(video_path, points, output_path, frame_idx, color='green', radius=6, diameter_hands=10, line_color='blue'):
    """
    Overlays 2D points and lines between specified keypoints on a video frame and saves the result as a PDF file.

    Args:
        video_path: A video on which to overlay points and lines.
        points: List of 2D points in (x, y) format to overlay on the frame.
        output_path: Path to save the frame as a PDF file.
        frame_idx: frame index to render.
        color: Color of the points (default is green).
        radius: Radius of the points to plot (default is 5).
        line_color: Color of the lines connecting the keypoints (default is blue).
    """
    def draw_condition(x,y):
        return ~np.isnan(x) and ~np.isnan(y) and x >= 0 and x < 1980 and y >= 0 and y < 1080
    
    # Load the image in color mode
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame
    success, frame = cap.read()
    if not success:
        print(f"Frame {frame_idx} does not exist in the video.")
        return
    # Convert BGR image (OpenCV format) to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the dimensions of the frame (for DPI calculation)
    height, width, _ = frame_rgb.shape

    # Create a matplotlib figure with the same aspect ratio as the frame
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Display the frame without interpolation to avoid blurring
    ax.imshow(frame_rgb, interpolation='none')

    # Draw lines between specified keypoint pairs
    for (i, j) in KEYPOINT_PAIRS:
        # Ensure both points exist and are valid
        if points[i] is not None and points[j] is not None and (i not in hidden_keypoints and j not in hidden_keypoints):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Draw a line between the points
            if draw_condition(x1,y1) and draw_condition(x2,y2): 
                ax.plot([x1, x2], [y1, y2], color=line_color, zorder=1, linewidth=5)

    # Overlay each point on the frame
    for i, point in enumerate(points):
        if 'zoom' in video_path and point is not None and (i == 7 or i == 4 or (i >= 26 and i <= 45) or (i >= 47 and i <= 66)):
            x, y = point[0], point[1]
            if draw_condition(x,y):
                if i >= 15:
                    # Plot a circle at each point using matplotlib
                    ax.scatter(x, y, s=radius**2, c=color, zorder=2)
                else:
                    ax.scatter(x, y, s=radius**2, c=color, zorder=2)
        elif 'zoom' not in video_path and point is not None and i not in hidden_keypoints:
            x, y = point[0], point[1]
            if draw_condition(x,y):
                if i >= 15:
                    # Plot a circle at each point using matplotlib
                    ax.scatter(x, y, s=10, c=color, zorder=2)
                else:
                    ax.scatter(x, y, s=radius**2, c=color, zorder=2)


    # Hide axes
    ax.axis('off')

    # Save the modified frame as a PDF, using a tight bounding box and proper DPI
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=100, pad_inches=0)

    # Close the figure to release memory
    plt.close()
    cap.release()

    print(f"Frame with overlaid points and lines saved as {output_path}")

def show_3d_pose(points_3d, output_path, xlim=None, ylim=None, zlim=None, show=True):
    """
    Plots 3D points as matplotlib pyplot.

    Args:
        points_3d: List of 3D points in (x, y, z) format to plot.
        output_path: Path to save the plot.
    """
    poses_3d_vis_x = []
    poses_3d_vis_y = []
    poses_3d_vis_z = []
    for point in points_3d:
        if not (point[0] == point[1] == point[2] == 0):
            poses_3d_vis_x.append(point[0])
            poses_3d_vis_y.append(point[1])
            poses_3d_vis_z.append(point[2])
    ax = plt.axes(projection='3d')
    ax.scatter3D(poses_3d_vis_x, poses_3d_vis_y, poses_3d_vis_z)
    if xlim is not None and ylim is not None and zlim is not None:
        ax.axes.set_xlim3d(left=xlim[0], right=xlim[1]) 
        ax.axes.set_ylim3d(bottom=ylim[0], top=ylim[1]) 
        ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
    
    for pair in KEYPOINT_PAIRS:
        idx1, idx2 = pair
        x1, y1, z1 = points_3d[idx1]
        x2, y2, z2 = points_3d[idx2]
        ax.plot([x1, x2], [y1, y2], zs=[z1,z2])
    if show:
        plt.show()
    ax.figure.savefig(output_path)

def img2video(output_dir, video_name, fps=30, pose_dir='pose/'):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir, pose_dir, '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(os.path.join(output_dir,  video_name + '.mp4'), fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()

def pose2video(poses_3d_body, output_dir, video_name, poses_3d_hands=None, fps=30, xlim=None, ylim=None, zlim=None, 
             left_wrist_index=9, right_wrist_index=10, render_body=True, dynamic_limits=False, use_wholebody=False):
        
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # Select the appropriate keypoint pairs based on the model type
    if use_wholebody:
        keypoint_pairs = WHOLEBODY_KEYPOINT_PAIRS
        num_model_keypoints = num_wholebody_keypoints
    else:
        keypoint_pairs = KEYPOINT_PAIRS
        num_model_keypoints = num_keypoints
    
    # Calculate dynamic limits if requested
    if dynamic_limits:
        # Initialize with the first point
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        # Consider both body and hand points for dynamic limits
        if render_body:
            for frame in range(len(poses_3d_body)):
                valid_points = poses_3d_body[frame][~np.isnan(poses_3d_body[frame]).any(axis=1)]
                if len(valid_points) > 0:
                    min_x = min(min_x, np.min(valid_points[:, 0]))
                    max_x = max(max_x, np.max(valid_points[:, 0]))
                    min_y = min(min_y, np.min(valid_points[:, 1]))
                    max_y = max(max_y, np.max(valid_points[:, 1]))
                    min_z = min(min_z, np.min(valid_points[:, 2]))
                    max_z = max(max_z, np.max(valid_points[:, 2]))
            
        if poses_3d_hands is not None:
            for frame in range(len(poses_3d_hands)):
                valid_points = poses_3d_hands[frame][~np.isnan(poses_3d_hands[frame]).any(axis=1)]
                if len(valid_points) > 0:
                    min_x = min(min_x, np.min(valid_points[:, 0]))
                    max_x = max(max_x, np.max(valid_points[:, 0]))
                    min_y = min(min_y, np.min(valid_points[:, 1]))
                    max_y = max(max_y, np.max(valid_points[:, 1]))
                    min_z = min(min_z, np.min(valid_points[:, 2]))
                    max_z = max(max_z, np.max(valid_points[:, 2]))

        # Calculate ranges
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        # Find the largest range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (max_x + min_x) / 2
        y_center = (max_y + min_y) / 2
        z_center = (max_z + min_z) / 2
        
        # Add padding
        padding = 0.1
        max_range_padded = max_range * (1 + 2 * padding)
        
        # Set equal ranges around each center point
        xlim = (x_center - max_range_padded/2, x_center + max_range_padded/2)
        ylim = (y_center - max_range_padded/2, y_center + max_range_padded/2)
        zlim = (z_center - max_range_padded/2, z_center + max_range_padded/2)
    
    # Set the axis limits if provided
    if xlim is not None and ylim is not None and zlim is not None:
        ax.axes.set_xlim3d(left=xlim[0], right=xlim[1]) 
        ax.axes.set_ylim3d(bottom=ylim[0], top=ylim[1]) 
        ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1])     # Initialize variables for body rendering if enabled
    scat = None
    lines = []
    
    if render_body:
        scat = ax.scatter3D(poses_3d_body[0, :, 0], poses_3d_body[0, :, 1], poses_3d_body[0, :, 2])
        for pair in keypoint_pairs:
            idx1, idx2 = pair
            if idx1 < num_model_keypoints and idx2 < num_model_keypoints:
                x1, y1, z1 = poses_3d_body[0, idx1]
                x2, y2, z2 = poses_3d_body[0, idx2]
                lines.append(ax.plot([x1, x2], [y1, y2], zs=[z1,z2]))
            
    # Hand rendering setup
    scat_hands = None
    hands = []
    if poses_3d_hands is not None:
        num_hands = poses_3d_hands.shape[1] // num_keypoints_hands
        scat_hands = ax.scatter3D(poses_3d_hands[0, :, 0], poses_3d_hands[0, :, 1], poses_3d_hands[0, :, 2])
        hands = []
        for i in range(num_hands):
            lines_hands = []
            for pair in KEYPOINT_PAIRS_HANDS:
                idx1, idx2 = pair
                if idx1 < num_keypoints_hands and idx2 < num_keypoints_hands:
                    idx1 += i*num_keypoints_hands
                    idx2 += i*num_keypoints_hands
                    x1, y1, z1 = poses_3d_hands[0, idx1]
                    x2, y2, z2 = poses_3d_hands[0, idx2]
                    lines_hands.append(ax.plot([x1, x2], [y1, y2], zs=[z1,z2]))   
            hands.append(lines_hands)    
    
    def update(frame_idx):
        # Update body if rendering is enabled
        if render_body and scat is not None:
            scat._offsets3d = (poses_3d_body[frame_idx, :, 0], poses_3d_body[frame_idx, :, 1], poses_3d_body[frame_idx, :, 2])
            for pair, line in zip(keypoint_pairs, lines):
                idx1, idx2 = pair
                if idx1 < num_model_keypoints and idx2 < num_model_keypoints:
                    line[0].set_data_3d(np.array([poses_3d_body[frame_idx, idx1], poses_3d_body[frame_idx, idx2]]).T)
        
        # Update hands if available
        if poses_3d_hands is not None and scat_hands is not None:
            scat_hands._offsets3d = (poses_3d_hands[frame_idx, :, 0], poses_3d_hands[frame_idx, :, 1], poses_3d_hands[frame_idx, :, 2])
            for i, hand in enumerate(hands):
                for pair, line in zip(KEYPOINT_PAIRS_HANDS, hand):
                    idx1, idx2 = pair
                    if idx1 < num_keypoints_hands and idx2 < num_keypoints_hands:
                        idx1 += i*num_keypoints_hands
                        idx2 += i*num_keypoints_hands
                        line[0].set_data_3d(np.array([poses_3d_hands[frame_idx, idx1], poses_3d_hands[frame_idx, idx2]]).T)

    ani = animation.FuncAnimation(fig, update, len(poses_3d_body))

    FFwriter = animation.FFMpegWriter(fps=fps, bitrate=1000)
    ani.save(os.path.join(output_dir, f'{video_name}.mp4'), writer=FFwriter)


def overlay_keypoints_on_video(video_path, poses_2d, output_path, wholebody=False):
    """
    Overlay keypoints on each frame of the video.
    Args:
        video_path: Path to the input video file.
        poses_2d: 2d poses projected to the camera's position
        output_path: Path to the output video file.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15#cap.get(cv2.CAP_PROP_FPS)
    total_frames = len(poses_2d)
    
    # Define the codec and create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        # Overlay keypoints on the frame
        if frame_number < len(poses_2d):
            frame = draw_keypoints_and_lines(frame, poses_2d[frame_number], wholebody=wholebody)
        else:
            break
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames...")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def overlay_keypoints_on_video_frame(video_path, pose_2d, frame_idx, output_path, img_name):
    """
    Overlay keypoints on one frame of the video.
    Args:
        video_path: Path to the input video file.
        pose_2d: 2d pose projected to the camera's position.
        frame_idx: index of the frame to overlay.
        output_path: Path to the output image file.
        img_name: name of the saved image.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Set the video to the nth frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Frame {frame_idx} does not exist in the video.")
    
    frame = draw_keypoints_and_lines(frame, pose_2d)

    # Save the modified image as a PNG
    cv2.imwrite(os.path.join(output_path, img_name), frame)

    print(f"Modified image saved as {os.path.join(output_path, img_name)}")