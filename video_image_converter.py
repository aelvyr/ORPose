#!/usr/bin/env python
import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

def images_to_video(image_folder, output_video_path, fps=30, image_format='png'):
    """
    Convert a sequence of images to a video file.
    
    Args:
        image_folder (str): Path to the folder containing image sequences
        output_video_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        image_format (str): Format of input images (png, jpg, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    
    # Get all images in the folder with the specified format
    images = sorted(glob.glob(os.path.join(image_folder, f'*.{image_format}')))
    
    if not images:
        print(f"No {image_format} images found in {image_folder}")
        return False
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # Process all images
    print(f"Converting {len(images)} images to video...")
    for img_path in tqdm(images):
        frame = cv2.imread(img_path)
        out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_video_path}")
    return True


def video_to_images(video_path, output_folder, image_format='png'):
    """
    Extract frames from a video file and save them as individual images.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the folder where images will be saved
        image_format (str): Format to save images (png, jpg, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video information:")
    print(f"- Total frames: {frame_count}")
    print(f"- FPS: {fps}")
    
    # Extract frames
    print(f"Extracting frames to {output_folder}...")
    frame_idx = 0
    
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as an image
            output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.{image_format}")
            cv2.imwrite(output_path, frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release the video capture
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_folder}")
    return True


def convert(direction, input_path, output_path, fps=30, image_format='png'):
    """
    Convert between video and image sequences.
    
    Args:
        direction (str): 'images_to_video' or 'video_to_images'
        input_path (str): Path to input video file or folder containing images
        output_path (str): Path to output video file or folder for extracted images
        fps (int): Frames per second (for images_to_video)
        image_format (str): Format of images (png, jpg, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if direction == 'images_to_video':
        return images_to_video(input_path, output_path, fps, image_format)
    elif direction == 'video_to_images':
        return video_to_images(input_path, output_path, image_format)
    else:
        print(f"Invalid direction: {direction}. Use 'images_to_video' or 'video_to_images'.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between video and image sequences")
    parser.add_argument(
        "direction",
        choices=["images_to_video", "video_to_images"],
        help="Conversion direction"
    )
    parser.add_argument(
        "input_path",
        help="Path to input video file or folder containing images"
    )
    parser.add_argument(
        "output_path",
        help="Path to output video file or folder for extracted images"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Frames per second (for images_to_video)"
    )
    parser.add_argument(
        "--format", 
        default="png",
        help="Format of images (png, jpg, etc.)"
    )
    
    args = parser.parse_args()
    
    convert(args.direction, args.input_path, args.output_path, args.fps, args.format)
