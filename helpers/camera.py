import numpy as np
import cv2
import json
import os
from scipy.spatial.transform import Rotation as R
from helpers.definitions import *

def undistort_point(u, v, K, dist_coeffs):
	# see link for reference on the equation
	# http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
	k1,k2,p1,p2, k3 = dist_coeffs
	u0 = K[0,2] # cx
	v0 = K[1,2] # cy
	fx = K[0,0]
	fy = K[1,1]
	_fx = 1.0/fx
	_fy = 1.0/fy
	y = (v - v0)*_fy
	x = (u - u0)*_fx

	r = np.sqrt(x**2 + y**2)

	u_undistort = (x * (1+ (k1*r**2) + (k2*r**4) + (k3*r**6))) + 2*p1*x*y + p2*(r**2 + 2*x**2)
	v_undistort = (y * (1+ (k1*r**2) + (k2*r**4) + (k3*r**6))) + 2*p2*y*x + p1*(r**2 + 2*y**2)

	x_undistort = fx*u_undistort+ u0
	y_undistort = fy*v_undistort+ v0

	return x_undistort, y_undistort

def create_camera_matrix(intrinsics, extrinsics):
    """
    Create the camera projection matrix from intrinsics and extrinsics.

    Args:
        intrinsics: A 3x3 intrinsic matrix (K).
        extrinsics: A tuple containing rotation (3x3) and translation (3x1) matrices.

    Returns:
        A 3x4 projection matrix (P).
    """
    #print(intrinsics)
    rot, t = extrinsics
    # Concatenate R and t to form a 3x4 matrix
    Rt = np.hstack((rot, t.reshape(-1, 1)))
    # Projection matrix is K * [R | t]
    return intrinsics @ Rt

def euler_to_rotation_matrix(euler_ZYX, cam_name=None):
    """
    Converts Euler angles (ZYX) to a 3x3 rotation matrix.
    Args:
        euler_ZYX: A 3-element vector (roll, pitch, yaw) in radians.
    Returns:
        A 3x3 rotation matrix.
    """
    seq = 'ZYX'
    rotation = R.from_euler(seq, euler_ZYX, degrees=True)
    return rotation.as_matrix()

def project_point(point_3d, intrinsic_matrix, rot, t, distortion):
    """
    Projects a 3D point into 2D using camera intrinsics, extrinsics, and distortion.
    Args:
        point_3d: The 3D point as a np.array of shape (3,).
        intrinsic_matrix: Camera intrinsic matrix (3x3).
        rot: Rotation matrix as a 3x3 matrix.
        t: Translation vector as a 3x1 vector.
        distortion: Distortion coefficients.
    Returns:
        2D point coordinates as an np.array of shape (2,).
    """

    (rvec, jac) = cv2.Rodrigues(rot)
    ## Map the 3D point to 2D point 
    points_2d, _ = cv2.projectPoints(point_3d, 
                                 rvec, t, 
                                 intrinsic_matrix, 
                                 distortion)
    

    return points_2d.flatten()

def project_points(points_3d, intrinsic_matrix, rot, t, distortion):
    """
    Projects 3D points into 2D using camera intrinsics, extrinsics, and distortion.
    Args:
        points_3d: The 3D points as an np.array of shape (N, 3).
        intrinsic_matrix: Camera intrinsic matrix (3x3).
        rot: Rotation matrix as a 3x3 matrix.
        t: Translation vector as a 3x1 vector.
        distortion: Distortion coefficients.
    Returns:
        2D point coordinates as an np.array of shape (N, 2).
    """
    (rvec, jac) = cv2.Rodrigues(rot)
    ## Map the 3D points to 2D points
    points_2d, _ = cv2.projectPoints(points_3d,
                                    rvec, t, 
                                    intrinsic_matrix, 
                                    distortion)
    return points_2d.squeeze()

def load_camera_params(intr_path : str, extr_path : str, convert=True, suffix: str = "", named=False, import_extrinsics_matrix=False):
    """
    Loads camera params from specified paths
    Params:
        intr_path: A path to the folder containing the intrinics for each camera. 
            Must contain files named {camera_name}_intrinsics.json
        extr_path: The path to the file containing the camera_poses
    Returns:
        (camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices)
    """
    if named:
        camera_intrinsics = {}
        camera_extrinsics = {}
        distortion_coeffs = {}
        camera_matrices = {}
    else:
        camera_intrinsics = []
        camera_extrinsics = []
        distortion_coeffs = []
        camera_matrices = []
    with open(extr_path, 'r') as f:
        extr = json.load(f)
    for camera in cam_names:
        cam = f"{camera}{suffix}"
        with open(os.path.join(intr_path, f'{cam}_intrinsics.json'), 'r') as f:
            intr = json.load(f)['sensors']['RGB']
        if named:
            camera_intrinsics[camera] = np.array(intr['intrinsics']['data']).reshape(3,3)
            distortion_coeffs[camera] = tuple(intr['distortionCoefficients']['data'])
        else:
            camera_intrinsics.append(np.array(intr['intrinsics']['data']).reshape(3,3))
            distortion_coeffs.append(tuple(intr['distortionCoefficients']['data']))

        if import_extrinsics_matrix:
            extrinsics = (np.array(extr[f"blender2{camera}"]).reshape(4,4)[:3, :3], np.array(extr[f"blender2{camera}"]).reshape(4,4)[:3, 3])
        elif convert:
            rot = euler_to_rotation_matrix(extr[cam]["euler_ZYX"], cam)
            t = np.array(extr[cam]["t"])
            rot = rot.T
            t = -rot @ t

            # Create camera projection matrix
            extrinsics = (rot, t)
        else:
            rot = np.array(extr[cam]["rot_mat"]).T
            t = - rot @ np.array(extr[cam]["t"]).flatten()
            extrinsics = (rot, t)

        camera_matrix = create_camera_matrix(np.array(intr['intrinsics']['data']).reshape(3,3), extrinsics)

        if named:
            camera_extrinsics[camera] = extrinsics
            camera_matrices[camera] = camera_matrix
        else:
            camera_extrinsics.append(extrinsics)
            camera_matrices.append(camera_matrix)

    return camera_intrinsics, camera_extrinsics, distortion_coeffs, camera_matrices

def project_points_safe(points_3d, intrinsic_matrix, rot, t, distortion):
    """
    Projects 3D points into 2D using camera intrinsics, extrinsics, and distortion.
    This version includes safety checks for NaN values and other errors.
    
    Args:
        points_3d: The 3D points as an np.array of shape (N, 3).
        intrinsic_matrix: Camera intrinsic matrix (3x3).
        rot: Rotation matrix as a 3x3 matrix.
        t: Translation vector as a 3x1 vector.
        distortion: Distortion coefficients.
    Returns:
        2D point coordinates as an np.array of shape (N, 2).
    """
    # Check for NaN values in the input
    if np.isnan(points_3d).any():
        print("Warning: NaN values detected in 3D points during projection")
        # Replace NaN with zeros to avoid crashes
        points_3d = np.nan_to_num(points_3d, nan=0.0)
    
    try:
        (rvec, jac) = cv2.Rodrigues(rot)
        
        # Map the 3D points to 2D points
        points_2d, _ = cv2.projectPoints(points_3d,
                                        rvec, t, 
                                        intrinsic_matrix, 
                                        distortion)
        
        # Check for NaN or infinity in the result
        if np.isnan(points_2d).any() or np.isinf(points_2d).any():
            print("Warning: NaN or Inf values detected in projected 2D points")
            points_2d = np.nan_to_num(points_2d, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        return points_2d.reshape(-1, 2)
    
    except Exception as e:
        print(f"Error in point projection: {str(e)}")
        # Return zero coordinates as fallback
        return np.zeros((len(points_3d), 2))
