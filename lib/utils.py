"""
Utility functions for camera calibration and image processing.
"""
import cv2
import numpy as np
import yaml
import os
from typing import Tuple

# save the camera matrix and distortion coefficients to a yaml file
def save_camera_params(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, filename:str = 'camera_params.yaml') -> None:
    """
    Save camera parameters to a YAML file.
    """
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    with open(filename, 'w') as file:
        yaml.dump(data, file)
    print(f"Camera parameters saved to {filename}")

# load the camera matrix and distortion coefficients from a yaml file
def load_camera_params(filename: str='camera_params.yaml') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera parameters from a YAML file.
    Args:
        filename (str): Path to the YAML file.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Camera matrix and distortion coefficients.
    """
    
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    
    return camera_matrix, dist_coeffs