"""
Method for edge detection.

This module is made as a separate file to allow for future upgrades, for now, we implement the basic Canny edge detection.
"""
import cv2
import numpy as np
from typing import Tuple, List

def get_edges(image: np.ndarray) -> np.ndarray:
    """
    Get edges from an image using Canny edge detection.
    
    Args:
        image (numpy.ndarray): The input image.
        
    Returns:
        numpy.ndarray: The edges detected in the image.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 10, 50)
    
    return edges