"""
Segmentation of an image using HSV color space.
"""
import cv2
import numpy as np
from typing import Tuple, List

LOWER_BACKGROUND_COLOR = [80, 0, 0]
UPPER_BACKGROUND_COLOR = [120, 255, 255]

def get_mask(img: np.ndarray) -> np.ndarray:
    """
    Returns the mask of the image.
    """
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array(LOWER_BACKGROUND_COLOR), np.array(UPPER_BACKGROUND_COLOR))
    mask = cv2.bitwise_not(mask)
    # remove noise from the picture
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]  # Get sizes of components, excluding the background
    min_size = 100  # Minimum size of components to keep

    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            new_mask[labels == i] = 255

    mask = new_mask
    return mask

def segment_img(img: np.ndarray) -> np.ndarray:
    """
    Segments the image.
    """
    mask = get_mask(img)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img