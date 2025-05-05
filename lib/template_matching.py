""""""

import cv2
import yaml
import numpy as np
import os
import sys
from typing import List, Tuple
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.utils import load_camera_params

X_STEP = 20
Y_STEP = 20
MIN_IMG_SIZE = 50
MIN_EDGE_POINTS = 300

def partial_template_match(image: np.ndarray, template :np.ndarray, x_step:int = X_STEP, y_step:int = Y_STEP) -> Tuple[Tuple[int, int], float]:
    """
    Perform partial template matching using normalized cross-correlation.
    
    Args:
        image (numpy.ndarray): The input image in which to search for the template.
        template (numpy.ndarray): The template to match against the image.
        x_step (int): Step size for x direction (img width).
        y_step (int): Step size for y direction (img height).

    Returns:
        best_loc (tuple): The (x, y) coordinates of the top-left corner of the best match.
        best_score (float): The score of the best match.
    """
    img_h, img_w = image.shape[:2]
    temp_h, temp_w = template.shape[:2]
    num_of_img_points = np.sum(image > 0)
    num_of_temp_points = np.sum(template > 0)
    best_score = -np.inf
    best_loc = None

    # Slide template across the image (including positions where it overflows)
    for y in range(-temp_h + 1, img_h, y_step):
        for x in range(-temp_w + 1, img_w, x_step):
            # Define overlapping regions
            img_x_start = max(x, 0)
            img_y_start = max(y, 0)
            img_x_end = min(x + temp_w, img_w)
            img_y_end = min(y + temp_h, img_h)
            temp_x_start = max(0, -x)
            temp_y_start = max(0, -y)
            temp_x_end = min(temp_w, img_w - x)
            temp_y_end = min(temp_h, img_h - y)
            # Extract overlapping parts
            img_patch = image[img_y_start:img_y_end, img_x_start:img_x_end]
            temp_patch = template[temp_y_start:temp_y_end, temp_x_start:temp_x_end]
            curr_num_of_img_points = np.sum(img_patch > 0)
            curr_num_of_temp_points = np.sum(temp_patch > 0)
            # Check if the patch is too small
            if img_patch.shape[0]< MIN_IMG_SIZE or img_patch.shape[1]< MIN_IMG_SIZE or temp_patch.shape[0] < MIN_IMG_SIZE or temp_patch.shape[1] < MIN_IMG_SIZE:
                continue
            if np.sum(temp_patch) < MIN_EDGE_POINTS or np.sum(img_patch) < MIN_EDGE_POINTS:
                continue
            # Calculate the score based on the method
            # Normalized cross-correlation
            img_patch = img_patch.astype(np.float32)
            temp_patch = temp_patch.astype(np.float32)
            img_patch -= np.mean(img_patch)
            temp_patch -= np.mean(temp_patch)
            score = np.sum(img_patch * temp_patch) / (np.linalg.norm(img_patch) * np.linalg.norm(temp_patch))*curr_num_of_temp_points
            # Update best score and location     
            if score >= best_score:
                best_score = score
                best_loc = (x, y)           

    return best_loc, best_score

def get_pose_estimation(center: Tuple[int, int], z: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the translation from the center of the object in the camera coordinates.
    
    Args:
        center (tuple): The (x, y) coordinates of the center of the object.
        z (float): The z coordinate of the object in the camera coordinates.

    Returns:
        tvec (numpy.ndarray): The translation vector.
    """
    camera_matrix, dist_coeffs = load_camera_params()
    centre_homogeneous = np.array([center[0], center[1], 1])
    centre_homogeneous = np.dot(np.linalg.inv(camera_matrix), centre_homogeneous)
    centre_homogeneous = centre_homogeneous / centre_homogeneous[2]
    centre_homogeneous*= z
    return centre_homogeneous
    


def find_best_match(edges: np.ndarray, template_folder: str, template_yaml:str, x_step:int = X_STEP, y_step:int = Y_STEP):
    """
    Find the best match for a template in an image using partial template matching.
    
    Args:
        image (numpy.ndarray): The input image in which to search for the template.
        template_folder (str): Path to the folder containing the templates.
        template_yaml (str): Path to the YAML file containing template metadata.
        x_step (int): Step size for x direction (img width).
        y_step (int): Step size for y direction (img height).

    Returns:
        tvec (numpy.ndarray): The translation vector.
        rotation (numpy.ndarray): The rotation vector.
        image_name (str): The name of the best matching template image.
    """
    # load the rendered templates
    # open the yaml file
    with open(template_yaml, 'r') as file:
        metadata = yaml.safe_load(file)
    print(f"Loaded {len(metadata)} templates from {template_yaml}")
    max_score = -np.inf
    location = None
    best_translation = None
    best_rotation = None
    best_image_name = None
    # preprocess the image
    test_img = deepcopy(edges)
    min_x = np.min(np.argwhere(test_img > 0)[:, 0])
    min_y = np.min(np.argwhere(test_img > 0)[:, 1])
    max_x = np.max(np.argwhere(test_img > 0)[:, 0])
    max_y = np.max(np.argwhere(test_img > 0)[:, 1])
    test_img = test_img[min_x:max_x, min_y:max_y]
    test_img = cv2.GaussianBlur(test_img, (3, 3), 0)
    test_img = (test_img*255).astype(np.uint8)

    for entry in metadata:
        # load the template image and metadata
        image_name = entry['image_path']
        translation = entry['translation']
        rotation = entry['rotation']
        # print(f"Image: {image_name}, Translation: {translation}, Rotation: {rotation}")
        template_img = cv2.imread(os.path.join(template_folder, image_name), cv2.IMREAD_GRAYSCALE)
        template_img = (template_img*255).astype(np.uint8)
        loc, score = partial_template_match(test_img, template_img)
        if score >= max_score:
            max_score = score
            location = loc
            best_translation = translation
            best_rotation = rotation
            best_image_name = image_name
    print("best location: ", location)
    print("best score: ", max_score)
    print("best translation: ", best_translation)
    print("best rotation: ", best_rotation)
    print("best image name: ", best_image_name)
    # the translation is in the form of [0,0,z]
    # we need to convert the translation using the image pixel coordinates
    template_img = cv2.imread(os.path.join(template_folder, best_image_name), cv2.IMREAD_GRAYSCALE)
    template_img = (template_img*255).astype(np.uint8)
    # get the size of the template image
    temp_h, temp_w = template_img.shape[:2]
    print(min_x, min_y)
    location = (location[0] + min_y, location[1] + min_x)
    # get the center of the template image
    temp_center = (temp_w // 2, temp_h // 2)
    # get the center of the location
    loc_center = (location[0] + temp_center[0], location[1] + temp_center[1])
    tvec = get_pose_estimation(loc_center, best_translation[2])
    tvec /= 1000 # convert to meters
    return location, np.array(tvec), np.array(best_rotation), best_image_name