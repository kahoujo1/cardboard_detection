"""
Test for the template matching module.
"""

import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.template_matching import find_best_match
from lib.color_segmentation import segment_img
from lib.edges import get_edges
from lib.utils import load_camera_params

def main():
    """
    Main function to run the tests.
    """
    # Load the test image
    imgs_folder = "test_images"
    output_folder = "output/template_matching_results"
    os.makedirs(output_folder, exist_ok=True)
    camera_matrix, dist_coeffs = load_camera_params()
    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            segmented_img = segment_img(img)
            edges = get_edges(segmented_img)
            # Perform template matching
            location, tvec, rvec, best_img = find_best_match(edges, "rendered_images", "rendered_images.yaml")
            print(f"Best match for {img_name}: {best_img}")
            print(f"Translation vector: {tvec}")
            print(f"Rotation vector: {rvec}")
            print(f"Location: {location}")
            best_img = cv2.imread(os.path.join("rendered_images", best_img), cv2.IMREAD_GRAYSCALE)
            # Draw the best match on the original image
            h, w = best_img.shape
            x, y = location
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # Draw a rectangle around the best match
            cv2.rectangle(edges, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # project the translation vector on the image
            img_point = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float64), rvec, tvec, camera_matrix, dist_coeffs)
            img_point = img_point[0].reshape(-1, 2)
            img_point = img_point.astype(int)
            cv2.circle(edges, (img_point[0][0], img_point[0][1]), 5, (255, 0, 0), -1)
            # Save the result
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    main()