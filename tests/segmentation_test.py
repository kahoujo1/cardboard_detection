import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.color_segmentation import segment_img
"""
Tests for the segmentation module.
"""

def main():
    """
    Main function to run the tests.
    """
    imgs_folder = "test_images"
    output_folder = "output/segmentation_results"
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            segmented_img = segment_img(img)
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, segmented_img)
if __name__ == "__main__":
    main()