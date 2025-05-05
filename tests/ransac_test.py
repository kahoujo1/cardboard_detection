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
from lib.ransac_module import RANSAC, TransformModel,sample_img_points
from lib.utils import load_camera_params
from lib.cad_extract import get_cad_points

def get_position_estimate(points, camera_matrix, dist_coef, z_distance):
    """
    Get the position estimate of the object.
    """
    centre = np.mean(points, axis=0)
    centre_homogeneous = np.array([centre[0], centre[1], 1])
    centre_homogeneous = np.dot(np.linalg.inv(camera_matrix), centre_homogeneous)
    centre_homogeneous = centre_homogeneous / centre_homogeneous[2]
    centre_homogeneous*= z_distance
    return centre_homogeneous

def draw_best_match(img, tvec, rvec, cad_path, camera_matrix, dist_coeffs):
    """
    Draw the best match on the original image.
    """
    img_height, img_width = img.shape[:2]
    # Load the CAD model points
    obj_points = get_cad_points(cad_path)
    proj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    proj_points = proj_points.squeeze()
    proj_points = proj_points[(proj_points[:, 0] >= 0) & (proj_points[:, 0] < img_width) &
                            (proj_points[:, 1] >= 0) & (proj_points[:, 1] < img_height)]
    # Draw the projected points on the image
    for point in proj_points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    return img


def main():
    """
    Main function to run the tests.
    """
    # Load the test image
    imgs_folder = "test_images"
    output_folder = "output/ransac_results"
    dxf_path = "cad_files/cardboard1.dxf"
    os.makedirs(output_folder, exist_ok=True)
    camera_matrix, dist_coeffs = load_camera_params()
    cad_points = get_cad_points(dxf_path, num_points=500)
    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            segmented_img = segment_img(img)
            edges = get_edges(segmented_img)
            points = sample_img_points(edges, num_of_img_points=500)  

            # Perform template matching
            model = TransformModel()
            model.set_image_size(edges.shape[1], edges.shape[0])
            model.set_camera(camera_matrix, dist_coeffs)
            pose = get_position_estimate(points, camera_matrix, dist_coeffs, 0.6)
            model.set_init_params(np.array([pose[0], pose[1], pose[2], 0, 0, 1], dtype=np.float64))
            ransac = RANSAC(n=50, m= 100, k=100, threshold=3.0, model=model)
            ransac.fit(points, cad_points)
            tvec = ransac.model.tvec
            rvec = ransac.model.rvec
            # Draw the best match on the original image
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # project the translation vector on the image
            edges = draw_best_match(edges, tvec, rvec, dxf_path, camera_matrix, dist_coeffs)
            # Save the result
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    main()