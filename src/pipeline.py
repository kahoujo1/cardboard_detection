"""
The code for the whole pipeline.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.color_segmentation import segment_img
from lib.edges import get_edges
from lib.template_matching import find_best_match
from lib.utils import load_camera_params
from lib.optimization import optimize_chamfer_distance
from lib.cad_extract import get_cad_points


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
    # Load the test image
    img_filepath = "test_images/photo_8.png"
    cad_path = "cad_files/cardboard1.dxf"
    # Step 1: Segment the image
    img = cv2.imread(img_filepath)
    segmented_img = segment_img(img)
    print("Image segmented.")
    # Step 2: Get edges
    edges = get_edges(segmented_img)
    print("Edges detected.")
    # Step 3: Perform template matching
    camera_matrix, dist_coeffs = load_camera_params()
    location, tvec, rvec, best_img = find_best_match(edges, "rendered_images", "rendered_images.yaml")
    print(f"Best match for {img_filepath}: {best_img}")
    print(f"Translation vector: {tvec}")
    print(f"Rotation vector: {rvec}")
    # Step 4: Perform the optimization
    init_params = np.array([tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2]])
    tvec, rvec = optimize_chamfer_distance(edges, cad_path, init_params)
    print(f"Optimized translation vector: {tvec}")
    print(f"Optimized rotation vector: {rvec}")
    # Step 5: Draw the best match on the original image
    final_img = draw_best_match(img, tvec, rvec, cad_path, camera_matrix, dist_coeffs)
    # Save the result
    output_folder = "output/pipeline_results"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(img_filepath))
    cv2.imwrite(output_path, final_img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()