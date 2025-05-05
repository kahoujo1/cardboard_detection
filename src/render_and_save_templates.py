import numpy as np
import cv2
import ezdxf
import yaml
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.render import Render


def main():
    render = Render()
    z_min = 500
    z_max = 1000
    z_step = 100
    angle_step = 0.1 # rad
    cad_path = "cad_files/cardboard1.dxf"
    yaml_path = "rendered_images.yaml"
    imgs_folder = "rendered_images/"
    camera_matrix = np.array([[
        611.4880761967455,
        0.0,
        319.14726484541006
    ],
    [
        0.0,
        609.9234611812698,
        243.73808468652115
    ],
    [
        0.0,
        0.0,
        1.0
    ]])
    dist_coeffs = np.array([
        0.09950062845024064,
        -0.129679646398333,
        0.0004996523001434527,
        0.0009287372096340519,
        -0.2667766396346066
    ])
    all_metadata = [] # list to store all metadata
    # set the camera matrix and image shape
    render.set_camera_matrix(camera_matrix, dist_coeffs)
    render.set_image_shape((480, 640))
    # render the CAD model with different translations and rotations and save it to the yaml file
    index = 0
    for z in range(z_min, z_max, z_step):
        for angle in np.arange(0, 2 * np.pi, angle_step):
            translation = [0, 0, z]
            rotation = [np.pi, 0, angle.item()]
            edges = render.render(cad_path, translation, rotation)
            edges = render.crop_image(edges)
            edges = cv2.Canny(edges, 10, 50)
            # Save the rendered image to a file
            image_path = f"rendered_image_{index}.png"
            cv2.imwrite(imgs_folder+image_path, edges)   
            # Save the rendered image to the yaml file
            metadata = {
                'image_path': image_path,
                'translation': translation,
                'rotation': rotation
            }
            all_metadata.append(metadata)
            index += 1
    # Save all metadata to the yaml file
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(all_metadata, yaml_file)
            

if __name__ == "__main__":
    main()