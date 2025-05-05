"""
Module for optimization of the chamfer distance.
"""

import numpy as np
from typing import List, Tuple
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.utils import load_camera_params
from lib.cad_extract import get_cad_points
from scipy.optimize import least_squares, minimize
from scipy.spatial import KDTree

def smallest_distance(A, B):
    """
    Calculates the smallest distance between the points A and a set of points B.
    """
    tree = KDTree(B)
    dist = tree.query(A)[0]
    return dist

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def objective_function(params, camera_matrix, dist_coef, img_points, obj_points, img_width, img_height):
    """
    Objective function for optimization
    """
    tvec = np.array([params[0], params[1], params[2]])
    rvec = np.array([params[3], params[4], params[5]])
    proj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coef)
    proj_points = proj_points.squeeze()
    # Filter out projected points that are outside the image dimensions
    proj_points = proj_points[(proj_points[:, 0] >= 0) & (proj_points[:, 0] < img_width) &
                            (proj_points[:, 1] >= 0) & (proj_points[:, 1] < img_height)]
    # print(proj_points[0])
    if proj_points.shape[0] < 50:
        return 1e6
    dist = chamfer_distance(img_points, proj_points)
    # print(chamfer_distance(img_points, proj_points))
    return dist

def sample_img_points(edges:np.ndarray, num_of_img_points:int = 500) -> np.ndarray:
    """
    Samples points from the edges of the image.

    Args:
        edges (numpy.ndarray): The edges of the image.
        num_of_img_points (int): Number of points to sample.
    
    Returns:
        numpy.ndarray: Sampled points.
    """
    points = np.argwhere(edges > 0)
    # print(points.shape)
    idx = np.random.choice(points.shape[0], min(num_of_img_points,len(points)), replace=False)
    points = points[idx]
    tmp = points[:,0].copy()
    points[:,0] = points[:,1]
    points[:,1] = tmp
    return points.astype(np.float32)

def optimize_chamfer_distance(edges:np.ndarray, cad_filepath: str, init_params:np.ndarray, num_of_object_points:int = 500, num_of_img_points:int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimizes the chamfer distance between the projected edges and the image edges.

    Args:
        edges (numpy.ndarray): The edges of the image.
        cad_filepath (str): Path to the CAD file.
        init_params (numpy.ndarray): Initial parameters for optimization.
        num_of_object_points (int): Number of object points to sample.
        num_of_img_points (int): Number of image points to sample.
    
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The translation and rotation vectors.
    """
    bounds = [(-1, 1), (-1, 1), (0.3, 2), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]  # Bounds for each parameter
    params = init_params
    camera_matrix, dist_coeffs = load_camera_params()
    img_height, img_width = edges.shape     
    cad_points = get_cad_points(cad_filepath, num_of_object_points)
    img_points = sample_img_points(edges, num_of_img_points)

    res = minimize(objective_function, params, args=(camera_matrix, dist_coeffs, img_points, cad_points, img_width, img_height), method='L-BFGS-B', bounds=bounds)
    print("Optimization result:", res.fun)
    tvec = res.x[:3]
    rvec = res.x[3:6]
    return tvec, rvec

