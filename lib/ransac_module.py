"""
Implementation of RANSAC algorithm for 3D pose estimation.
"""
import numpy as np
import cv2
import sys
import os
from scipy.optimize import minimize
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.optimization import objective_function
from scipy.spatial import KDTree
import random

def smallest_distance(A, B):
    """
    Calculates the smallest distance between the points A and a set of points B.
    """
    tree = KDTree(B)
    dist = tree.query(A)[0]
    return dist

Z_DISTANCE = 0.6  # Rough estimate for the initialization of the optimization

def euler_to_rvec(euler_angles):
    """
    Convert Euler angles to rotation vector.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                    [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])
    R_y = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                        [0, 1, 0],
                        [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])
    R_z = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                        [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                        [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

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

class TransformModel:
    def __init__(self):
        self.tvec = np.array([0.,0.,0.], dtype=np.float64)
        self.rvec = np.array([0.,0.,0.], dtype=np.float64)
        self.camera_matrix = None
        self.init_params = np.array([0, 0, 1, np.pi, 0, 0], dtype=np.float64)
        self.dist_coef = None
        self.img_width = 0
        self.img_height = 0

    def set_image_size(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def set_init_params(self, init_params):
        self.init_params = init_params
        self.tvec = init_params[:3]
        self.rvec = init_params[3:]

    def set_camera(self, camera_matrix, dist_coef):
        self.camera_matrix = camera_matrix
        self.dist_coef = dist_coef

    def fit(self, img_points, obj_points):
        # print("img_points:", img_points, img_points.shape)
        # print("obj_points:", obj_points, obj_points.shape)
        bounds = [(-1, 1), (-1, 1), (0.2, 2), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]  # Bounds for each parameter
        # bounds = [(-2, 2), (-2, 2), (0, 3), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        obj_points = np.array(obj_points, dtype=np.float64)  # Convert to float32
        img_points = np.array(img_points, dtype=np.float64)  # Convert to float32
        params = deepcopy(self.init_params)
        res = minimize(objective_function, params, args=(self.camera_matrix, self.dist_coef, img_points, obj_points, self.img_width, self.img_height), bounds=bounds, method='L-BFGS-B')
        self.tvec = res.x[:3]
        self.rvec = res.x[3:]
        

    def predict(self, obj_points):
        proj_points, _ = cv2.projectPoints(obj_points, self.rvec, self.tvec, self.camera_matrix, self.dist_coef)
        return proj_points.squeeze()
    
class RANSAC:
    def __init__(self, n=10, m = 50, k=1000, threshold=1.0, model=None, loss=smallest_distance, min_n_inliers = 20):
        self.n = n # number of points to sample
        self.m = m # number of points to sample for the model
        self.k = k # number of iterations
        self.threshold = threshold # threshold for inlier
        self.model = model # a model class with fit and predict methods
        self.loss = loss # a loss function
        self.max_inliers = 0
        self.min_n_inliers = min_n_inliers
        self.max_value = -np.inf

    def fit(self, img_points, obj_points):
        """
        Fits the model to the data using RANSAC algorithm

        Args:
            X (np.ndarray): The fit points
            Y (np.ndarray): The target points
        """
        if self.model is None or self.loss is None:
            raise ValueError('Model and loss must be set')
        pose = get_position_estimate(img_points, self.model.camera_matrix, self.model.dist_coef, Z_DISTANCE)
        print('Pose:', pose)
        rvec = euler_to_rvec(np.array([0, 0, 0])) 
        for _ in range(self.k):
            if _ % 10 == 0:
                print("Iteration:", _)
            img_points_, obj_points_ = self.get_random_correspondences(img_points, obj_points)
            maybe_model = deepcopy(self.model)
            yaw = np.random.uniform(-2*np.pi, 2*np.pi)
            roll = random.choice([0, np.pi])
            rvec = euler_to_rvec(np.array([roll, 0, yaw]))
            init_params = np.array([pose[0], pose[1], pose[2], rvec[0], rvec[1], rvec[2]], dtype=np.float64)
            maybe_model.set_init_params(init_params)
            maybe_model.fit(img_points_, obj_points_)
            predicted = maybe_model.predict(obj_points)
            if np.isnan(predicted).any() or np.isinf(predicted).any():
                continue
            loss = self.loss(img_points, maybe_model.predict(obj_points))
            # print('Loss:', loss)
            thresholded = loss < self.threshold
            # print('Number of inliers:', np.sum(thresholded))
            value = np.sum(thresholded)/len(predicted)
            if value > self.max_value:
                print('New best model found with value:', value)
                self.max_value = value
                self.max_inliers = np.sum(thresholded)
                self.model = maybe_model

    def predict(self, X):
        return self.model.predict(X)

    def get_random_correspondences(self, X, Y):
        """
        Gets n random correspondences from the data

        Args:
            X (np.ndarray): The fit points
            Y (np.ndarray): The target points
        """
        indices = np.random.choice(len(X), min(len(X), self.n))
        x = X[indices]
        indices = np.random.choice(len(Y), min(len(Y), self.m))
        y = Y[indices]
        return x, y