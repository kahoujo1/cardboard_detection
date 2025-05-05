"""
Script to render a CAD model with a given translation and rotation.
"""

import numpy as np
import cv2
import ezdxf



class Render:
    def __init__(self):
        self.tvec = np.zeros((3, 1), dtype=np.float32)
        self.rvec = np.zeros((3, 1), dtype=np.float32)
        self.camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        self.image_shape = (1000, 1000)  # Default image shape


    def approximate_arc(self, center, radius, start_angle_deg, end_angle_deg, segments=20):
        angles = np.linspace(np.deg2rad(start_angle_deg), np.deg2rad(end_angle_deg), segments)
        points = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ])
        return [(points[i], points[i+1]) for i in range(len(points) - 1)]
    
    def extract_polyline_edges(self, polyline):
        edges = []
        points = [v for v in polyline.vertices()]  # 👈 FIX: call the method
        for i in range(len(points) - 1):
            edges.append(([points[i][0], points[i][1]], [points[i+1][0], points[i+1][1]]))
        # Check if it's closed (optional)
        if polyline.closed:
            edges.append(([points[-1][0], points[-1][1]], [points[0][0], points[0][1]]))
        return edges
        
    def load_dxf_edges(self, filepath, arc_segments=50):
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        edges = []

        for entity in msp:
            if entity.dxftype() == "LINE":
                edges.append((
                    [entity.dxf.start[0], entity.dxf.start[1]],
                    [entity.dxf.end[0], entity.dxf.end[1]]
                ))
            
            elif entity.dxftype() == "ARC":
                center = [entity.dxf.center[0], entity.dxf.center[1]]
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                arc_edges = self.approximate_arc(center, radius, start_angle, end_angle, arc_segments)
                edges.extend(arc_edges)

            elif entity.dxftype() == "CIRCLE":
                center = [entity.dxf.center[0], entity.dxf.center[1]]
                radius = entity.dxf.radius
                circle_edges = self.approximate_arc(center, radius, 0, 360, arc_segments)
                edges.extend(circle_edges)

            elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                polyline_edges = self.extract_polyline_edges(entity)
                edges.extend(polyline_edges)

        # center the edges around the origin
        edges = np.array(edges)
        edges[:, :, 0] -= np.mean(edges[:, :, 0])
        edges[:, :, 1] -= np.mean(edges[:, :, 1])
        return np.array(edges, dtype=np.float32)


    def euler_to_rvec(self, euler_angles):
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
    
    def to_3d_edges(self, edges_2d):
        edges_3d = []
        for start, end in edges_2d:
            p1 = np.array([start[0], start[1], 0.0], dtype=np.float32)
            p2 = np.array([end[0], end[1], 0.0], dtype=np.float32)
            edges_3d.append((p1, p2))
        return edges_3d
    
    def project_edges(self, edges_3d, rvec, tvec, camera_matrix, dist_coeffs):
        projected_edges = []
        for p1, p2 in edges_3d:
            p1_img, _ = cv2.projectPoints(p1.reshape(1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
            p2_img, _ = cv2.projectPoints(p2.reshape(1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
            projected_edges.append((tuple(p1_img[0, 0]), tuple(p2_img[0, 0])))
        return projected_edges
    
    def render_edge_image(self, image_shape, projected_edges):
        canvas = np.ones(image_shape, dtype=np.uint8) * 255  # white background
        for p1, p2 in projected_edges:
            pt1 = tuple(map(int, p1))
            pt2 = tuple(map(int, p2))
            cv2.line(canvas, pt1, pt2, 0, 1, cv2.LINE_AA)  # antialiased black line
        return canvas


    def set_camera_matrix(self, camera_matrix, dist_coeffs=None):
        """
        Set the camera matrix.
        
        Args:
            camera_matrix (numpy.ndarray): Camera matrix.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1), dtype=np.float32)

    def crop_image(self, proj_edge):
        coords = np.column_stack(np.where(proj_edge < 255))
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # Crop the image
        return proj_edge[x_min:x_max+1, y_min:y_max+1]


    def set_image_shape(self, image_shape):
        self.image_shape = image_shape
    def render(self, filepath, translation, rotation):
        """
        Render a CAD model with the given translation and rotation.
        
        Args:
            filepath (str): Path to the DXF file.
            translation (list): Translation vector [tx, ty, tz].
            rotation (list): Tait-Bryan angles [roll, pitch, yaw] in radians.
        """
        # Load edges from DXF file
        edges = self.load_dxf_edges(filepath)
        edges_3d = self.to_3d_edges(edges)
        
        # Convert translation and rotation to appropriate formats
        self.tvec = np.array(translation, dtype=np.float32).reshape((3, 1))
        self.rvec = self.euler_to_rvec(rotation).reshape((3, 1))
        proj_edges = self.project_edges(edges_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        edge_img = self.render_edge_image(self.image_shape, proj_edges)
        # edge_img = self.crop_image(edge_img)
        return edge_img

