"""
Methods for extracting CAD data from a file.
"""
import ezdxf
import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p2) - np.array(p1))

def extract_outline_segments(dxf_path):
    """Extract line and arc segments from the DXF outline."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    segments = []  # Store (start, end) points of lines and polylines
    arc_segments = []  # Store arc data (center, radius, angles)
    total_length = 0  # Perimeter of the shape

    for entity in msp:
        if entity.dxftype() == 'LINE':
            p1, p2 = np.array(entity.dxf.start), np.array(entity.dxf.end)
            segments.append((p1, p2))
            total_length += distance(p1, p2)

        elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = np.array(entity.get_points())
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i+1]
                segments.append((p1, p2))
                total_length += distance(p1, p2)

        elif entity.dxftype() == 'ARC':
            center = np.array([entity.dxf.center.x, entity.dxf.center.y])
            radius = entity.dxf.radius
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)

            # Ensure correct angle direction
            if start_angle > end_angle:
                end_angle += 2 * np.pi

            arc_length = radius * (end_angle - start_angle)
            arc_segments.append((center, radius, start_angle, end_angle))
            total_length += arc_length

    return segments, arc_segments, total_length

def sample_outline(dxf_path, num_samples=100):
    """Sample evenly spaced points along the entire outline."""
    segments, arc_segments, total_length = extract_outline_segments(dxf_path)

    # Compute distance per sample
    step_size = total_length / num_samples
    sampled_points = []
    distance_accumulated = 0

    # Sample points along line segments
    for (p1, p2) in segments:
        segment_length = distance(p1, p2)
        num_points = int(segment_length / step_size)
        if num_points > 0:
            points = np.linspace(p1, p2, num_points, endpoint=False)
            sampled_points.extend(points)
            distance_accumulated += num_points * step_size

    # Sample points along arc segments
    for (center, radius, start_angle, end_angle) in arc_segments:
        arc_length = radius * (end_angle - start_angle)
        num_points = int(arc_length / step_size)
        angles = np.linspace(start_angle, end_angle, num_points, endpoint=False)
        points = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
        sampled_points.extend(points)
        distance_accumulated += num_points * step_size

    # Ensure exactly num_samples
    sampled_points = np.array(sampled_points)
    if len(sampled_points) > num_samples:
        sampled_points = sampled_points[:num_samples]

    return sampled_points

def center_points(pts):
    """
    Center the points around the origin.
    """
    center = np.mean(pts, axis=0)
    return pts - center

def get_cad_points(dxf_path, num_points=500):
    """
    Extract points from a DXF file and center them around the origin.
    
    Args:
        dxf_path (str): Path to the DXF file.
        num_points (int): Number of points to extract.
    
    Returns:
        numpy.ndarray: Centered points extracted from the DXF file.
    """
    sampled_points = sample_outline(dxf_path, num_samples=num_points)
    centered_points = center_points(sampled_points)
    return centered_points/1000 # Convert to meters