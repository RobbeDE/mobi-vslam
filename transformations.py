import numpy as np
from datatypes import OccupancyGrid
from constants import *
from transformations import *

# ---------------------------------------
# RAW TRANSFORMATION MATRICES
# ---------------------------------------

# (local/world) Camera coordinate frame (X right, Y down, Z forward) to (local/world) Robot coordinate frame (X forward, Y left, Z up)
X_C_R = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, -0.3],
        [0, 0, 0, 1]

    ],
    dtype=float,
)

# Local robot coordinate frame (X forward, Y left, Z up) to World robot coordinate frame in 2D
def X_Rw_R_2d(robot_pose_Rw_2d: np.ndarray) -> np.ndarray:
    X_R_W = np.eye(3)
    X_R_W[:2, :2] = robot_pose_Rw_2d[:2, :2]
    X_R_W[:2, 2] = robot_pose_Rw_2d[:2, 2]
    return X_R_W
    
# Grid coordinate frame (X right, Y down) to World robot frame (X up, Y left) in meters
def X_Rw_G_2d(map_size_cells: int, cell_size: float) -> np.ndarray:
    # Origin of Occupancy Grid frame in the world frame
    translation = np.array(
        [
            [1, 0, int(map_size_cells/2) * cell_size],
            [0, 1, int(map_size_cells/2) * cell_size],
            [0, 0, 1]
        ],
        dtype=float,
    )
    # Axes of Occupancy Grid frame in the world frame 
    # Columns are the basis vectors of the occupancy grid frame, rows are basis vectors of the world frame.
    rotation = np.array(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ],
        dtype=float,
    )
    # Scale from cells to meters by multiplying by cell_size
    scale = np.array(
        [
            [cell_size, 0, 0],
            [0, cell_size, 0],
            [0, 0, 1]
        ],
        dtype=float,
    )
    return translation @ rotation @ scale

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def discretize_pose_2d(pose_grid: np.ndarray) -> np.ndarray:
    """
    Discretize the pose in the grid frame.
    Note that the type may still be a float as all elements in numpy arrays have the same type.
    """
    pose_grid[0, 2] = int(pose_grid[0, 2])
    pose_grid[1, 2] = int(pose_grid[1, 2])
    return pose_grid

def discretize_points_2d(points_grid: np.ndarray) -> np.ndarray:
    """
    Discretize the points in the grid frame.
    Note that the type may still be a float as all elements in numpy arrays have the same type.
    """
    points_grid[:, 0] = points_grid[:, 0].astype(int)
    points_grid[:, 1] = points_grid[:, 1].astype(int)
    return points_grid


def angle_to_R(theta: float) -> np.ndarray:
    """
    Create a 2D rotation matrix for a given angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def normalize_angle(theta: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

def R_to_angle(R: np.ndarray) -> float:
    """
    Extract rotation angle (in radians) from a 2x2 rotation matrix.
    """
    return normalize_angle(np.arctan2(R[1, 0], R[0, 0]))

def pose_to_xy_plane(pose: np.ndarray) -> np.ndarray:
    pose_2d = np.eye(3)
    pose_2d[:2, :2] = pose[:2, :2]
    pose_2d[:2, 2] = pose[:2, 3]
    return pose_2d


# ---------------------------------------
# HIGH-LEVEL TRANSFORMATION FUNCTIONS
# ---------------------------------------

# ------- 3D transformations between camera world frame and robot world frame -------

def points_Cw_to_Rw(points_camera: np.ndarray) -> np.ndarray:
    """
    Convert points from camera world frame (X right, Y down, Z forward)
    to Robot world frame (X forward, Y left, Z up).
    """

    return (np.linalg.inv(X_C_R[:3, :3]) @ points_camera.T).T # Transposes are necessary for correct shapes in matrix multiplication

def pose_Cw_to_Rw(pose_camera: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transformation matrix representing the robot's pose in the camera world frame
    to the robot world frame.
    """

    return np.linalg.inv(X_C_R) @ pose_camera @ X_C_R

# ------- 2D transformations between robot world frame and grid frame -------

def pose_Rw_2d_to_G(pose_Rw_2d: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transformation matrix representing the robot's pose in the robot world frame
    to a pose in the grid frame.
    """

    return discretize_pose_2d(np.linalg.inv(X_Rw_G_2d(grid.map_size_cells, grid.cell_size)) @ pose_Rw_2d)

def pose_G_to_Rw_2d(pose_G: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    """
    Convert a 3x3 homogeneous transformation matrix representing the robot's pose in the grid frame
    to a 4x4 homogeneous transformation matrix in the robot world frame.
    """

    return X_Rw_G_2d(grid.map_size_cells, grid.cell_size) @ pose_G

def points_Rw_2d_to_G(points_Rw_2d: np.ndarray, cell_size: float, map_size_cells: int) -> np.ndarray:
    """
    Convert robot world coordinates (x, y) in meters to grid coordinates (cx, cy) in cells.
    Handles both 1D arrays (single point) and 2D arrays (multiple points).
    """
    
    # Track if input was 1D
    is_1d = points_Rw_2d.ndim == 1
    
    # Handle 1D input (single point)
    if is_1d:
        points_Rw_2d = points_Rw_2d.reshape(1, -1)
    
    # Convert to homogeneous coordinates
    points = np.column_stack([points_Rw_2d[:, :2], np.ones(len(points_Rw_2d))])
    
    # Apply transformation matrix X_G_Rw_2d
    transform = np.linalg.inv(X_Rw_G_2d(map_size_cells, cell_size))
    transformed = (transform @ points.T).T
    
    # Return discretized x and y coordinates (non-homogeneous)
    result = discretize_points_2d(transformed[:, :2])
    
    # Return as 1D if input was 1D
    return result[0] if is_1d else result

def points_G_to_Rw_2d(points_grid: np.ndarray, cell_size: float, map_size_cells: int) -> np.ndarray:
    """
    Convert grid coordinates (cx, cy) in cells to robot world coordinates (x, y) in meters.
    Handles both 1D arrays (single point) and 2D arrays (multiple points).
    """
    
    # Track if input was 1D
    is_1d = points_grid.ndim == 1
    
    # Handle 1D input (single point)
    if is_1d:
        points_grid = points_grid.reshape(1, -1)
    
    # Convert to homogeneous coordinates
    points = np.column_stack([points_grid[:, :2], np.ones(len(points_grid))])
    
    # Apply transformation matrix X_Rw_G_2d
    transform = X_Rw_G_2d(map_size_cells, cell_size)
    transformed = (transform @ points.T).T
    
    # Return x and y coordinates (non-homogeneous)
    result = transformed[:, :2]
    
    # Return as 1D if input was 1D
    return result[0] if is_1d else result

# ------- 2D transformations between local robot frame and robot world frame -------

def pose_Rw_2d_to_R(robot_pose_Rw_2d: np.ndarray, target_pose_Rw_2d: np.ndarray) -> np.ndarray:
    """
    Convert a pose in the robot world frame to the robot's local frame.
    """

    return np.linalg.inv(X_Rw_R_2d(robot_pose_Rw_2d)) @ target_pose_Rw_2d

def points_Rw_2d_to_R(robot_pose_Rw_2d: np.ndarray, points_Rw_2d: np.ndarray) -> np.ndarray:
    """
    Convert points in the robot world frame to the robot's local frame.
    """

    # Track if input was 1D
    is_1d = points_Rw_2d.ndim == 1

    # Handle 1D input (single point)
    if is_1d:
        points_Rw_2d = points_Rw_2d.reshape(1, -1)

    # Convert to homogeneous coordinates
    points = np.column_stack([points_Rw_2d[:, :2], np.ones(len(points_Rw_2d))])

    # Apply transformation matrix X_R_Rw_2d
    transformed = (np.linalg.inv(X_Rw_R_2d(robot_pose_Rw_2d)) @ points.T).T

    # Return x and y coordinates (non-homogeneous)
    result = transformed[:, :2]
    # Return as 1D if input was 1D
    return result[0] if is_1d else result

