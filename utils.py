import cv2
import numpy as np
from airo_typing import PointCloud
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
from datatypes import OccupancyGrid
from constants import *
from typing import Optional

# Camera coordinate frame (X right, Y down, Z forward) to World coordinate frame (X forward, Y left, Z up)
X_C_W = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]

    ],
    dtype=float,
)

# Local robot coordinate frame (X forward, Y left, Z up) to World coordinate frame (X right, Y forward, Z up)
def X_W_R_2d(robot_pose_world: np.ndarray) -> np.ndarray:
    X_R_W = np.eye(3)
    X_R_W[:2, :2] = robot_pose_world[:2, :2]
    X_R_W[:2, 2] = robot_pose_world[:2, 3]
    return X_R_W
    

# Grid coordinate frame (X right, Y down) to World frame (X right, Y up) in meters
def X_W_G_2d(map_size_cells: int, cell_size: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, -int(map_size_cells/2) * cell_size],
            [0, -1, -int(map_size_cells/2) * cell_size],
            [0, 0, 1]
        ],
        dtype=float,
    )

# World coordinate frame (X right, Y up) to Grid frame (X right, Y down) in cells
def X_G_W_2d(map_size_cells: int) -> np.ndarray:
    return np.array(
        [
            [0, -1, int(map_size_cells/2)],
            [-1, 0, int(map_size_cells/2)],
            [0, 0, 1]
        ],
        dtype=float,
    )

def R_2d(theta: float) -> np.ndarray:
    """
    Create a 2D rotation matrix for a given angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def points_camera_to_world(points_camera: np.ndarray) -> np.ndarray:
    """
    Convert points from camera frame (X right, Y down, Z forward)
    to World frame (X forward, Y left, Z up).
    """

    return (np.linalg.inv(X_C_W[:3, :3]) @ points_camera.T).T # Transposes are necessary for correct shapes in matrix multiplication

def pose_camera_to_world(pose_camera: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transformation matrix representing the robot's pose in the camera frame
    to the world frame.
    """

    return np.linalg.inv(X_C_W) @ pose_camera

def pose_world_to_grid(pose_world: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transformation matrix representing the robot's pose in the world frame
    to a pose in the grid frame.
    """

    # First, we convert the world pose to a 2D pose (ignoring Z and roll/pitch)
    pose_world_2d = np.eye(3)
    pose_world_2d[:2, :2] = pose_world[:2, :2]
    pose_world_2d[:2, 2] = pose_world[:2, 3]

    return X_G_W_2d(grid.map_size_cells) @ pose_world_2d 

def points_world_to_grid(points_world: np.ndarray, cell_size: float, map_size_cells: int) -> np.ndarray:
    """
    Convert world coordinates (x, y) in meters to grid coordinates (cx, cy) in cells.
    """

    # Convert to homogeneous coordinates
    points = np.column_stack([points_world[:, :2], np.ones(len(points_world))])
    
    # Apply transformation matrix X_G_W_2d
    transform = X_G_W_2d(map_size_cells)
    transformed = (transform @ points.T).T
    
    # Return only x and y coordinates (non-homogeneous)
    return transformed[:, :2]

def save_spatial_map_to_npz(spatial_map: ZedSpatialMap, filepath: str):
    data_dict = {}
    
    # Save the chunks updated list
    data_dict["chunks_updated"] = np.array(spatial_map.chunks_updated)
    
    # Save each chunk's points and colors dynamically
    for i, chunk in enumerate(spatial_map.chunks):
        data_dict[f"points_{i}"] = chunk.points
        if chunk.colors is not None:
            data_dict[f"colors_{i}"] = chunk.colors
            
    np.savez_compressed(filepath, **data_dict)
    print(f"Saved spatial map arrays to {filepath}")

def load_spatial_map_from_npz(filepath: str) -> ZedSpatialMap:
    data = np.load(filepath)
    
    chunks_updated = data["chunks_updated"].tolist()
    chunks =[]
    
    # Iterate based on how many chunks_updated booleans there are
    for i in range(len(chunks_updated)):
        points = data[f"points_{i}"]
        # Check if colors exist for this chunk in the npz file
        colors = data[f"colors_{i}"] if f"colors_{i}" in data else None
        
        chunks.append(PointCloud(points=points, colors=colors))
        
    return ZedSpatialMap(chunks=chunks, chunks_updated=chunks_updated)

def save_occupancy_grid(occupancy_grid: OccupancyGrid, filename: str):
    np.savez_compressed(filename, 
                        map_size=occupancy_grid.map_size, 
                        cell_size=occupancy_grid.cell_size, 
                        grid=occupancy_grid.grid)
    print(f"Occupancy grid saved to {filename}")

def load_occupancy_grid(filepath: str) -> OccupancyGrid:
    """
    Loads an occupancy grid and its metadata from an .npz file.
    """
    if not filepath.endswith('.npz'):
        raise ValueError("Please provide an .npz file, which contains the map_size and cell_size metadata.")
        
    data = np.load(filepath)
    
    # Extract data (convert scalars back to standard Python floats)
    grid_data = data["grid"]
    map_size = float(data["map_size"])
    cell_size = float(data["cell_size"])
    
    return OccupancyGrid(map_size, cell_size, grid_data)


def world_to_occupancy_grid(points_world: np.ndarray) -> OccupancyGrid:
    # 1. Mark all cells as UNKNOWN by default (Gray background)
    grid_img = np.full((MAP_SIZE_CELLS, MAP_SIZE_CELLS, 3), 128, dtype=np.uint8)

    if points_world.shape[0] == 0:
        return OccupancyGrid(MAP_SIZE_METERS, CELL_SIZE, grid_img)

    # 2. Convert world coordinates to grid coordinates
    points_grid = points_world_to_grid(points_world, CELL_SIZE, MAP_SIZE_CELLS)
    px_all = points_grid[:, 0].astype(int)
    py_all = points_grid[:, 1].astype(int)
    z_all = points_world[:, 2]

    # 3. Filter points within grid bounds
    valid_bounds = (px_all >= 0) & (px_all < MAP_SIZE_CELLS) & (py_all >= 0) & (py_all < MAP_SIZE_CELLS)
    px_valid = px_all[valid_bounds]
    py_valid = py_all[valid_bounds]
    z_valid = z_all[valid_bounds]

    if z_valid.shape[0] == 0:
        return OccupancyGrid(MAP_SIZE_METERS, CELL_SIZE, grid_img)

    # 4. Filter points by height
    z_min_ground = -CAMERA_HEIGHT - 0.25
    z_max_ground = -CAMERA_HEIGHT + 0.25
    z_max_obstacle = 0.5
    
    valid_height = (z_valid > z_min_ground) & (z_valid <= z_max_obstacle)
    px_valid = px_valid[valid_height]
    py_valid = py_valid[valid_height]
    z_valid = z_valid[valid_height]

    if z_valid.shape[0] == 0:
        return OccupancyGrid(MAP_SIZE_METERS, CELL_SIZE, grid_img)

    # 5. Create boolean grids for cell status
    has_any_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)
    has_remaining_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)

    has_any_point[py_valid, px_valid] = True
    
    is_remaining = (z_valid > z_max_ground)
    has_remaining_point[py_valid[is_remaining], px_valid[is_remaining]] = True

    # 6. Mark free and occupied cells
    is_free = has_any_point & ~has_remaining_point
    is_occupied = has_remaining_point

    grid_img[is_free] = (255, 255, 255)      # White for free cells
    grid_img[is_occupied] = (0, 0, 0)        # Black for occupied cells

    return OccupancyGrid(MAP_SIZE_METERS, CELL_SIZE, grid_img)

def draw_occupancy_grid(window_name: str, occupancy_grid: OccupancyGrid, pose_world: Optional[np.ndarray] = None):
    grid_img = occupancy_grid.grid.copy()

    if pose_world is not None:
        draw_robot_pose_on_grid(pose_world, grid_img, occupancy_grid.cell_size, occupancy_grid.map_size_cells)
    cv2.imshow(window_name, grid_img)
        

def theta_xy(R: np.ndarray):
    return np.arctan2(R[1, 0], R[0, 0])
    

def draw_robot_pose_on_grid(pose_world: np.ndarray, grid_img: np.ndarray, cell_size: float, map_size_cells: int) -> None:
    # --- RENDER THE ROBOT POSE ---
    # Extract Robot Translation (X and Z)
    robot_x = pose_world[0, 3]
    robot_y = pose_world[1, 3]
    r_px, r_py = coordinate_world_to_grid(robot_x, robot_y, cell_size, map_size_cells)

    # Calculate Heading (Assume the camera looks along the +Z axis)
    # We project a point 0.5 meters in front of the camera to draw a line
    # (Note: depending on the exact ZED coordinate frame, forward might be -Z. 
    # We use +0.5 on Z here, adjust to -0.5 if the line points backward)
    forward_local = np.array([0, 0, 0.5, 1.0])
    forward_world = pose_world @ forward_local
    f_px, f_py = coordinate_world_to_grid(forward_world[0], forward_world[1], cell_size, map_size_cells)

    # Draw Robot Center (Red Circle)
    cv2.circle(grid_img, (r_px, r_py), 4, (0, 0, 255), -1)
    # Draw Robot Heading (Green Line)
    cv2.line(grid_img, (r_px, r_py), (f_px, f_py), (0, 255, 0), 1)



if __name__ == "__main__":
    robot_pose_world = np.array([
        [0, -1, 0, 1],  # 90 degree rotation
        [1, 0, 0, 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    print("Robot Pose in World Frame:\n", robot_pose_world)
    print("Robot Pose in Grid Frame:\n", X_W_R(robot_pose_world))