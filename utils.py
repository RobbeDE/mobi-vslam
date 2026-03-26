import cv2
import numpy as np
from airo_typing import PointCloud
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
from datatypes import OccupancyGrid
from constants import *
from typing import Optional

X_W_C = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]

    ],
    dtype=float,
)

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



def world_to_occupancy_grid(points_world: np.ndarray):
    # 1. Mark all cells as UNKNOWN by default (Gray background)
    grid_img = np.full((MAP_SIZE_CELLS, MAP_SIZE_CELLS, 3), 128, dtype=np.uint8)

    if points_world.shape[0] > 0:
        
        # Map horizontal axes (X and Y) to pixel coordinates
        # X maps to the image horizontal axis, Y maps to the image vertical axis (depth)
        px_all = ((MAP_SIZE_CELLS / 2) + (points_world[:, 0] / CELL_SIZE)).astype(int)
        py_all = ((MAP_SIZE_CELLS / 2) - (points_world[:, 1] / CELL_SIZE)).astype(int)

        px_all, py_all = coordinate_world_to_grid(points_world[:, 0], points_world[:, 1], CELL_SIZE, MAP_SIZE_CELLS)

        # Filter out points that fall outside the 2D image boundaries
        valid_bounds = (px_all >= 0) & (px_all < MAP_SIZE_CELLS) & (py_all >= 0) & (py_all < MAP_SIZE_CELLS)
        
        px_valid = px_all[valid_bounds]
        py_valid = py_all[valid_bounds]
        z_valid = points_world[valid_bounds, 2]  # Extract corresponding Z (Height) values

        if z_valid.shape[0] > 0:
            # Define the height thresholds for a Z-UP coordinate system
            # Assuming camera is at Z=0 and ground is at Z = -CAMERA_HEIGHT
            # (If your world frame already puts ground at Z=0, change these to around 0)
            z_min_ground = -CAMERA_HEIGHT - 0.25
            z_max_ground = -CAMERA_HEIGHT + 0.25
            z_max_obstacle = 0.5 # Upper limit for obstacles (0.5m above camera)

            # Filter points to only consider the vertical range of interest.
            # (In Z-up, higher values mean higher physical height)
            valid_height = (z_valid > z_min_ground) & (z_valid <= z_max_obstacle)
            
            px_valid = px_valid[valid_height]
            py_valid = py_valid[valid_height]
            z_valid = z_valid[valid_height]

            if z_valid.shape[0] > 0:
                # 2. Create boolean grids to track cell status
                has_any_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)
                has_remaining_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)

                # Mark pixels that contain AT LEAST ONE valid point (Ground OR Obstacle)
                has_any_point[py_valid, px_valid] = True
                
                # 3. Filter the ground points to identify the "remaining" points
                # (Anything above the ground threshold is a remaining obstacle point)
                is_remaining = (z_valid > z_max_ground)
                
                # Mark pixels that contain at least one remaining (non-ground) point
                has_remaining_point[py_valid[is_remaining], px_valid[is_remaining]] = True

                # 4. Apply the new logic:
                # - Free: The cell had points, but NO points left after filtering ground
                is_free = has_any_point & ~has_remaining_point
                
                # - Occupied: The cell still has points remaining after filtering ground
                is_occupied = has_remaining_point

                # 5. Draw the pixels on the grid
                grid_img[is_free] = (255, 255, 255)  # White for free cells
                grid_img[is_occupied] = (0, 0, 0)    # Black for occupied cells
                # (Cells where has_any_point is False simply remain Gray/Unknown)

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

def points_camera_to_world(points_camera: np.ndarray) -> np.ndarray:
    """
    Convert points from camera frame (X right, Y down, Z forward)
    to World frame (X right, Y forward, Z up).
    """
    assert points_camera.shape[1] == 3
    points_world = (X_W_C[:3, :3] @ points_camera.T).T  
    return points_world

def pose_camera_to_world(pose_camera: np.ndarray) -> np.ndarray:
    assert pose_camera.shape == (4, 4)
    pose_world = X_W_C @ pose_camera
    return pose_world

def pose_world_to_grid(pose_world: np.ndarray, grid: OccupancyGrid) -> tuple[int, int, float]:
    """
    Convert a 4x4 homogeneous transformation matrix representing the robot's pose in the world frame
    to grid coordinates (cell_x, cell_y) and heading (theta).
    """
    robot_x = pose_world[0, 3]
    robot_y = pose_world[1, 3]
    cell_x, cell_y = coordinate_world_to_grid(robot_x, robot_y, grid.cell_size, grid.map_size_cells)
    
    # Extract rotation and compute heading
    R = pose_world[:3, :3]
    heading = theta_xy(R) + np.pi/2  # Adjust if needed based on how your robot's forward direction maps to the grid

    return cell_x, cell_y, heading

def coordinate_world_to_grid(x: float | np.ndarray, y: float | np.ndarray, cell_size: float, map_size_cells: int) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Converts 3D world coordinates (X, Y) to 2D image coordinates (col, row).

    Supports both scalar inputs and NumPy arrays.

    We map +Y to "Up" on the screen and +X to "Right".
    """

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    cx = (map_size_cells / 2) + (x_arr / cell_size)
    cy = (map_size_cells / 2) - (y_arr / cell_size)

    # Convert to integer grid indices
    cx = cx.astype(int)
    cy = cy.astype(int)

    # If inputs were scalars, return scalars
    if np.isscalar(x) and np.isscalar(y):
        return int(cx), int(cy)

    return cx, cy
