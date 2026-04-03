import cv2
import numpy as np
from airo_typing import PointCloud
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
from datatypes import OccupancyGrid
from constants import *
from typing import Optional
from transformations import *

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
    points_grid = points_Rw_2d_to_G(points_world, CELL_SIZE, MAP_SIZE_CELLS)
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
        draw_robot_pose_on_grid(pose_world, grid_img, occupancy_grid)
    cv2.imshow(window_name, grid_img)
    

def draw_robot_pose_on_grid(pose_Rw_3d: np.ndarray, grid_img: np.ndarray, occupancy_grid: OccupancyGrid) -> None:
    pose_Rw_2d = pose_to_xy_plane(pose_Rw_3d)
    pose_grid = pose_Rw_2d_to_G(pose_Rw_2d, occupancy_grid)
    robot_x = pose_grid[0, 2]
    robot_y = pose_grid[1, 2]

    forward_local = np.array([0.5, 0, 1.0])
    forward_world = pose_Rw_2d @ forward_local
    forward_grid = points_Rw_2d_to_G(forward_world, occupancy_grid.cell_size, occupancy_grid.map_size_cells)

    # Draw Robot Center (Red Circle)
    cv2.circle(grid_img, (int(robot_x), int(robot_y)), 4, (0, 0, 255), -1)
    # Draw Robot Heading (Green Line)
    cv2.line(grid_img, (int(robot_x), int(robot_y)), (int(forward_grid[0]), int(forward_grid[1])), (0, 255, 0), 1)



if __name__ == "__main__":
    pass
    