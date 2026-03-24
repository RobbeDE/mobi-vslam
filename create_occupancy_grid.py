import numpy as np
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
from datatypes import OccupancyGrid
from utils import *
from constants import *

# --- CONFIGURATION ---
SPATIAL_MAP_NAME = "spatial_map5.npz"  # The .npz file containing the saved spatial map data
OCCUPANCY_GRID_NAME = "occupancy_grid5.npz"  # The .npz file to save the occupancy grid data


def create_occupancy_grid(spatial_map: ZedSpatialMap) -> OccupancyGrid:
    # 1. Initialize a 1-Channel 2D array to 128 (Unknown)
    grid_data = np.full((MAP_SIZE_CELLS, MAP_SIZE_CELLS), 128, dtype=np.uint8)

    points = points_camera_to_world(spatial_map.full_pointcloud.points)

    if points.shape[0] > 0:
        px_all = ((MAP_SIZE_CELLS / 2) + (points[:, 0] / CELL_SIZE)).astype(int)
        py_all = ((MAP_SIZE_CELLS / 2) - (points[:, 1] / CELL_SIZE)).astype(int)

        valid_bounds = (px_all >= 0) & (px_all < MAP_SIZE_CELLS) & (py_all >= 0) & (py_all < MAP_SIZE_CELLS)
        
        px_valid = px_all[valid_bounds]
        py_valid = py_all[valid_bounds]
        z_valid = points[valid_bounds, 2]  

        if z_valid.shape[0] > 0:
            z_min_ground = -CAMERA_HEIGHT - GROUND_MARGIN
            z_max_ground = -CAMERA_HEIGHT + GROUND_MARGIN
            z_max_obstacle = -CAMERA_HEIGHT + MAX_OBSTACLE_HEIGHT

            valid_height = (z_valid > z_min_ground) & (z_valid <= z_max_obstacle)
            
            px_valid = px_valid[valid_height]
            py_valid = py_valid[valid_height]
            z_valid = z_valid[valid_height]

            if z_valid.shape[0] > 0:
                has_any_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)
                has_remaining_point = np.zeros((MAP_SIZE_CELLS, MAP_SIZE_CELLS), dtype=bool)

                has_any_point[py_valid, px_valid] = True
                
                is_remaining = (z_valid > z_max_ground)
                has_remaining_point[py_valid[is_remaining], px_valid[is_remaining]] = True

                is_free = has_any_point & ~has_remaining_point
                is_occupied = has_remaining_point

                # 2. Populate data with target specification values
                grid_data[is_free] = 255     # 255 = Free
                grid_data[is_occupied] = 0  # 0 = Occupied

    return OccupancyGrid(MAP_SIZE_METERS, CELL_SIZE, grid_data)




if __name__ == "__main__":
    spatial_map = load_spatial_map_from_npz(SPATIAL_MAP_NAME)
    OccupancyGrid = create_occupancy_grid(spatial_map)
    save_occupancy_grid(OccupancyGrid, OCCUPANCY_GRID_NAME)
    visualize_occupancy_grid(OccupancyGrid)
    