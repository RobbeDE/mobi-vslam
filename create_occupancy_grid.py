from utils import *
from constants import *
import cv2

# --- CONFIGURATION ---
SPATIAL_MAP_NAME = "spatial_maps/spatial_map15.npz"  # The .npz file containing the saved spatial map data
OCCUPANCY_GRID_NAME = "occupancy_grids/new_occupancy_grid15.npz"  # The .npz file to save the occupancy grid data

if __name__ == "__main__":
    # Load spatial map and transform points to robot world coordinate frame
    spatial_map = load_spatial_map_from_npz(SPATIAL_MAP_NAME)
    points_world = points_Cw_to_Rw(spatial_map.full_pointcloud.points)

    # Create and save the occupancy grid
    occupancy_grid = world_to_occupancy_grid(points_world)
    save_occupancy_grid(occupancy_grid, OCCUPANCY_GRID_NAME)

    # Visualize the occupancy grid
    cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
    draw_occupancy_grid("Occupancy Grid", occupancy_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    