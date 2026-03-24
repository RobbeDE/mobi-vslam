import cv2
import numpy as np
import rerun as rr
from airo_typing import PointCloud
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap

# --- CONFIGURATION ---
MAP_SIZE_METERS = 30.0      # The map will cover 30x30 meters
MAP_SIZE_PIXELS = 600       # The map will be rendered at 600x600 pixels
RESOLUTION = MAP_SIZE_METERS / MAP_SIZE_PIXELS # meters per pixel
CAMERA_HEIGHT = 0.45       # The height of the ZED camera in meters

X_W_C = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]

    ],
    dtype=float,
)

def points_camera_to_world(points_camera: np.ndarray) -> np.ndarray:
    """
    Convert points from camera frame (X right, Y down, Z forward)
    to World frame (X right, Y forward, Z up).
    """
    assert points_camera.shape[1] == 3 # check that input is Nx3

    points_world = (X_W_C[:3, :3] @ points_camera.T).T  # Rotate points from camera frame to world frame

    # points_world[:, 0] = points_camera[:, 0]  # X -> X
    # points_world[:, 1] = points_camera[:, 2]  # Z -> Y
    # points_world[:, 2] = -points_camera[:, 1]  # -Y -> Z

    return points_world

def pose_camera_to_world(pose_camera: np.ndarray) -> np.ndarray:
    assert pose_camera.shape == (4, 4)

    pose_world = X_W_C @ pose_camera

    return pose_world

def load_spatial_map_from_npz(filepath: str) -> ZedSpatialMap:
    data = np.load(filepath)
    
    chunks_updated = data["chunks_updated"].tolist()
    chunks =[]
    
    for i in range(len(chunks_updated)):
        points = data[f"points_{i}"]
        colors = data[f"colors_{i}"] if f"colors_{i}" in data else None
        chunks.append(PointCloud(points=points, colors=colors))
        
    return ZedSpatialMap(chunks=chunks, chunks_updated=chunks_updated)



def visualize_occupancy_grid(spatial_map):
    # 1. Mark all cells as UNKNOWN by default (Gray background)
    grid_img = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), 128, dtype=np.uint8)

    # Extract full 3D points
    points = points_camera_to_world(spatial_map.full_pointcloud.points)

    if points is not None and points.shape[0] > 0:
        
        # Map horizontal axes (X and Y) to pixel coordinates
        # X maps to the image horizontal axis, Y maps to the image vertical axis (depth)
        px_all = ((MAP_SIZE_PIXELS / 2) + (points[:, 0] / RESOLUTION)).astype(int)
        py_all = ((MAP_SIZE_PIXELS / 2) - (points[:, 1] / RESOLUTION)).astype(int)

        # Filter out points that fall outside the 2D image boundaries
        valid_bounds = (px_all >= 0) & (px_all < MAP_SIZE_PIXELS) & (py_all >= 0) & (py_all < MAP_SIZE_PIXELS)
        
        px_valid = px_all[valid_bounds]
        py_valid = py_all[valid_bounds]
        z_valid = points[valid_bounds, 2]  # Extract corresponding Z (Height) values

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
                has_any_point = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=bool)
                has_remaining_point = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=bool)

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

    # Show text overlay
    cv2.putText(grid_img, f"Points: {spatial_map.size}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the image
    cv2.imshow("Occupancy Grid", grid_img)
    print("Map loaded. Press any key in the OpenCV window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(1)

if __name__ == "__main__":
    print("Loading spatial map...")
    spatial_map = load_spatial_map_from_npz("spatial_map5.npz")
    
    # --- 1. RERUN VISUALIZATION ---
    print("Starting Rerun...")
    rr.init("gert")
    rr.spawn(memory_limit="2GB")
    
    rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points = points_camera_to_world(spatial_map.full_pointcloud.points)
    if points.shape[0] > 0:
        rr.log(
            "World/spatial_map",
            rr.Points3D(
                positions=points,
                colors=spatial_map.full_pointcloud.colors,
            )
        )
    else:
        print("Warning: Spatial map is empty.")

    # --- 2. OPENCV GRID VISUALIZATION ---
    visualize_occupancy_grid(spatial_map)