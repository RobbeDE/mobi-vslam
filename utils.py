import cv2
import numpy as np
from airo_typing import PointCloud
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
from datatypes import OccupancyGrid

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

X_W_C = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]

    ],
    dtype=float,
)

def spatial_map_to_occupancy_grid()

def visualize_occupancy_grid(occupancy_grid: OccupancyGrid):

    print("Visualizing occupancy grid... (Press 'q' or 'Esc' to exit)")

    cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Occupancy Grid", 800, 800)

    while True:
        display_img = occupancy_grid.grid.copy()
        cv2.imshow("Occupancy Grid", display_img)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == 27 or key == ord('q'):  
            break

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

def world_to_grid(x: float, y: float, cell_size: float, map_size_pixels: int) -> tuple[int, int]:
    """
    Converts 3D world coordinates (X, Y) to 2D image coordinates (col, row).
    We map +Y to "Up" on the screen and +X to "Right".
    """
    # Center of the image is (0,0) in world coordinates
    px = int((map_size_pixels / 2) + (x / cell_size))
    # In OpenCV, row 0 is top. To make +Y go UP, we subtract from height/2
    py = int((map_size_pixels / 2) - (y / cell_size))
    return px, py
