from airo_typing import PointCloud
import numpy as np
from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
import rerun as rr


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

def camera_to_rerun(points_cam: np.ndarray) -> np.ndarray:
    """
    Convert points from camera frame (X right, Y down, Z forward)
    to Rerun frame (X right, Y forward, Z up).
    """
    assert points_cam.shape[1] == 3

    points_rerun = np.empty_like(points_cam)

    points_rerun[:, 0] = points_cam[:, 0]  # X -> X
    points_rerun[:, 1] = points_cam[:, 2]  # Z -> Y
    points_rerun[:, 2] = -points_cam[:, 1]  # -Y -> Z

    return points_rerun

def render_saved_map_with_rerun(spatial_map: ZedSpatialMap):
    """
    Spawns a Rerun viewer and logs the loaded ZedSpatialMap.
    """
    full_pointcloud = spatial_map.full_pointcloud
    
    if len(full_pointcloud.points) == 0:
        print("The point cloud is empty. Nothing to render.")
        return

    print(f"Preparing to render {len(full_pointcloud.points)} points in Rerun...")

    # 1. Initialize Rerun and spawn the viewer
    rr.init("zed_saved_spatial_map_viewer")
    rr.spawn(memory_limit="2GB")

    # 2. Setup World Coordinate System (matches your real-time script)
    rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Optional: Log a fixed camera origin just to have a reference point in the viewer
    rr.log(
        "World/Camera/Pinhole",
        rr.Pinhole(
            resolution=[1280, 720],
            focal_length=700,
        ),
        static=True,
    )

    # 3. Log the point cloud data
    # Rerun natively handles the numpy arrays and uint8 [0, 255] colors, 
    # so no extra color conversions are necessary.
    rr.log(
        "World/spatial_map",
        rr.Points3D(
            positions=camera_to_rerun(full_pointcloud.points),
            colors=full_pointcloud.colors,
        ),
    )
    
    print("Map successfully loaded into Rerun!")


if __name__ == "__main__":
    # Point this to the file you saved using save_spatial_map_to_npz()
    OUTPUT_POINTCLOUD_FILE = "spatial_map6.npz" 
    
    try:
        # Load the saved chunks back into the ZedSpatialMap dataclass
        loaded_spatial_map = load_spatial_map_from_npz(OUTPUT_POINTCLOUD_FILE)
        
        # Render the map in Rerun
        render_saved_map_with_rerun(loaded_spatial_map)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{OUTPUT_POINTCLOUD_FILE}'. Make sure you saved it first.")