from airo_camera_toolkit.cameras.zed.zed import ZedSpatialMap
import rerun as rr
from utils import points_camera_to_world, load_spatial_map_from_npz

FILE_PATH = "spatial_map5.npz"

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
            positions=points_camera_to_world(full_pointcloud.points),
            colors=full_pointcloud.colors,
        ),
    )
    
    print("Map successfully loaded into Rerun!")


if __name__ == "__main__":
    spatial_map = load_spatial_map_from_npz(FILE_PATH)
    render_saved_map_with_rerun(spatial_map)