from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
import cv2
import numpy as np
import cv2
import rerun as rr
from airo_typing import PointCloud

# --- CONFIGURATION ---
MAP_SIZE_METERS = 50.0      # The map will cover 50x50 meters
MAP_SIZE_PIXELS = 600       # The map will be rendered at 600x600 pixels
RESOLUTION = MAP_SIZE_METERS / MAP_SIZE_PIXELS # meters per pixel
CAMERA_HEIGHT = 1.425       # The height of the ZED camera in meters
OUTPUT_AREA_FILE = "test1.area"
OUTPUT_POINTCLOUD_FILE = "spatial_map.npz"

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

def visualize_occupancy_grid(spatial_map, pose):
    # --- RENDER THE 2D MAP ---
    # Create a blank black image (Occupancy Grid)
    grid_img = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)

    # Extract full 3D points
    points = spatial_map.full_pointcloud.points

    if points is not None and points.shape[0] > 0:
        # --- FILTER POINTS (CAMERA HEIGHT LOGIC) ---
        # Since align_with_gravity=True, the ZED Y-axis points DOWN.
        # Camera origin is at Y = 0.0m
        # The floor is at Y = +1.425m
        # We want obstacles 5cm above the floor (Y < 1.375) 
        # and up to 1.5m above the floor (Y > -0.075)
        
        y_max_ground = -CAMERA_HEIGHT + 0.1
        not_ground_mask = (points[:, 1] > y_max_ground)
        obstacles = points[not_ground_mask]

        if obstacles.shape[0] > 0:
            # Map X and Z coordinates to pixel indices efficiently using NumPy
            px = ((MAP_SIZE_PIXELS / 2) + (obstacles[:, 0] / RESOLUTION)).astype(int)
            pz = ((MAP_SIZE_PIXELS / 2) - (obstacles[:, 2] / RESOLUTION)).astype(int)

            # Filter out points that fall outside the image boundaries
            valid = (px >= 0) & (px < MAP_SIZE_PIXELS) & (pz >= 0) & (pz < MAP_SIZE_PIXELS)
            px = px[valid]
            pz = pz[valid]

            # Draw obstacles as white pixels on the grid
            grid_img[pz, px] = (255, 255, 255)

    # --- RENDER THE ROBOT POSE ---
    # Extract Robot Translation (X and Z)
    robot_x = pose[0, 3]
    robot_z = pose[2, 3]
    r_px, r_pz = world_to_grid(robot_x, robot_z)

    # Calculate Heading (Assume the camera looks along the +Z or -Z axis)
    # We project a point 0.5 meters in front of the camera to draw a line
    # (Note: depending on the exact ZED coordinate frame, forward might be -Z. 
    # We use +0.5 on Z here, adjust to -0.5 if the line points backward)
    forward_local = np.array([0, 0, 0.5, 1.0])
    forward_world = pose @ forward_local
    f_px, f_pz = world_to_grid(forward_world[0], forward_world[2])

    # Draw Robot Center (Red Circle)
    cv2.circle(grid_img, (r_px, r_pz), 6, (0, 0, 255), -1)
    # Draw Robot Heading (Green Line)
    cv2.line(grid_img, (r_px, r_pz), (f_px, f_pz), (0, 255, 0), 2)

    # Show text overlay
    cv2.putText(grid_img, f"Points: {spatial_map.size}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(grid_img, f"Pos: ({robot_x:.2f}, {robot_z:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the image
    cv2.imshow("Occupancy Grid", grid_img)

def world_to_grid(x, z):
    """
    Converts 3D world coordinates (X, Z) to 2D image coordinates (col, row).
    We map +Z to "Up" on the screen and +X to "Right".
    """
    # Center of the image is (0,0) in world coordinates
    px = int((MAP_SIZE_PIXELS / 2) + (x / RESOLUTION))
    # In OpenCV, row 0 is top. To make +Z go UP, we subtract from height/2
    pz = int((MAP_SIZE_PIXELS / 2) - (z / RESOLUTION))
    return px, pz

if __name__ == "__main__":

    tracking_params = Zed.TrackingParams(align_with_gravity=True, enable_2d_ground_mode=True, area_file_path = "test1.area")
    mapping_params = Zed.MappingParams(max_memory_usage = 4096)

    spatial_map = load_spatial_map_from_npz("spatial_map.npz")

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_mapping_params=mapping_params,
    ) as zed:
        
        print("Starting real-time 2D occupancy grid mapping...")
        print("Press 'q' in the OpenCV window to exit.")

        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)

        rr.init("gert")
        rr.spawn(memory_limit="2GB")

        # 1. Setup World Coordinate System (Z-Up)
        rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        rr.log(
            "World/Camera/Pinhole",
            rr.Pinhole(
                resolution=[1280, 720],  # Arbitrary resolution for visualization
                focal_length=700,  # Arbitrary FOV for visualization
            ),
            static=True,
        )

        while True:

            # Retrieve images and data from shared memory
            zed._grab_images()
            image = zed.get_rgb_image_as_int()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pose_matrix = zed._retrieve_camera_pose()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            visualize_occupancy_grid(spatial_map, pose_matrix)

            # If enabled, log spatial map to rerun
            full_pointcloud = spatial_map.full_pointcloud
            rr.log(
                "World/spatial_map",
                rr.Points3D(
                    positions=camera_to_rerun(full_pointcloud.points),
                    colors=full_pointcloud.colors,
                ),
            )

            # Visualize camera pose in Rerun

            # Extract Translation (first 3 rows, 4th column)
            translation = pose_matrix[:3, 3]

            # Extract Rotation Matrix (3x3 top-left)
            rotation_mat = pose_matrix[:3, :3]

            R_x_90 = np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
                dtype=float,
            )

            rotation_mat = rotation_mat @ R_x_90

            # Log the transform.
            # This moves "World/Camera" (and its child "Pinhole") to the new location.
            rr.log("World/Camera", rr.Transform3D(translation=translation, mat3x3=rotation_mat))

            key = cv2.waitKey(10)
            if key == ord("q"):
                break

        cv2.destroyAllWindows()