from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
import numpy as np
import rerun as rr
from utils import points_camera_to_world, pose_camera_to_world, save_spatial_map_to_npz, world_to_grid


# --- CONFIGURATION ---
MAP_SIZE_METERS = 50.0      # The map will cover 50x50 meters
MAP_SIZE_PIXELS = 600       # The map will be rendered at 600x600 pixels
RESOLUTION = MAP_SIZE_METERS / MAP_SIZE_PIXELS # meters per pixel
CAMERA_HEIGHT = 0.45      # The height of the ZED camera in meters
RERUN = False
OUTPUT_AREA_FILE = "test10.area"
OUTPUT_POINTCLOUD_FILE = "spatial_map10.npz"


def visualize_occupancy_grid(world_points, world_pose):
    # 1. Mark all cells as UNKNOWN by default (Gray background)
    grid_img = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), 128, dtype=np.uint8)

    if world_points.shape[0] > 0:
        
        # Map horizontal axes (X and Y) to pixel coordinates
        # X maps to the image horizontal axis, Y maps to the image vertical axis (depth)
        px_all = ((MAP_SIZE_PIXELS / 2) + (world_points[:, 0] / RESOLUTION)).astype(int)
        py_all = ((MAP_SIZE_PIXELS / 2) - (world_points[:, 1] / RESOLUTION)).astype(int)

        # Filter out points that fall outside the 2D image boundaries
        valid_bounds = (px_all >= 0) & (px_all < MAP_SIZE_PIXELS) & (py_all >= 0) & (py_all < MAP_SIZE_PIXELS)
        
        px_valid = px_all[valid_bounds]
        py_valid = py_all[valid_bounds]
        z_valid = world_points[valid_bounds, 2]  # Extract corresponding Z (Height) values

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

    # --- RENDER THE ROBOT POSE ---
    # Extract Robot Translation (X and Z)
    robot_x = world_pose[0, 3]
    robot_y = world_pose[1, 3]
    r_px, r_py = world_to_grid(robot_x, robot_y, RESOLUTION, MAP_SIZE_PIXELS)

    # Calculate Heading (Assume the camera looks along the +Z or -Z axis)
    # We project a point 0.5 meters in front of the camera to draw a line
    # (Note: depending on the exact ZED coordinate frame, forward might be -Z. 
    # We use +0.5 on Z here, adjust to -0.5 if the line points backward)
    forward_local = np.array([0, 0, 1.0, 1.0])
    forward_world = world_pose @ forward_local
    f_px, f_py = world_to_grid(forward_world[0], forward_world[1], RESOLUTION, MAP_SIZE_PIXELS)

    # Draw Robot Center (Red Circle)
    cv2.circle(grid_img, (r_px, r_py), 6, (0, 0, 255), -1)
    # Draw Robot Heading (Green Line)
    cv2.line(grid_img, (r_px, r_py), (f_px, f_py), (0, 255, 0), 2)

    # Show text overlay
    cv2.putText(grid_img, f"Points: {world_points.shape[0]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(grid_img, f"Pos: ({robot_x:.2f}, {robot_y:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the image
    cv2.imshow("Occupancy Grid", grid_img)

def main():

    tracking_params = Zed.TrackingParams(align_with_gravity=True, enable_2d_ground_mode=True)
    mapping_params = Zed.MappingParams(max_memory_usage = 2048)

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_mapping_params=mapping_params,
        serial_number=31733653
    ) as zed:
        
        print("mode: ",zed._zed_tracking_params.mode)
        print("sdk version: ", zed.camera.get_sdk_version())

        print("Starting real-time 2D occupancy grid mapping...")
        print("Press 'q' in the OpenCV window to exit.")

        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)

        if RERUN:
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
            zed._request_spatial_map_update()
            spatial_map = zed._retrieve_spatial_map()
            pose_matrix = zed._retrieve_camera_pose()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            world_points = points_camera_to_world(spatial_map.full_pointcloud.points)
            world_pose = pose_camera_to_world(pose_matrix)

            visualize_occupancy_grid(world_points, world_pose)

            if RERUN:
                rr.log(
                    "World/spatial_map",
                    rr.Points3D(
                        positions=world_points,
                        colors=spatial_map.full_pointcloud.colors,
                    ),
                )

            # Extract Rotation Matrix (3x3 top-left)
            translation_world = world_pose[:3, 3]
            rotation_world = world_pose[:3, :3]

            print(f"Translation pose (world): {translation_world}")

            if RERUN:
                # Log the transform.
                # This moves "World/Camera" (and its child "Pinhole") to the new location.
                rr.log("World/Camera", rr.Transform3D(translation=translation_world, mat3x3=rotation_world))

            key = cv2.waitKey(10)
            if key == ord("s"):
                zed.save_area_map(OUTPUT_AREA_FILE)
                save_spatial_map_to_npz(spatial_map, OUTPUT_POINTCLOUD_FILE)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()