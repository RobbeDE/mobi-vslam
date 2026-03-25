from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
import numpy as np
import rerun as rr
from utils import pose_camera_to_world, save_spatial_map_to_npz, world_to_occupancy_grid, draw_occupancy_grid, points_camera_to_world
from constants import *


# --- CONFIGURATION ---
RERUN = False
OUTPUT_AREA_FILE = "area_files/test15.area"
OUTPUT_POINTCLOUD_FILE = "spatial_maps/spatial_map15.npz"

if __name__ == "__main__":
    tracking_params = Zed.TrackingParams(align_with_gravity=True)
    mapping_params = Zed.MappingParams(max_memory_usage = 2048)
    runtime_params = Zed.RuntimeParams(confidence_threshold=30)

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        fps = 60,
        camera_tracking_params=tracking_params,
        camera_mapping_params=mapping_params,
        camera_runtime_params=runtime_params,
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

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            points_world = points_camera_to_world(spatial_map.full_pointcloud.points)
            pose_world = pose_camera_to_world(pose_matrix)
            print(f"Translation pose (world): {pose_world[:3, 3]}")

            occupancy_grid = world_to_occupancy_grid(points_world)
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world)

            if RERUN:
                rr.log(
                    "World/spatial_map",
                    rr.Points3D(
                        positions=points_world,
                        colors=spatial_map.full_pointcloud.colors,
                    ),
                )

            if RERUN:
                # Log the transform.
                # This moves "World/Camera" (and its child "Pinhole") to the new location.
                rr.log("World/Camera", rr.Transform3D(translation=pose_world[:3, 3], mat3x3=pose_world[:3, :3]))

            key = cv2.waitKey(10)
            if key == ord("s"):
                zed.save_area_map(OUTPUT_AREA_FILE)
                save_spatial_map_to_npz(spatial_map, OUTPUT_POINTCLOUD_FILE)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()