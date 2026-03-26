from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
from constants import *
from utils import *

# --- CONFIGURATION ---
AREA_FILE = "area_files/test13.area"
OCCUPANCY_GRID_FILE = "occupancy_grids/occupancy_grid13.npz"


if __name__ == "__main__":

    tracking_params = Zed.TrackingParams(align_with_gravity=True, area_file_path = AREA_FILE, enable_localization_only=True)
    runtime_params = Zed.RuntimeParams(confidence_threshold=30)
    occupancy_grid = load_occupancy_grid(OCCUPANCY_GRID_FILE)

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_runtime_params=runtime_params,
        fps = 60,
        serial_number=31733653
    ) as zed:
        
        print("Starting real-time 2D occupancy grid mapping...")
        print("Press 'q' in the OpenCV window to exit.")

        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)

        while True:
            zed._grab_images()
            image = zed.get_rgb_image_as_int()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pose_matrix = zed._retrieve_camera_pose()

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            pose_world = pose_camera_to_world(pose_matrix)
            print(f"Translation pose (world): {pose_world[:3, 3]}")
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world)

            key = cv2.waitKey(10)

            if key == ord("q"):
                break