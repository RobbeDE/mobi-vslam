import threading
from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
from constants import *
from utils import *
import zmq
from datatypes import KalmanTrackProxy, TrackBuffer

        
def receiver_thread(buffer: TrackBuffer):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        track_dicts = socket.recv_pyobj()  # blocking, OK here
        track_proxies = [KalmanTrackProxy(track_dict) for track_dict in track_dicts]
        buffer.update(track_proxies)


# --- CONFIGURATION ---
AREA_FILE = "area_files/test15.area"
OCCUPANCY_GRID_FILE = "occupancy_grids/edited_occupancy_grid15.npz"


if __name__ == "__main__":

    tracking_params = Zed.TrackingParams(align_with_gravity=True, area_file_path = AREA_FILE, enable_localization_only=True)
    runtime_params = Zed.RuntimeParams(confidence_threshold=30)
    occupancy_grid = load_occupancy_grid(OCCUPANCY_GRID_FILE)

    # Start the ZMQ receiver thread to get radar tracks
    buffer = TrackBuffer()
    threading.Thread(
        target=receiver_thread,
        args=(buffer,),
        daemon=True
    ).start()

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_runtime_params=runtime_params,
        fps = 60,
        serial_number=31733653
    ) as zed:
        
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

            pose_world = pose_Cw_to_Rw(pose_matrix)
            tracks = buffer.get()
            
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world, tracks)
            
            if tracks is not None:
                print(f"Received {len(tracks)} radar tracks")
                human_tracks, robot_track = filter_robot_track(tracks, pose_world)
                if robot_track is not None:
                    print(f"Translation pose (world): {pose_world[:3, 3]}")
                    print(f"Identified robot track at position: {robot_track.x[:2]}")

            key = cv2.waitKey(10)

            if key == ord("q"):
                break