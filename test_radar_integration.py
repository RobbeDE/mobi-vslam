import threading
from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
from constants import *
from utils import *
import zmq

# ZeroMQ subscriber to receive radar tracks.
class TrackBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None

    def update(self, data):
        with self.lock:
            self.latest = data

    def get(self):
        with self.lock:
            return self.latest
        
def receiver_thread(buffer: TrackBuffer):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        msg = socket.recv_pyobj()  # blocking, OK here
        buffer.update(msg)

def filter_robot_track(tracks, robot_pose_world):
    """Return tuple of (filtered_tracks, robot_track) where robot_track is the track closest to the robot's pose."""
    filtered_tracks = []
    robot_track = None
    for track in tracks:
        track_pos = np.array(track['x'][:2])  # Assuming track['x'] is [x, y, z]
        robot_pos = robot_pose_world[:2, 3]  # Extract x, y from pose
        distance = np.linalg.norm(track_pos - robot_pos)
        if distance < 0.3:  # Threshold for considering a track as the robot's track
            robot_track = track
        else:
            filtered_tracks.append(track)
    return filtered_tracks, robot_track

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
            print(f"Translation pose (world): {pose_world[:3, 3]}")
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world)

            tracks = buffer.get()
            if tracks is not None:
                print(f"Received {len(tracks)} radar tracks")
                human_tracks, robot_track = filter_robot_track(tracks, pose_world)
                if robot_track is not None:
                    print(f"Identified robot track at position: {robot_track['x'][:2]}")
                for track in human_tracks:
                    track_pos = track['x'][:2]
                    print(f"Human track at position: {track_pos}")

            key = cv2.waitKey(10)

            if key == ord("q"):
                break