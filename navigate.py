import threading
import numpy as np
from loguru import logger
from datatypes import KalmanTrackProxy, TrackBuffer
from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
from constants import *
from utils import *
import zmq
import time
import numpy as np
from loguru import logger
from airo_robots.drives.hardware.kelo_robile import KELORobile
from RiskAStar import RiskAStar
from datatypes import OccupancyGrid
from constants import *
from utils import *

robot_pose_world = np.eye(4)  # 4x4 homogeneous transformation matrix
robot_pose_lock = threading.Lock()  # to protect access to robot_pose across threads


class MobiNavigator:
    MAX_LIN_SPEED = 0.2      # m/s (tune)
    MAX_ANG_SPEED = np.pi/8. # rad/s (tune)
    SMOOTHING_ALPHA = 0.95   # for exponential smoothing of commanded velocity
    DRIVE_LOOP_HZ = 20.0     # control loop frequency
    DRIVE_LOOP_PERIOD = 1.0 / DRIVE_LOOP_HZ
    GOAL_TOLERANCE = 0.2     # meters

    def __init__(self, occupancy_grid: OccupancyGrid, track_buffer: TrackBuffer):
        self.kelo = None
        self.occupancy_grid = occupancy_grid
        self.track_buffer = track_buffer

        self._nav_thread = None
        self._stop_event = threading.Event()

        self.current_goal_pose_Rw = None
        self.goal_pose_lock = threading.Lock()

        self.frozen = True # Start in frozen state until first goal is set

        self.waypoints =[]
        self.current_waypoint_index = 0
        self.waypoints_lock = threading.Lock()

    def connect_to_kelo_safe(self) -> bool:
        try:
            self.kelo = KELORobile("10.42.0.1")
            logger.success("KELO connected ✅")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to KELO: {e}")
            return False

    def _get_linear_drive_commands(
            self, 
            prev_cmd: tuple[float, float, float], 
            robot_pose_Rw_2d: np.ndarray, 
            goal_pose_Rw_2d: np.ndarray
            ) -> tuple[float, float, float]:

            goal_pose_R_2d = pose_Rw_2d_to_R(robot_pose_Rw_2d, goal_pose_Rw_2d)

            # Calculate linear error and angular error in local robot frame R
            error_linear_R = goal_pose_R_2d[:2, 2]
            distance = np.linalg.norm(error_linear_R)
            error_angular_R = R_to_angle(goal_pose_R_2d[:2, :2])

            v_R = error_linear_R / distance * self.MAX_LIN_SPEED if distance > 0 else np.zeros(2)
            w_R = np.clip(error_angular_R, -self.MAX_ANG_SPEED, self.MAX_ANG_SPEED)

            logger.info(f"Robot pose: {robot_pose_Rw_2d}")
            logger.info(f"Goal pose: {goal_pose_Rw_2d}")
            logger.info(f"Linear error (R): {error_linear_R}, Angular error (R): {np.degrees(error_angular_R):.1f} deg")

            # smoothing (exponential)
            vx = self.SMOOTHING_ALPHA * v_R[0] + (1 - self.SMOOTHING_ALPHA) * prev_cmd[0]
            vy = self.SMOOTHING_ALPHA * v_R[1] + (1 - self.SMOOTHING_ALPHA) * prev_cmd[1]
            wz = self.SMOOTHING_ALPHA * w_R + (1 - self.SMOOTHING_ALPHA) * prev_cmd[2]

            return vx, vy, wz

    def _create_waypoint_poses(self, waypoints_coordinates_G: np.ndarray) -> list[np.ndarray]:
        waypoints_coordinates_Rw = points_G_to_Rw_2d(waypoints_coordinates_G, self.occupancy_grid.cell_size, self.occupancy_grid.map_size_cells)
        waypoints_poses = []
        lookahead = 8  # number of future points to average

        N = len(waypoints_coordinates_Rw)

        for i in range(N):
            waypoint_pose = np.eye(3)

            # Position
            p = waypoints_coordinates_Rw[i]
            waypoint_pose[:2, 2] = p

            # Compute mean direction
            direction = np.zeros(2)
            count = 0

            for j in range(1, lookahead + 1):
                if i + j < N:
                    direction += waypoints_coordinates_Rw[i + j] - p
                    count += 1
            
            if count > 0:
                direction /= count
                
                # Convert direction → rotation matrix
                theta = normalize_angle(np.arctan2(direction[1], direction[0]))

                if len(waypoints_poses) > 0 and abs(theta - R_to_angle(waypoints_poses[-1][:2, :2])) < np.radians(10):
                    theta = R_to_angle(waypoints_poses[-1][:2, :2]) 
            else:
                theta = R_to_angle(waypoints_poses[-1][:2, :2])

            R = angle_to_R(theta)
            waypoint_pose[:2, :2] = R

            waypoints_poses.append(waypoint_pose)

        return waypoints_poses

    def set_goal_grid(self, goal_coordinate_grid: np.ndarray) -> None:
        # Get current robot pose in robot world frame
        with robot_pose_lock:
            robot_pose_Rw_2d = pose_to_xy_plane(robot_pose_world)  # Extract current rotation

        # Get waypoint coordinates in grid frame
        robot_coordinate_grid = points_Rw_2d_to_G(robot_pose_Rw_2d[:2, 2], self.occupancy_grid.cell_size, self.occupancy_grid.map_size_cells)
        astar = RiskAStar(self.occupancy_grid.get_risk_map(sigma=0.4))
        waypoints_coordinates_G = astar.plan(robot_coordinate_grid, goal_coordinate_grid)

        if waypoints_coordinates_G.shape[0] == 0:
            logger.error("No path found to the goal!")
            return

        # Print waypoints in grid coordinates for debugging
        logger.info("Waypoints in grid coordinates:")
        for i, wp in enumerate(waypoints_coordinates_G):
            logger.info(f"Waypoint {i}: {wp}")


        # Get waypoint poses in robot world frame and set them
        waypoints_poses_Rw = self._create_waypoint_poses(waypoints_coordinates_G)
        with self.waypoints_lock:
            self.waypoints = waypoints_poses_Rw
            self.current_waypoint_index = 0
            for i, wp in enumerate(self.waypoints):
                logger.info(f"Waypoint {i}: coordinate: {wp[:2, 2]}, heading: {np.degrees(R_to_angle(wp[:2, :2])):.1f} deg")

        # Start the background control loop if it hasn't been started yet
        if self._nav_thread is None or not self._nav_thread.is_alive():
            self._stop_event.clear()
            self._nav_thread = threading.Thread(target=self.navigate, args=(), daemon=True)
            self._nav_thread.start()

    def wait_until_clear(self):
        logger.info("Waiting for vicinity to clear...")
        while True:
            human_tracks = self.track_buffer.get()
            if human_tracks is not None:
                if all(not is_in_vicinity(track, robot_pose_world[:2, 3]) for track in human_tracks):
                    logger.info("Vicinity is clear. Resuming navigation.")
                    return
            time.sleep(0.5)

    def navigate(self):
        prev_cmd = (0.0, 0.0, 0.0)
        self.current_waypoint_index = 0

        with self.waypoints_lock:
            current_waypoint = self.waypoints[self.current_waypoint_index]

        while not self._stop_event.is_set():
            start = time.monotonic()

            with robot_pose_lock:
                robot_pose_Rw_2d = pose_to_xy_plane(robot_pose_world)

            tracks = self.track_buffer.get()
            if tracks is not None:
                human_tracks, robot_track = filter_robot_track(tracks, robot_pose_world)

                for track in human_tracks:
                    if is_in_vicinity(track, robot_pose_Rw_2d[:2, 2]):
                        logger.warning(f"Human track {track.id} is in the vicinity! Freezing navigation.")
                        self.frozen = True
                        self.wait_until_clear()
                        continue  # After waiting, re-check tracks and re-compute command

            # Check if we've reached the current waypoint
            distance = np.linalg.norm(current_waypoint[:2,2] - robot_pose_Rw_2d[:2,2])
            if distance < self.GOAL_TOLERANCE:
                logger.info(f"Waypoint {self.current_waypoint_index} reached!")
                self.current_waypoint_index += 1

                with self.waypoints_lock:
                    if self.current_waypoint_index < len(self.waypoints):
                        current_waypoint = self.waypoints[self.current_waypoint_index]
                        logger.info(f"Moving to next waypoint: {current_waypoint}")
                    else:
                        logger.info("All waypoints reached.")
                        self.current_goal_pose_Rw = None
                        self.waypoints =[]
                        self.current_waypoint_index = 0
                        return

            # Get linear drive commands to move towards the current waypoint
            cmd = self._get_linear_drive_commands(prev_cmd, robot_pose_Rw_2d, current_waypoint)

            # Send command to robot
            try:
                if self.kelo:
                    self.kelo.set_platform_velocity_target(*cmd, timeout=self.DRIVE_LOOP_PERIOD)
                    logger.info(f"Commanded velocity: vx={cmd[0]:.2f} m/s, vy={cmd[1]:.2f} m/s, wz={np.degrees(cmd[2]):.1f} deg/s")
            except Exception as e:
                logger.error("Robot command failed:", e)

            # sleep until next tick
            elapsed = time.monotonic() - start
            to_sleep = self.DRIVE_LOOP_PERIOD - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            else:
                logger.warning(f"Kelo drive loop overload: cycle took {elapsed*1000:.1f} ms")


# --- CONFIGURATION ---
AREA_FILE = "area_files/test15.area"
OCCUPANCY_GRID_FILE = "occupancy_grids/edited_occupancy_grid15.npz"

def radar_thread(buffer: TrackBuffer):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        track_dicts = socket.recv_pyobj()  # blocking, OK here
        track_proxies = [KalmanTrackProxy(track_dict) for track_dict in track_dicts]
        buffer.update(track_proxies)


if __name__ == "__main__":

    occupancy_grid = load_occupancy_grid(OCCUPANCY_GRID_FILE)

    risk_map = occupancy_grid.get_risk_map(sigma=0.4)

    # Mouse Callback for clicking on the grid
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # Ensure the click is within the grid bounds
            if 0 <= y < occupancy_grid.grid.shape[0] and 0 <= x < occupancy_grid.grid.shape[1]:
                pixel_val = occupancy_grid.grid[y, x]

                # Check pixel color: White (255) is Free, Black (0) is Occupied, Gray (128) is Unknown
                if np.all(pixel_val == 255):
                    navigator.set_goal_grid(np.array([x, y]))
                    logger.info(f"Goal set to grid cell: ({x}, {y})")
                elif np.all(pixel_val == 0):
                    logger.error("Goal rejected: Cell is OCCUPIED.")
                else:
                    logger.error("Goal rejected: Cell is UNKNOWN.")

    # Start the ZMQ receiver thread to get radar tracks
    buffer = TrackBuffer()
    threading.Thread(
        target=radar_thread,
        args=(buffer,),
        daemon=True
    ).start()

    # Initialize Navigator
    navigator = MobiNavigator(occupancy_grid, buffer)
    navigator.connect_to_kelo_safe()

    # Set up ZED camera parameters
    tracking_params = Zed.TrackingParams(align_with_gravity=True, area_file_path = AREA_FILE, enable_localization_only=True)
    runtime_params = Zed.RuntimeParams(confidence_threshold=30)

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_runtime_params=runtime_params,
        fps=60,
        serial_number=31733653
    ) as zed:
        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
        
        # Attach the mouse callback to the window
        cv2.setMouseCallback("Occupancy Grid", on_mouse_click)

        while True:
            zed._grab_images()
            image = zed.get_rgb_image_as_int()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pose_matrix = zed._retrieve_camera_pose()

            with robot_pose_lock:
                robot_pose_world = pose_Cw_to_Rw(pose_matrix)

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            
            human_tracks = buffer.get()
            if human_tracks is not None:
                human_tracks, robot_track = filter_robot_track(human_tracks, robot_pose_world)

            # pose_world is a local copy for visualization
            pose_world = pose_Cw_to_Rw(pose_matrix)
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world, human_tracks)

            key = cv2.waitKey(10)

            if key == ord("q"):
                break