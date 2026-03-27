import threading
import time
import numpy as np
from loguru import logger
from airo_robots.drives.hardware.kelo_robile import KELORobile
from RiskAStar import RiskAStar
from datatypes import OccupancyGrid
from airo_camera_toolkit.cameras.zed.zed import Zed
import cv2
from constants import *
from utils import *

robot_pose_world = np.eye(4)  # 4x4 homogeneous transformation matrix
robot_pose_lock = threading.Lock()  # to protect access to robot_pose across threads

# --- CONFIGURATION ---
AREA_FILE = "area_files/test15.area"
OCCUPANCY_GRID_FILE = "occupancy_grids/edited_occupancy_grid15.npz"

class MobiNavigator:
    MAX_LIN_SPEED = 0.2      # m/s (tune)
    MAX_ANG_SPEED = np.pi/8. # rad/s (tune)
    SMOOTHING_ALPHA = 0.95   # for exponential smoothing of commanded velocity
    DRIVE_LOOP_HZ = 20.0     # control loop frequency
    DRIVE_LOOP_PERIOD = 1.0 / DRIVE_LOOP_HZ
    GOAL_TOLERANCE = 0.3     # meters

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.kelo = None
        self.occupancy_grid = occupancy_grid
        self._nav_thread = None
        self._stop_event = threading.Event()

        self.waypoints =[]
        self.current_waypoint_index = 0
        self.waypoints_lock = threading.Lock()

    def get_linear_drive_commands(self, prev_cmd: tuple, robot_pose_grid: tuple, goal_cell: tuple, grid: OccupancyGrid) -> Optional[tuple]:
            robot_cx, robot_cy, robot_heading = robot_pose_grid
            
            # Calculate errors
            error_x = (goal_cell[0] - robot_cx) * grid.cell_size
            error_y = (goal_cell[1] - robot_cy) * grid.cell_size 
            distance = np.hypot(error_x, error_y)
            angle_to_goal = np.arctan2(error_y, error_x) % (2 * np.pi)
            if angle_to_goal > np.pi:
                angle_to_goal -= 2 * np.pi
            angle_error = (angle_to_goal - robot_heading) % (2 * np.pi)
            if angle_error > np.pi:
                angle_error -= 2 * np.pi

            if distance < self.GOAL_TOLERANCE:
                logger.info("Goal reached!")
                return None
            
            # Direction to goal in world frame (unit vector)
            dir_x = np.cos(angle_to_goal)
            dir_y = np.sin(angle_to_goal)

            # Desired velocity in world frame
            vx_w = np.clip(distance, 0, self.MAX_LIN_SPEED) * dir_x
            vy_w = np.clip(distance, 0, self.MAX_LIN_SPEED) * dir_y

            # Transform to robot (local) frame
            vx =  np.cos(robot_heading) * vx_w + np.sin(robot_heading) * vy_w
            vy = np.sin(robot_heading) * vx_w - np.cos(robot_heading) * vy_w

            # Angular velocity stays the same
            wz = np.clip(-angle_error, -self.MAX_ANG_SPEED, self.MAX_ANG_SPEED)

            logger.info(f"Robot pose (grid): ({robot_cx}, {robot_cy}), Heading: {np.degrees(robot_heading):.1f} deg")
            logger.info(f"Goal: {goal_cell}, Dist Error: {distance:.2f} m, Angle to goal: {np.degrees(angle_to_goal):.1f} deg, Angle Error: {np.degrees(angle_error):.1f} deg")
            logger.info(f"Commanded velocity (robot frame): vx={vx:.2f} m/s, vy={vy:.2f} m/s, wz={np.degrees(wz):.1f} deg/s")

            # smoothing (exponential)
            vx = self.SMOOTHING_ALPHA * vx + (1 - self.SMOOTHING_ALPHA) * prev_cmd[0]
            vy = self.SMOOTHING_ALPHA * vy + (1 - self.SMOOTHING_ALPHA) * prev_cmd[1]
            wz = self.SMOOTHING_ALPHA * wz + (1 - self.SMOOTHING_ALPHA) * prev_cmd[2]
            
            return vx, vy, wz

    def connect_to_kelo_safe(self) -> bool:
        try:
            self.kelo = KELORobile("10.42.0.1")
            logger.success("KELO connected ✅")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to KELO: {e}")
            return False

    def set_goal(self, goal_cell: tuple, grid: OccupancyGrid):
        """Update the target goal. Starts the background thread if it isn't running."""
        
        with robot_pose_lock:
            robot_pose = pose_world_to_grid(robot_pose_world, grid)

        robot_cx, robot_cy, heading = robot_pose
        print(f"Current robot grid pose: ({robot_cx}, {robot_cy}), Heading: {np.degrees(heading):.1f} deg")

        robot_cx, robot_cy, _ = robot_pose
        astar = RiskAStar(self.occupancy_grid.get_risk_map(sigma=1.0))
        waypoints = astar.plan((robot_cx, robot_cy), goal_cell)

        # subsample waypoints
        waypoints = waypoints[4::5]

        if waypoints is None or len(waypoints) == 0:
            logger.error("No path found to the goal!")
            return

        self.active_goal = goal_cell

        print(f"Planned waypoints: {waypoints}")

        # Start the background control loop if it hasn't been started yet
        if self._nav_thread is None or not self._nav_thread.is_alive():
            self._stop_event.clear()
            self._nav_thread = threading.Thread(target=self.navigate, args=(waypoints, grid), daemon=True)
            self._nav_thread.start()

    def navigate(self, waypoints: list[tuple], grid: OccupancyGrid):
        prev_cmd = (0.0, 0.0, 0.0)
        current_waypoint_index = 0
        current_goal = waypoints[current_waypoint_index]

        while not self._stop_event.is_set():
            start = time.monotonic()
            
            # If there's no active goal, make sure the robot is stopped and wait
            if current_goal is None:
                if self.kelo:
                    try:
                        self.kelo.set_platform_velocity_target(0.0, 0.0, 0.0, timeout=self.DRIVE_LOOP_PERIOD)
                    except: pass
                time.sleep(self.DRIVE_LOOP_PERIOD)
                continue

            with robot_pose_lock:
                robot_pose = pose_world_to_grid(robot_pose_world, grid)
            
            error_x = (current_goal[0] - robot_pose[0]) * grid.cell_size
            error_y = (current_goal[1] - robot_pose[1]) * grid.cell_size
            distance = np.hypot(error_x, error_y)

            if distance < self.GOAL_TOLERANCE:
                logger.info(f"Waypoint {current_waypoint_index} reached!")
                current_waypoint_index += 1
                if current_waypoint_index < len(waypoints):
                    current_goal = waypoints[current_waypoint_index]
                    logger.info(f"Moving to next waypoint: {current_goal}")
                else:
                    logger.info("All waypoints reached. Stopping.")
                    current_goal = None
                    continue

            cmd = self.get_linear_drive_commands(prev_cmd, robot_pose, current_goal, grid)
            prev_cmd = cmd

            # Send command to robot
            try:
                if self.kelo:
                    self.kelo.set_platform_velocity_target(*cmd, timeout=self.DRIVE_LOOP_PERIOD)
                    logger.info(f"Sent command to KELO: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={np.degrees(cmd[2]):.1f} deg/s")
            except Exception as e:
                logger.error("Robot command failed:", e)

            # sleep until next tick
            elapsed = time.monotonic() - start
            to_sleep = self.DRIVE_LOOP_PERIOD - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            else:
                logger.warning(f"Kelo drive loop overload: cycle took {elapsed*1000:.1f} ms")

    # def navigate(self, grid: OccupancyGrid):
    #     logger.info("Kelo drive thread started.")
    #     prev_cmd = (0.0, 0.0, 0.0)

    #     while not self._stop_event.is_set():
    #         start = time.monotonic()
            
    #         goal_cell = self.active_goal
            
    #         # If there's no active goal, make sure the robot is stopped and wait
    #         if goal_cell is None:
    #             if self.kelo:
    #                 try:
    #                     self.kelo.set_platform_velocity_target(0.0, 0.0, 0.0, timeout=self.DRIVE_LOOP_PERIOD)
    #                 except: pass
    #             time.sleep(self.DRIVE_LOOP_PERIOD)
    #             continue

    #         with robot_pose_lock:
    #             robot_cx, robot_cy, robot_heading = pose_world_to_grid(robot_pose_world, grid)
            
    #         # Calculate errors
    #         error_x = (goal_cell[0] - robot_cx) * grid.cell_size
    #         error_y = (goal_cell[1] - robot_cy) * grid.cell_size 
    #         distance = np.hypot(error_x, error_y)

    #         angle_to_goal = np.arctan2(error_y, error_x) % (2 * np.pi)
    #         if angle_to_goal > np.pi:
    #             angle_to_goal -= 2 * np.pi
    #         angle_error = (angle_to_goal - robot_heading) % (2 * np.pi)
    #         if angle_error > np.pi:
    #             angle_error -= 2 * np.pi

    #         if distance < self.GOAL_TOLERANCE:
    #             logger.success(f"Goal reached! (within {distance:.3f} m)")
    #             self.active_goal = None
    #             continue
            
    #         # Direction to goal in world frame (unit vector)
    #         dir_x = np.cos(angle_to_goal)
    #         dir_y = np.sin(angle_to_goal)

    #         # Desired velocity in world frame
    #         vx_w = np.clip(distance, 0, self.MAX_LIN_SPEED) * dir_x
    #         vy_w = np.clip(distance, 0, self.MAX_LIN_SPEED) * dir_y

    #         # Transform to robot (local) frame
    #         vx =  np.cos(robot_heading) * vx_w + np.sin(robot_heading) * vy_w
    #         vy = np.sin(robot_heading) * vx_w - np.cos(robot_heading) * vy_w

    #         # Angular velocity stays the same
    #         wz = np.clip(-angle_error, -self.MAX_ANG_SPEED, self.MAX_ANG_SPEED)

    #         logger.info(f"Robot pose (grid): ({robot_cx}, {robot_cy}), Heading: {np.degrees(robot_heading):.1f} deg")
    #         logger.info(f"Goal: {goal_cell}, Dist Error: {distance:.2f} m, Angle to goal: {np.degrees(angle_to_goal):.1f} deg, Angle Error: {np.degrees(angle_error):.1f} deg")
    #         logger.info(f"Commanded velocity (robot frame): vx={vx:.2f} m/s, vy={vy:.2f} m/s, wz={np.degrees(wz):.1f} deg/s")

    #         # smoothing (exponential)
    #         vx = self.SMOOTHING_ALPHA * vx + (1 - self.SMOOTHING_ALPHA) * prev_cmd[0]
    #         vy = self.SMOOTHING_ALPHA * vy + (1 - self.SMOOTHING_ALPHA) * prev_cmd[1]
    #         wz = self.SMOOTHING_ALPHA * wz + (1 - self.SMOOTHING_ALPHA) * prev_cmd[2]
    #         prev_cmd = (vx, vy, wz)

    #         # Send command to robot
    #         try:
    #             if self.kelo:
    #                 self.kelo.set_platform_velocity_target(vx, vy, wz, timeout=self.DRIVE_LOOP_PERIOD)
    #                 logger.info(f"Sent command to KELO: vx={vx:.2f}, vy={vy:.2f}, wz={np.degrees(wz):.1f} deg/s")
    #         except Exception as e:
    #             logger.error("Robot command failed:", e)

    #         # sleep until next tick
    #         elapsed = time.monotonic() - start
    #         to_sleep = self.DRIVE_LOOP_PERIOD - elapsed
    #         if to_sleep > 0:
    #             time.sleep(to_sleep)
    #         else:
    #             logger.warning(f"Kelo drive loop overload: cycle took {elapsed*1000:.1f} ms")



if __name__ == "__main__":

    occupancy_grid = load_occupancy_grid(OCCUPANCY_GRID_FILE)

    # Initialize Navigator
    navigator = MobiNavigator(occupancy_grid)
    navigator.connect_to_kelo_safe()

    tracking_params = Zed.TrackingParams(align_with_gravity=True, area_file_path = AREA_FILE, enable_localization_only=True)
    runtime_params = Zed.RuntimeParams(confidence_threshold=30)
    
    risk_map = occupancy_grid.get_risk_map(sigma=1.0)


    # vis = cv2.normalize(risk_map, None, 0, 255, cv2.NORM_MINMAX)
    # vis = vis.astype(np.uint8)

    # color_vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    # cv2.imshow("Risk Map", color_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Mouse Callback for clicking on the grid
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            grid_img = occupancy_grid.grid
            
            # Ensure the click is within the grid bounds
            if 0 <= y < grid_img.shape[0] and 0 <= x < grid_img.shape[1]:
                pixel_val = grid_img[y, x]
                
                # Check pixel color: White (255) is Free, Black (0) is Occupied, Gray (128) is Unknown
                if np.all(pixel_val == 255):
                    logger.info(f"Navigating to free cell: ({x}, {y})")
                    navigator.set_goal((x, y), occupancy_grid)
                elif np.all(pixel_val == 0):
                    logger.error("Goal rejected: Cell is OCCUPIED.")
                else:
                    logger.error("Goal rejected: Cell is UNKNOWN.")

    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_runtime_params=runtime_params,
        fps=60,
        serial_number=31733653
    ) as zed:
        
        print("Starting real-time 2D occupancy grid mapping...")
        print("Press 'q' in the OpenCV window to exit.")

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
                robot_pose_world = pose_camera_to_world(pose_matrix)

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)

            # pose_world is a local copy for visualization
            pose_world = pose_camera_to_world(pose_matrix)
            draw_occupancy_grid("Occupancy Grid", occupancy_grid, pose_world)

            key = cv2.waitKey(10)

            if key == ord("q"):
                break