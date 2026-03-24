import pyzed.sl as sl
from visual_slam.shared_variables import shared_zed_data
import time
import threading
from visual_slam.utils import timestamp

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

# ZED Initialization Parameters
init = sl.InitParameters()
init.sdk_gpu_id = -1 # Default, searches for all CUDA capable devices and picks the best one.
init.enable_image_enhancement = True # Default, improve image quality using Enhanced Contrast Technology
init.depth_mode = sl.DEPTH_MODE.NEURAL # Default
init.camera_fps = 0 # Use max fps
init.coordinate_units = sl.UNIT.METER
init.camera_resolution = sl.RESOLUTION.HD720 # or HD1200, small resolutions offer higher framerate and lower computation time
init.depth_stabilization = 30 # Default, stabilization smoothness is linear from 1 to 100 (see API docs)
init.depth_minimum_distance = -1 # Default, no min distance. This value cannot be greater than 3 meters
init.depth_maximum_distance = -1 # Default, no max distance. ZED SDK turns higher values into inf types. No effect on spatial mapping and tracking!!!
init.open_timeout_sec = 5 # Default, time to wait before failing opening the camera.
init.grab_compute_capping_fps = 0 # Default, no fps capping.
init.enable_image_validity_check = 0 # Default, do not check image corruption.
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

# Spatial Mapping Parameters
mapping_parameters = sl.SpatialMappingParameters(
    resolution=sl.MAPPING_RESOLUTION.HIGH, # Max resolution is 2 cm. LOW = 8, MEDIUM = 5, HIGH = 2
    mapping_range=sl.MAPPING_RANGE.SHORT, # also enum sl.MAPPING_RANGE available: SHORT = 3.5, MEDIUM = 5, LONG = 10 meters
    map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD, # Mesh type also available
    max_memory_usage = 2048, # Default, max memory (in MB) allocated for the meshing process
    use_chunk_only=False # Default, ensure consistency between the mesh and its inner chunk data.
)
MAP_UPDATE_FRAME_INTERVAL = 3  # Update map every 3 frames

# Depth Sensing Parameters
downscale_factor = 8 # Downscale factor for depth sensing to improve performance
DEPTH_SENSING_RESOLUTION = sl.Resolution(
    1280 // downscale_factor,
    720 // downscale_factor
)

# Positional Tracking Parameters
tracking_parameters = sl.PositionalTrackingParameters()
tracking_parameters.set_initial_world_transform(sl.Transform()) # Default, identity transform. Pose of the camera in the world frame at the start.
tracking_parameters.enable_area_memory = True # Default, enable area memory for spatial mapping such that the camera remembers its surroundings.
tracking_parameters.enable_pose_smoothing = True # Default. apply smoothing to the camera pose.
tracking_parameters.set_floor_as_origin = False # Default, align tracking with the floor plane.
tracking_parameters.enable_imu_fusion = True # Default, use IMU data to improve tracking accuracy.
tracking_parameters.set_as_static = False # Default, the camera is not static.
tracking_parameters.depth_min_range = -1 # Default, no min range for depth data used in tracking. May be useful for when static objects block the view.
tracking_parameters.set_gravity_as_origin = True # Align one axis with gravity vector from IMU (override 2 / 3 rotations from inital_world_transform).

# Printing parameters
PRINT_FRAME_INTERVAL = 60 # Print status every 60 frames

# ---------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------

class ZedThread(threading.Thread):
    def __init__(self, daemon: bool = True, verbose: bool = False) -> None:
        super().__init__(daemon=daemon)
        self._zed = sl.Camera()
        self._stop_event = threading.Event()

        self._spatial_map = sl.FusedPointCloud()
        self._rgb_image = sl.Mat()
        self._depth_image = sl.Mat()
        self._live_point_cloud = sl.Mat()
        self._live_pose = sl.Pose()
        self._floor_plane = sl.Plane()
        self._reset_tracking_floor_plane = sl.Transform()

        self._start_time = time.perf_counter()
        self._verbose = verbose

    def run(self) -> None:
        if self._zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera, exiting...")
        if self._zed.enable_positional_tracking(tracking_parameters) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to enable positional tracking, exiting...")
        if self._zed.enable_spatial_mapping(mapping_parameters) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to enable spatial mapping, exiting...")
        

        # Print some camera information
        init_parameters = self._zed.get_init_parameters()
        print(f"Depth mode: {init_parameters.depth_mode}")
        print(f"Camera resolution: {init_parameters.camera_resolution} (HD720 = 1280*720)")
        print(f"Coordinate units: {init_parameters.coordinate_units}")
        print(f"Coordinate system: {init_parameters.coordinate_system}")

        
        self._zed_caputure()

        self.on_exit()

    def _zed_caputure(self) -> None:
        frame_count = 0

        while not self._stop_event.is_set():
            if self._zed.grab() == sl.ERROR_CODE.SUCCESS:

                if self._verbose: timestamp(f"--- Frame {frame_count} ---")

                # Request spatial map update
                if frame_count % MAP_UPDATE_FRAME_INTERVAL == 0: # Update map every 5 frames
                    self._zed.request_spatial_map_async()
                    if self._verbose: timestamp("Requested spatial map update.")

                # Request floor plane extraction update
                if self._zed.find_floor_plane(self._floor_plane, self._reset_tracking_floor_plane, floor_height_prior=1.45, floor_height_prior_tolerance=0.1) == sl.ERROR_CODE.SUCCESS:
                    if self._verbose: timestamp("Extracted floor plane successfully.")
                    shared_zed_data.set_floor_plane(self._floor_plane.get_plane_equation())

                # Retrieve live pose
                positional_tracking_state = self._zed.get_position(self._live_pose, sl.REFERENCE_FRAME.WORLD)
                if (positional_tracking_state == sl.POSITIONAL_TRACKING_STATE.OK):
                    # Convert to numpy array
                    pose_transform = self._live_pose.pose_data()
                    pose_np = pose_transform.m

                    # Update shared variable
                    shared_zed_data.set_live_pose(pose_np)

                    if self._verbose: timestamp("Updated live pose.")
                

                # Retrieve live pointcloud
                if (self._zed.retrieve_measure(self._live_point_cloud, sl.MEASURE.XYZRGBA, resolution = DEPTH_SENSING_RESOLUTION) == sl.ERROR_CODE.SUCCESS):
                    # Convert to numpy array
                    point_cloud_np = self._live_point_cloud.get_data()
                    point_cloud_np = point_cloud_np[:, :, :3] # drop the last channel
                    point_cloud_np = point_cloud_np.reshape(-1, 3) # reshape to (N, 3)

                    # Update shared variable
                    shared_zed_data.set_live_point_cloud(point_cloud_np)

                    if self._verbose: timestamp("Updated live point cloud.")


                # Retrieve rgb image
                if (self._zed.retrieve_image(self._rgb_image, sl.VIEW.LEFT) == sl.ERROR_CODE.SUCCESS):
                    # Convert to numpy array
                    rgb_image_np = self._rgb_image.get_data()

                    # Update shared variable
                    shared_zed_data.set_rgb_image(rgb_image_np)

                    if self._verbose: timestamp("Updated RGB image.")


                # Retrieve depth image
                if (self._zed.retrieve_measure(self._depth_image, sl.MEASURE.DEPTH) == sl.ERROR_CODE.SUCCESS):
                    # Convert to numpy array
                    depth_image_np = self._depth_image.get_data()

                    # Update shared variable
                    shared_zed_data.set_depth_image(depth_image_np)

                    if self._verbose: timestamp("Updated depth image.")


                # Update spatial map if ready
                if self._zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    if (self._zed.retrieve_spatial_map_async(self._spatial_map) == sl.ERROR_CODE.SUCCESS):
                        
                        # Convert to list of tuples (vertices, has_been_updated)
                        spatial_map = []
                        for chunk in self._spatial_map.chunks:
                            spatial_map.append((chunk.vertices[:, :3], chunk.has_been_updated))

                        # Update shared variable
                        shared_zed_data.set_spatial_map(spatial_map)

                        if self._verbose: timestamp("Updated spatial map.")

                shared_zed_data.mark_full_update()

                if frame_count % PRINT_FRAME_INTERVAL == 0:
                    print(f"ZED current FPS: {self._zed.get_current_fps()}")
                    print(f"Positional tracking state: {positional_tracking_state}")
                    print(f"Spatial mapping state: {self._zed.get_spatial_mapping_state()}")


                frame_count += 1
            
    def stop(self) -> None:
        self._stop_event.set()
    
    def on_exit(self) -> None:
        print("Terminating ZedCapture thread...")
        self._zed.disable_spatial_mapping()
        self._zed.disable_positional_tracking()
        self._zed.close()