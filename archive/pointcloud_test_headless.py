import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from loguru import logger
import time

# 1. ZED Imports
try:
    import pyzed.sl as sl
except ImportError:
    logger.error("ZED SDK not detected. Please install pyzed to use the camera.")
    sys.exit(1)


# 3. CuRobo Imports
from curobo.geom.types import WorldConfig, PointCloud
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld

# ---------------------------------------------------------
# FUNCTION: Capture Points from ZED
# ---------------------------------------------------------
def capture_zed_point_cloud(max_frames=200):
    logger.info("Initializing ZED Camera...")
    
    # --- Init parameters ---
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # --- Positional tracking parameters ---
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static (mounted on a tripod/robot base), set this:
    # positional_tracking_parameters.set_as_static = True 

    # --- Mapping parameters ---
    mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        mapping_range=sl.MAPPING_RANGE.AUTO,
        map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD,
    )

    # --- Open camera ---
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Camera Open Failed: {repr(status)}")
        return None

    # --- Enable modules ---
    if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
        logger.error("Enable Positional Tracking failed")
        zed.close()
        return None
    
    if zed.enable_spatial_mapping(mapping_parameters) != sl.ERROR_CODE.SUCCESS:
        logger.error("Enable Spatial Mapping failed")
        zed.close()
        return None

    mesh = sl.FusedPointCloud() # Create a mesh object to hold data
    timer = 0
    
    logger.info(f"Starting ZED Capture for {max_frames} frames...")

    # Grab frames
    while timer < max_frames:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            timer += 1

            # Request an update of the spatial map every 30 frames
            if timer % 30 == 0:
                zed.request_spatial_map_async()

            # Retrieve spatial_map when ready
            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and timer > 0:
                zed.retrieve_spatial_map_async(mesh)

            if timer % 50 == 0:
                print(f"Frame {timer}: {mesh.get_number_of_points()} points in cloud")

    # Final extraction
    logger.info("Extracting final spatial map...")
    zed.extract_whole_spatial_map(mesh)

        
    # Save as obj for debugging
    mesh.save("pointcloud.obj")
    
    # Get vertices as numpy array (N, 3)
    # mesh.vertices returns a list of floats, need to shape or cast to numpy
    points = mesh.vertices
    points_np = np.array(points)

    # --- FIX STARTS HERE ---
    # ZED often returns (N, 4) or flattened array of N*4 (x, y, z, color/pad)
    # We need to ensure we only pass (x, y, z) to CuRobo
    
    if points_np.size > 0:
        # 1. If it's a flat 1D array, reshape it to (N, 4) based on the size
        if len(points_np.shape) == 1:
            if points_np.size % 4 == 0:
                points_np = points_np.reshape(-1, 4)
            elif points_np.size % 3 == 0:
                points_np = points_np.reshape(-1, 3)
            else:
                logger.error(f"Unknown point cloud format with size {points_np.size}")
                return None

        # 2. If the shape is (N, 4), slice it to keep only (N, 3)
        if points_np.shape[1] == 4:
            points_np = points_np[:, :3]

    logger.info(f"Captured {points_np.shape[0]} points (Formatted to {points_np.shape}).")
    # --- FIX ENDS HERE ---

    logger.info(f"Captured {len(points_np)} points.")
    
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    if len(points_np) == 0:
        return None
        
    return points_np

# ---------------------------------------------------------
# SETUP: Isaac World and CuRobo Context
# ---------------------------------------------------------

# Define Robot Base Pose (World to Base transform)
arm_pos = (0.3545, 0, 0.74)
arm_euler = (np.pi/2, 0., -np.pi/2)
_X_World_Base = np.eye(4)
_X_World_Base[:3, 3] = np.array(arm_pos)
_X_World_Base[:3, :3] = R.from_euler("xyz", arm_euler).as_matrix()

# Load Robot Config
robot_arm_name = "ur5e"
robot_gripper_name = "robotiq_2f_85"
robot_camera_name = "realsense_d435"
robot_cfg_path = join_path(get_robot_configs_path(), f"{robot_arm_name}_{robot_gripper_name}_{robot_camera_name}.yml")

# Initialize CuRobo
tensor_args = TensorDeviceType()
initial_world = WorldConfig(cuboid=[], sphere=[]) 
config = RobotWorldConfig.load_from_config(
    robot_cfg_path,
    initial_world,
    collision_activation_distance=0.0
)
curobo_fn = RobotWorld(config)

# ---------------------------------------------------------
# STEP 1: CAPTURE REAL DATA
# ---------------------------------------------------------

# Run ZED Capture
zed_points_np = capture_zed_point_cloud(max_frames=200)

if zed_points_np is not None:
    # ---------------------------------------------------------
    # STEP 2: PROCESS WITH CUROBO
    # ---------------------------------------------------------
    
    # Note: ZED coordinates are usually relative to where the camera started (World Frame of ZED).
    # CuRobo needs points in the Robot Base Frame.
    # If the camera is NOT at the robot base, you need a transform here:
    # points_robot_frame = (T_RobotBase_ZEDWorld @ points_homogenous)[:3]
    # For now, we assume raw points are acceptable or camera was at origin.
    



    # Convert to Tensor
    points_t = torch.tensor(zed_points_np, dtype=tensor_args.dtype, device=tensor_args.device)

    # Create CuRobo PointCloud
    # We define the pose as Identity because the points themselves contain the spatial info
    pc_pose = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=tensor_args.dtype, device=tensor_args.device)
    
    curobo_pc = PointCloud(
        name="zed_fused_scan",
        pose=pc_pose,
        points=points_t
    )

    # Get Bounding Spheres (Voxelization)
    logger.info("Voxelizing point cloud to spheres...")
    start_time = time.monotonic()
    pc_spheres_list = curobo_pc.get_bounding_spheres(n_spheres=100, surface_sphere_radius = 0.02)
    end_time = time.monotonic()
    logger.info(f"Voxelization completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Generated {len(pc_spheres_list)} spheres from ZED point cloud.")

    # Get mesh
    logger.info("Generating mesh from point cloud...")
    start_time = time.monotonic()
    pc_mesh = curobo_pc.get_mesh()
    end_time = time.monotonic()
    logger.info(f"Mesh generation completed in {end_time - start_time:.2f}")

    # Update CuRobo Collision World
    initial_world.sphere.extend(pc_spheres_list)
    curobo_fn.update_world(initial_world)

else:
    logger.warning("No points captured from ZED. World will be empty.")
    pc_spheres_list = []