import numpy as np
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from loguru import logger
from scipy.spatial.transform import Rotation as R
from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld
from curobo.geom.types import WorldConfig, Sphere
import torch

from curobo.types.base import TensorDeviceType


import sys
import time
import pyzed.sl as sl
import argparse
    


robot_arm_name = "ur5e"
robot_gripper_name = "robotiq_2f_85"
logger.info(f"Loading robot: {robot_arm_name} with gripper: {robot_gripper_name}...")

# Load robot configuration from yaml file
robot_cfg = load_yaml(join_path(get_robot_configs_path(), f"{robot_arm_name}.yml"))["robot_cfg"]

spheres = []
# 2. Add to objects list for Curobo
sphere_pos = np.array([0.1, 0.1, 0.1])
sphere_r = 8.0
# apply world to base transform
# sphere_pos_Base = np.linalg.inv(self._X_World_Base) @ np.hstack((sphere_pos, 1))
sphere_pos_Base = np.hstack((sphere_pos, 1))
print(f"sphere_pos_Base:{sphere_pos_Base[:3]}")
print(f"pose: {[*sphere_pos_Base[:3], 1, 0, 0, 0]}")
spheres.append(Sphere(
    name="my_sphere",
    pose=[*sphere_pos_Base[:3], 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    radius=sphere_r
))
sphere_pos_Base[0] += 1.0
spheres.append(Sphere(
    name="my_sphere2",
    pose=[*sphere_pos_Base[:3], 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    radius=sphere_r
))
logger.info(f"Added sphere obstacle at {sphere_pos} (base frame: {sphere_pos_Base[:3]}) with radius {sphere_r}")



# -------- Build WorldConfig for Curobo
_world_config = WorldConfig(
    sphere=spheres
)
# To use this world in a collision checker, we need to approximate some object types
# as cuRobo currently only provides a collision checker for cuboids and meshes.
# Capsules, cylinders, and spheres can be approximated to cuboids using
cuboid_world = WorldConfig.create_obb_world(_world_config) #

# CuRobo robot for kinematics and collision checking
tensor_args = TensorDeviceType()
config = RobotWorldConfig.load_from_config(
    f"{robot_arm_name}.yml",
    cuboid_world,
    collision_activation_distance=0.0
)
_curobo_fn = RobotWorld(config)







# --- Init parameters ---
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.camera_resolution = sl.RESOLUTION.HD720
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# --- Positiona tracking parameters ---
positional_tracking_parameters = sl.PositionalTrackingParameters()

# --- Mapping parameters ---
mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.MEDIUM, 
                                                mapping_range = sl.MAPPING_RANGE.AUTO, 
                                                map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD,)

# --- Open camera ---
zed = sl.Camera()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open :", repr(status))
    exit(1)

# --- Enable positional tracking & spatial mapping ---
if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
    print("Enable Positional Tracking failed")
    exit(1)
if zed.enable_spatial_mapping(mapping_parameters) != sl.ERROR_CODE.SUCCESS:
    print("Enable Spatial Mapping failed")
    exit(1)

mesh = sl.FusedPointCloud() # Create a mesh object
timer = 0

# Grab 500 frames and stop
while timer < 500 :
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
        # When grab() = SUCCESS, a new image, depth and pose is available.
        # Spatial mapping automatically ingests the new data to build the mesh.
        timer += 1

        # Request an update of the spatial map every 30 frames (0.5s in HD720 mode)
        if timer % 30 == 0 :
            zed.request_spatial_map_async()

        # Retrieve spatial_map when ready
        if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and timer > 0:
            zed.retrieve_spatial_map_async(mesh)

        if timer % 50 == 0:
            print("Frame number :", timer)
            print(mesh.get_number_of_points(), " points in the fused point cloud")

zed.extract_whole_spatial_map(mesh)
mesh.save("mesh.obj")

zed.disable_spatial_mapping()
zed.disable_positional_tracking()
zed.close()