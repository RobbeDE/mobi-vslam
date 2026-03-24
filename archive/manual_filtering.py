import sys
import time
import pyzed.sl as sl
import argparse
import numpy as np
import os
import torch

# 3. CuRobo Imports
from curobo.geom.types import WorldConfig, PointCloud, Sphere, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld

from scipy.spatial.transform import Rotation as R

VOXEL_SIZE = 0.05  # in meters


# --- Init parameters ---
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.camera_resolution = sl.RESOLUTION.HD720
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# --- Positiona tracking parameters ---
positional_tracking_parameters = sl.PositionalTrackingParameters()

# --- Mapping parameters ---
mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.LOW, 
                                                mapping_range = sl.MAPPING_RANGE.MEDIUM, 
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
initial_world = WorldConfig(cuboid=[])
cuboid_world = WorldConfig.create_obb_world(initial_world)
config = RobotWorldConfig.load_from_config(
    robot_cfg_path,
    cuboid_world,
    collision_activation_distance=0.0
)
curobo_fn = RobotWorld(config)


# create spheres with shape batch, horizon, n_spheres, 4.
q_sph = torch.tensor([[[[0, 0, 0, 500]]]], device=tensor_args.device, dtype=tensor_args.dtype)

# Grab 5000 frames and stop
while timer < 1000 :
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
        # When grab() = SUCCESS, a new image, depth and pose is available.
        # Spatial mapping automatically ingests the new data to build the mesh.
        timer += 1

        # Request an update of the spatial map every 30 frames
        if timer % 30 == 0 :
            zed.request_spatial_map_async()

        # Retrieve spatial_map when ready
        if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and timer > 0:
            zed.retrieve_spatial_map_async(mesh)

            print(mesh.get_number_of_points(), "original number of points in the mesh")

            num_chunks_updated = 0
            for chunk in mesh.chunks:
                if chunk.has_been_updated:
                    num_chunks_updated += 1
            print(f"{num_chunks_updated/len(mesh.chunks)*100:.2f}% updated chunks in the mesh")

            points_np = mesh.vertices[:, :3]  # (N,3) numpy array of point positions

            # Integer division (quantization)
            quantized = np.round(points_np / VOXEL_SIZE)
            
            # Remove duplicates (this effectively downsamples)
            unique_indices = np.unique(quantized, axis=0, return_index=True)[1]
            points_downsampled = points_np[unique_indices]

            print(points_downsampled.shape[0], "downsampled number of points in the mesh")

            # 1. Prepare poses in bulk (Vectorized)
            # We need (N, 7) for [x, y, z, w, x, y, z] (Position + Quaternion)
            # The Sphere class expects a list, so we prep numpy first then convert.
            
            N = points_downsampled.shape[0]
            
            # Create quaternion columns [1, 0, 0, 0] (Identity rotation)
            # Adjust this if your CuRobo version expects [x, y, z, w]
            quats = np.zeros((N, 4))
            quats[:, 0] = 1.0  # Set w=1
            
            # Concatenate [x,y,z] + [1,0,0,0] -> (N, 7)
            poses_np = np.hstack([points_downsampled, quats])
            
            # 2. Create the list of Sphere objects
            # We use a list comprehension which is slightly faster than a standard for-loop
            radius = float(VOXEL_SIZE)
            

            start = time.monotonic()
            cuboids = [
                Cuboid(
                    name=f"env_vox_{i}", 
                    dims=[radius/2, radius/2, radius/2], 
                    pose=row.tolist(),  # Convert numpy row to Python list
                    color=[0.0, 0.0, 1.0, 1.0]
                )
                for i, row in enumerate(poses_np)
            ]

            cuboid_world.cuboid = cuboids
            curobo_fn.update_world(cuboid_world)

            print(f"World updated with downsampled point cloud in {time.monotonic() - start:.2f} seconds.")


            start = time.monotonic()
            d = curobo_fn.get_collision_distance(q_sph)

            print(f"Collision distances computed in {time.monotonic() - start:.4f} seconds.")


zed.disable_spatial_mapping()
zed.disable_positional_tracking()
zed.close()