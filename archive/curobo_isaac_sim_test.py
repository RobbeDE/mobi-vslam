import threading

import numpy as np
import torch
from curobo.geom.types import WorldConfig, Sphere, Cuboid
from curobo.types.base import TensorDeviceType
# CuRobo
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld
# Isaac Sim
from isaacsim.simulation_app import SimulationApp
from loguru import logger
from scipy.spatial.transform import Rotation as R

simulation_app = SimulationApp( {"headless": False, "width": 1920, "height": 1080} )
from omni.isaac.core import World
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim

isaac_world = World(stage_units_in_meters=1.0)
# cuRobo assumes the world is represented with respect to the base frame of the robot from the kinematic chain
# Robot base position and orientation
arm_pos = (0.3545, 0, 0.74)
arm_euler = (np.pi/2,0.,-np.pi/2)
_X_World_Base = np.eye(4)  # World to Robot base transform
_X_World_Base[:3, 3] = np.array(arm_pos)
_X_World_Base[:3, :3] = R.from_euler("xyz", arm_euler).as_matrix()
# Add a ground plane
isaac_world.scene.add_default_ground_plane() # -> for visualization only


# Load robot configuration from yaml file
robot_arm_name = "ur5e"
robot_gripper_name = "robotiq_2f_85"
robot_camera_name = "realsense_d435"
logger.info(f"Loading robot: {robot_arm_name} with gripper: {robot_gripper_name} and camera: {robot_camera_name}")
robot_cfg = load_yaml(join_path(get_robot_configs_path(), f"{robot_arm_name}_{robot_gripper_name}_{robot_camera_name}.yml"))["robot_cfg"]
print(robot_cfg)

# 2. Add to objects list for Curobo
spheres = []
spheres.append(Sphere(
    name="my_spherex",
    pose=[1., 0., 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    radius=0.01,
))
spheres.append(Sphere(
    name="my_spherey",
    pose=[0., 1., 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    radius=0.01,
))
spheres.append(Sphere(
    name="my_spherez",
    pose=[0., 0., 1., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    radius=0.01,
))

cuboids = []
cuboids.append(Cuboid(
    name="arm_attachment_base",
    pose=[0.0, 0.0, -0.02, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[0.175, 0.18, 0.04], # along x,y,z axes
))
rotation = R.from_euler('xyz', [-np.deg2rad(35), 0, 0]).as_quat(scalar_first=True)
cuboids.append(Cuboid(
    name="arm_attachment_left",
    pose=[0.0825, 0.0, -0.16, *rotation], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[0.01, 0.01, 0.3], # along x,y,z axes
))
cuboids.append(Cuboid(
    name="arm_attachment_right",
    pose=[-0.0825, 0.0, -0.16, *rotation], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[0.01, 0.01, 0.3], # along x,y,z axes
))
cuboids.append(Cuboid(
    name="mobi",
    pose=[0., -0.4225, -0.375, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[0.54, 0.665, 0.72], # along x,y,z axes
))
cuboids.append(Cuboid(
    name="ground",
    pose=[0., -_X_World_Base[2, 3]-0.01/2., 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[5.0, 0.01, 5.0], # along x,y,z axes
))


# -------- Build WorldConfig for Curobo
_world_config = WorldConfig(
    sphere=spheres,
    cuboid=cuboids
)
# To use this world in a collision checker, we need to approximate some object types
# as cuRobo currently only provides a collision checker for cuboids and meshes.
# Capsules, cylinders, and spheres can be approximated to cuboids using
cuboid_world = WorldConfig.create_obb_world(_world_config)
cuboid_world.save_world_as_mesh("debug_collision_cuboid.obj")



# CuRobo robot for kinematics and collision checking
tensor_args = TensorDeviceType()
config = RobotWorldConfig.load_from_config(
    join_path(get_robot_configs_path(), f"{robot_arm_name}_{robot_gripper_name}_{robot_camera_name}.yml"),
    cuboid_world,
    collision_activation_distance=0.0
)
_curobo_fn = RobotWorld(config) # model can be updated by updating cuboid_world and then calling _robot_world.update_world(cuboid_world)

#-----------------------------------------------------------------
def convert_cuboid_world_to_isaacsim(cuboid_world: WorldConfig, isaac_world: World, X_World_Base: np.ndarray):
    for i, cuboid in enumerate(cuboid_world.cuboid):
        pos_B = cuboid.pose[:3]
        quat_B = cuboid.pose[3:]
        size = [dim/2. for dim in cuboid.dims]
        # Transform to world frame
        pos_W = (X_World_Base @ np.hstack((pos_B, 1)))[:3]
        quat_W = (R.from_matrix(X_World_Base[:3, :3]) * R.from_quat(quat_B, scalar_first=True)).as_quat(scalar_first=True)  # xyzw
        print(f"cuboid {i}: pos_W: {pos_W}, quat_W: {quat_W}, size: {size}")

        prim_path = f"/World/cuboid_{i}_{cuboid.name}" if cuboid.name else f"/World/cuboid_{i}"
        if not prim_utils.is_prim_path_valid(prim_path):
            cube_prim = prim_utils.create_prim(
                prim_path=prim_path,
                prim_type="Cube",
                position=pos_W,
                orientation=quat_W, # xyzw to wxyz
                scale=size
            )
            if cube_prim is None:
                logger.error(f"Failed to add cuboid {i} to Isaac Sim")
                continue
        else:
            # Update existing prim
            xform_prim = XFormPrim(prim_path)
            xform_prim.set_local_poses(translations=np.array([pos_W]), orientations=np.array([quat_W]))  # wxyz quaternion
            xform_prim.set_local_scales(np.array([size]))
            logger.info(f"Updated cuboid {prim_path} in Isaac Sim to pos {pos_W}, quat {quat_W}, size {size}")

def visualize_robot_collision_spheres(robot_world_fn: RobotWorld, q: np.ndarray, isaac_world: World, X_World_Base: np.ndarray):
    # Compute forward kinematics to get the robot base pose in the world frame
    q = torch.tensor(q, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
    collision_spheres = robot_world_fn.kinematics.get_state(q).link_spheres_tensor
    print(f"collision_spheres: {collision_spheres}")
    # Visualize the robot collision spheres in Isaac Sim
    for i, sphere in enumerate(collision_spheres[0]):
        pos_B = sphere[:3].detach().cpu().numpy()
        r = sphere[3].detach().cpu().numpy()
        if r < 1e-4:
            continue
        # Transform to world frame
        pos_W = (X_World_Base @ np.hstack((pos_B, 1)))[:3]
        print(f"robot sphere {i}: pos_W: {pos_W}, r: {r}")

        sphere_prim = prim_utils.create_prim(
            prim_path=f"/World/robot_sphere_{i}",
            prim_type="Sphere",
            position=pos_W,
            scale=[r]*3,
        )
        if sphere_prim is None:
            logger.error(f"Failed to add robot collision sphere {i} to Isaac Sim")
            continue

convert_cuboid_world_to_isaacsim(cuboid_world, isaac_world, _X_World_Base)
# q1 = np.array([1.5671,     -2.5338  ,    1.3717  ,  -0.41391   ,   1.5426     , 3.1205])
# q1 = np.array([0.94377    , -2.5572  ,   -2.4918    ,  5.0831    ,  2.5096  ,   -1.5525])
# q1 = np.array([1.8524   ,  -2.3003  ,   -2.2353    ,  4.5636     , 1.8264  ,  0.029593])
q1 = np.array([1.4452 ,    -3.2882    ,  1.4966  ,   -6.0955   ,  -4.6994  ,     3.282])
visualize_robot_collision_spheres(_curobo_fn, q1, isaac_world, _X_World_Base)
isaac_world.reset()
simulation_app.update()
def wait_for_input():
    global _running
    input("Press ENTER to quit visualization...\n")
    _running = False
    print("done")

_running = True
print("[INFO] Starting Isaac Sim viewer (interactive mode)")
# threading.Thread(target=wait_for_input, daemon=True).start()
# while simulation_app.is_running() and _running:
#     isaac_world.step(render=True)
#-----------------------------------------------------------------

q2 = np.array([1.4667,     -3.6893,      2.6451   ,   1.6978   ,  -1.5712   ,  -1.6414])
q3 = np.array([1.4665,     -3.6905 ,     2.8357    ,  1.7216    , -1.5712    , -1.6413])
q4 = np.array([1.4665,     -3.6905 ,     3.0357    ,  1.7216    , -1.5712    , -1.6413]) # SELF COLLISION WORKS YAY!
q_np = np.array([q1, q2, q3, q4])   # shape (4, 6) -----> collision check multiple collisions at once!
q_t = torch.tensor(q_np, dtype=torch.float32, device="cuda")
d_world, d_self = _curobo_fn.get_world_self_collision_distance_from_joints(q_t)
logger.debug(f"Checking collision for joint configuration: {q1} with gripper configuration: none")
logger.debug(f"Collision distances: d_world={d_world}, d_self={d_self}")
# q_sph = torch.tensor(
#     np.array([
#         [[[*sphere_pos_B, sphere_r]]],
#         [[[*(sphere_pos_B + np.array([1.0, 0., 0.])), sphere_r]]]
#     ]),  # shape = (2, 1, 1, 4)
#     dtype=tensor_args.dtype,
#     device=tensor_args.device
# )
#
# d = _robot_world.get_collision_distance(q_sph) # -----> between the spheres and the world, doesn't take robot into account! (at least in my tests)
# logger.debug(f"Sphere collision distances: {d} for {q_sph}")

# Add a table to the environment
# self._table_box = o3d.geometry.OrientedBoundingBox(center=np.array([1.32, 0.0, self._table_height_est - self.platform_height - self._table_height_est/2.]), extent=np.array([1.0, 1.5, self._table_height_est]),
#                                                    R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

print(f"xwb33: {_X_World_Base[2, 3]}")
# 1. Curobo
table_cuboid = Cuboid(
    name="table",
    pose=[0., 1.115/2.-_X_World_Base[2, 3], 1.0/2.+0.42, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
    dims=[1.5, 1.115, 1.0], # along x,y,z axes
)
cuboid_world.add_obstacle(table_cuboid) # only adds it to the list
_curobo_fn.update_world(cuboid_world) # updates entire collision checker world
### If the new world configuration has more glass_obstacles than initial cache,
# the collision cache will be recreated, breaking existing cuda graphs.
# This will lead to an exit with error if use_cuda_graph is enabled.



# 2. Isaac Sim
convert_cuboid_world_to_isaacsim(cuboid_world, isaac_world, _X_World_Base)

_running = True
print("[INFO] Starting Isaac Sim viewer (interactive mode)")
threading.Thread(target=wait_for_input, daemon=True).start()
while simulation_app.is_running() and _running:
    isaac_world.step(render=True)
#-----------------------------------------------------------------

q_np = np.array([q1, q2, q3, q4])   # shape (4, 6) -----> collision check multiple collisions at once!
q_t = torch.tensor(q_np, dtype=torch.float32, device="cuda")
d_world, d_self = _curobo_fn.get_world_self_collision_distance_from_joints(q_t)
logger.debug(f"Checking collision for joint configuration: {q1} with gripper configuration: none")
logger.debug(f"Collision distances: d_world={d_world}, d_self={d_self}")