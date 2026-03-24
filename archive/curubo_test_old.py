import numpy as np
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from loguru import logger
from scipy.spatial.transform import Rotation as R
from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld
from curobo.geom.types import WorldConfig, Sphere
import torch

from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType

import pyzed.sl as sl

def draw_points(voxels):
    # Third Party

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_points()
    if len(voxels) == 0:
        return

    jet = cm.get_cmap("plasma").reversed()

    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 1]
    # add smallest and largest values:
    # z_val = np.append(z_val, 1.0)
    # z_val = np.append(z_val,0.4)
    # scale values
    # z_val += 0.4
    # z_val[z_val>1.0] = 1.0
    # z_val = 1.0/z_val
    # z_val = z_val/1.5
    # z_val[z_val!=z_val] = 0.0
    # z_val[z_val==0.0] = 0.4

    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 1.0)]
    sizes = [10.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)

robot_arm_name = "ur5e"
robot_gripper_name = "robotiq_2f_85"
logger.info(f"Loading robot: {robot_arm_name} with gripper: {robot_gripper_name}...")

# Load robot configuration from yaml file
robot_cfg = load_yaml(join_path(get_robot_configs_path(), f"{robot_arm_name}.yml"))["robot_cfg"]

# spheres = []
# # 2. Add to objects list for Curobo
# sphere_pos = np.array([0.1, 0.1, 0.1])
# sphere_r = 8.0
# # apply world to base transform
# # sphere_pos_Base = np.linalg.inv(self._X_World_Base) @ np.hstack((sphere_pos, 1))
# sphere_pos_Base = np.hstack((sphere_pos, 1))
# print(f"sphere_pos_Base:{sphere_pos_Base[:3]}")
# print(f"pose: {[*sphere_pos_Base[:3], 1, 0, 0, 0]}")
# spheres.append(Sphere(
#     name="my_sphere",
#     pose=[*sphere_pos_Base[:3], 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
#     radius=sphere_r
# ))
# sphere_pos_Base[0] += 1.0
# spheres.append(Sphere(
#     name="my_sphere2",
#     pose=[*sphere_pos_Base[:3], 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
#     radius=sphere_r
# ))
# logger.info(f"Added sphere obstacle at {sphere_pos} (base frame: {sphere_pos_Base[:3]}) with radius {sphere_r}")



# # -------- Build WorldConfig for Curobo
# _world_config = WorldConfig(
#     sphere=spheres
# )



# # To use this world in a collision checker, we need to approximate some object types
# # as cuRobo currently only provides a collision checker for cuboids and meshes.
# # Capsules, cylinders, and spheres can be approximated to cuboids using
# cuboid_world = WorldConfig.create_obb_world(_world_config) #

world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.03,
                }
            }
        }
    )


# CuRobo robot for kinematics and collision checking
tensor_args = TensorDeviceType()
config = RobotWorldConfig.load_from_config(
    f"{robot_arm_name}.yml",
    world_cfg,
    collision_activation_distance=0.0,
    collision_checker_type = CollisionCheckerType.BLOX
)
model = RobotWorld(config)


# --- Init parameters ---
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.camera_resolution = sl.RESOLUTION.HD720
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP


# --- Positiona tracking parameters ---
positional_tracking_parameters = sl.PositionalTrackingParameters()

# --- Open camera ---
zed = sl.Camera()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open :", repr(status))
    exit(1)

# --- Enable positional tracking  ---
status = zed.enable_positional_tracking(positional_tracking_parameters)
if status > sl.ERROR_CODE.SUCCESS:
    print(f"Failed to enable positional tracking: {status}")
    zed.close()
    exit(1)

intrinsics = zed.get_camera_information().camera_configuration.calibration_parameters
intrinsics_left = intrinsics.left_cam

# Depth intrinsics (left camera)
fx = intrinsics_left.fx
fy = intrinsics_left.fy
cx = intrinsics_left.cx
cy = intrinsics_left.cy

# Create 3x3 intrinsic matrix
intrinsics_left_tensor = torch.tensor([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=torch.float32)

pose = sl.Pose()
depth_map = sl.Mat()
image = sl.Mat()

i = 0

while True:
    try:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            model.world_model.decay_layer("world")
            
            zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) # Get the camera pose
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Get the depth map

            translation = pose.get_translation().get()
            rotation = pose.get_rotation_matrix().r

            T_torch = torch.eye(4, dtype=torch.float32)
            T_torch[:3, :3] = torch.from_numpy(rotation).float()
            T_torch[:3, 3] = torch.from_numpy(translation).float()
            T_W_C = T_torch  # Camera pose in world frame

            depth_frame = torch.from_numpy(depth_map.get_data()).float()# .to('cuda')  # Depth frame as torch tensor

            data_camera = CameraObservation(
            depth_image=depth_frame, intrinsics = intrinsics_left_tensor, pose=T_W_C
            )

            data_camera = data_camera.to(device=model.tensor_args.device)
            # print(data_camera.depth_image, data_camera.rgb_image, data_camera.intrinsics)
            # print("got new message")
            model.world_model.add_camera_frame(data_camera, "world")
            # print("added camera frame")
            model.world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            model.world_model.update_blox_hashes()
            bounding = Cuboid("t", dims=[1, 1, 1], pose=[0, 0, 0, 1, 0, 0, 0])
            voxels = model.world_model.get_voxels_in_bounding_box(bounding, 0.025)

            draw_points(voxels)

    except KeyboardInterrupt:
        print("Stopping spatial mapping...")
        break


zed.disable_positional_tracking()
zed.close()


# q1 = np.array([1.5671,     -2.5338  ,    1.3717  ,  -0.41391   ,   1.5426     , 3.1205])
# q2 = np.array([1.4667,     -3.6893,      2.6451   ,   1.6978   ,  -1.5712   ,  -1.6414])
# q3 = np.array([1.4665,     -3.6905 ,     2.8357    ,  1.7216    , -1.5712    , -1.6413])
# q4 = np.array([1.4665,     -3.6905 ,     3.0357    ,  1.7216    , -1.5712    , -1.6413]) # SELF COLLISION WORKS YAY!
# q_np = np.array([q1, q2, q3, q4])   # shape (4, 6) -----> collision check multiple collisions at once!
# q_t = torch.tensor(q_np, dtype=torch.float32, device="cuda")
# d_world, d_self = _curobo_fn.get_world_self_collision_distance_from_joints(q_t)
# logger.debug(f"Checking collision for joint configuration: {q1} with gripper configuration: none")
# logger.debug(f"Collision distances: d_world={d_world}, d_self={d_self}")
# q_sph = torch.tensor(
#     np.array([
#         [[[*sphere_pos, sphere_r]]],
#         [[[*(sphere_pos + np.array([1.0, 0., 0.])), sphere_r]]]
#     ]),  # shape = (2, 1, 1, 4)
#     dtype=tensor_args.dtype,
#     device=tensor_args.device
# )

# d = _curobo_fn.get_collision_distance(q_sph) # -----> between the spheres and the world, doesn't take robot into account! (at least in my tests)
# logger.debug(f"Sphere collision distances: {d} for {q_sph}")