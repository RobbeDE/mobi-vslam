import time
from typing import List, Any

import numpy as np
import torch
from airo_planner import NoPathFoundError
# Airo
from airo_typing import JointConfigurationType, HomogeneousMatrixType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types import math
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
# CuRobo
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult
from loguru import logger
from numpy import ndarray, dtype, bool_
from scipy.spatial.transform import Rotation as R
from ur_analytic_ik_ext import ur5e

from airo_barista.control.AbstractTrajectory import AbstractTrajectory


class CuroboTrajectoryAdapter(AbstractTrajectory):
    def __init__(self, traj, interp_dt, motion_time):
        self._traj = traj  # list or tensor
        self._dt = interp_dt
        self._n_steps = self._traj.shape[0]
        # self._duration = motion_time
        self._duration = self._n_steps*self._dt
        self._speed_factor = 1.0  # 1.0 = original speed

    @staticmethod
    def from_curobo_traj_result(curobo_traj_result: MotionGenResult):
        return CuroboTrajectoryAdapter(
            curobo_traj_result.get_interpolated_plan().position.detach().cpu().numpy(),
            curobo_traj_result.interpolation_dt,
            curobo_traj_result.motion_time.item() if torch.is_tensor(curobo_traj_result.motion_time) else curobo_traj_result.motion_time.cpu()
        )

    @property
    def duration(self):
        if self.is_empty:
            return 0.0
        return self._duration / self._speed_factor

    @property
    def path_interval(self):
        t_start = 0.0
        t_end = self._duration
        return t_start, t_end

    @property
    def is_empty(self):
        return self._traj is None

    def sample(self, t):
        # Scale time to effectively speed up or slow down trajectory
        scaled_t = np.clip(t * self._speed_factor, 0, self._duration)
        idx = int(scaled_t / self._dt)
        idx = max(0, min(idx, self._n_steps - 1))
        return self._traj[idx]

    def retime(self, speed_factor: float):
        """
        Adjusts trajectory speed by scaling time.
        - speed_factor > 1.0 → faster (shorter duration)
        - speed_factor < 1.0 → slower (longer duration)
        """
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive and nonzero.")
        self._speed_factor = speed_factor
        return self  # allows chaining

    @property
    def speed_factor(self):
        """Returns the current speed factor."""
        return self._speed_factor

class RobotEnvironment:
    max_gripper_width = 0.085  # meters
    def __init__(self, robot_arm_name="ur5e", robot_gripper_name="robotiq_2f_85", robot_camera_name = "realsense_d435", init_q=None):
        # cuRobo assumes the world is represented with respect to the base frame of the robot from the kinematic chain
        # Robot base position and orientation
        # arm_pos = (0.3545, 0, 0.756) # measuring from floor to base gives 75.6cm
        arm_pos = (0.3545, 0, 0.746) # experimental observation gives 74.6cm
        arm_euler = (np.pi/2, 0., np.pi/2)
        self._X_World_Base = np.eye(4)  # World to Robot base transform
        self._X_World_Base[:3, 3] = np.array(arm_pos)
        self._X_World_Base[:3, :3] = R.from_euler("xyz", arm_euler).as_matrix()

        self._X_Tool0Tcp = np.eye(4)  # Tool0 to TCP transform
        self._X_Tool0Tcp[2, 3] = 0.175  # 17.5 cm from tool0 to TCP along z axis

        self.rack_pos_B = np.array([0.27+0.0625, -0.145-0.09-0.035-0.003, 0.03-0.3])
        self.rack_pos_0_0_B = self.rack_pos_B + np.array([0.0025, 0.1-0.035+0.006, 0.3-0.03-0.03]) # position of bottle 0,0 in base frame
        self.rack_hole_offset_B = np.array([0., 0., -0.096]) # offset from rack position to hole center in base frame
        self.bottle_obstacle_scale_bottom = np.array([0.055, 0.14, 0.055])  # approximate bottle as cuboid for collision checking
        self.bottle_obstacle_scale_top = np.array([0.025, 0.055, 0.025])    # approximate bottle as cuboid for collision checking
        self.bottle_top_pos_TCP_offset = np.array([0., -0.11, -0.02])  # offset from TCP to bottle top

        self._create_curobo_collision_world()
        self._create_curobo_robot(robot_arm_name, robot_gripper_name, robot_camera_name)

        self._attached_object_names = set()
        self._cuboid_obstacle_classes = {}
        
    def get_X_World_Base(self) -> HomogeneousMatrixType:
        """Get the homogeneous transformation matrix from the robot base frame to the world frame."""
        return self._X_World_Base

    def _create_curobo_collision_world(self):
        logger.info("Creating cuRobo collision world...")
        spheres = []
        # -------------------------------- XYZ axes for debugging
        # spheres.append(Sphere(
        #     name="my_spherex",
        #     pose=[1., 0., 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     radius=0.01,
        # ))
        # spheres.append(Sphere(
        #     name="my_spherey",
        #     pose=[0., 1., 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     radius=0.01,
        # ))
        # spheres.append(Sphere(
        #     name="my_spherez",
        #     pose=[0., 0., 1., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     radius=0.01,
        # ))

        cuboids = []
        # -------------------------------- Mobi cuboid representation
        cuboids.append(Cuboid(
            name="arm_attachment_base",
            pose=[0.0, 0.0, -0.015, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.175-0.01, 0.18-0.02, 0.04-0.02], # along x,y,z axes (-buffer because collisions are a bit conservative)
        ))
        rotation_wxyz = R.from_euler('xyz', [-np.deg2rad(35), 0, 0]).as_quat(scalar_first=True)
        cuboids.append(Cuboid(
            name="arm_attachment_left",
            pose=[0.0825-0.01, -0.005, -0.14, *rotation_wxyz], # in base frame (x, y, z, qw, qx, qy, qz) (-buffer because collisions are a bit conservative)
            dims=[0.01, 0.01, 0.28], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="arm_attachment_right",
            pose=[-(0.0825-0.01), -0.005, -0.14, *rotation_wxyz], # in base frame (x, y, z, qw, qx, qy, qz) (-buffer because collisions are a bit conservative)
            dims=[0.01, 0.01, 0.28], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="mobi",
            pose=[0., -0.4225, -0.375, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.54, 0.665, 0.72], # along x,y,z axes
        ))
        # cuboids.append(Cuboid(
        #     name="bottle_rack_left",
        #     pose=[*self.rack_pos_B, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     dims=[0.125, 0.07, 0.59], # along x,y,z axes
        # ))
        cuboids.append(Cuboid(
            name="bottle_rack_left_bottom",
            pose=[*(self.rack_pos_B+np.array([0., -0.035+0.003-0.005, 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.125, 0.006, 0.59], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="bottle_rack_left_sidel",
            pose=[*(self.rack_pos_B+np.array([0.0625-0.01, 0., 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.02, 0.07, 0.59], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="bottle_rack_left_sider",
            pose=[*(self.rack_pos_B+np.array([-0.0625+0.01+0.005, 0., 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.02, 0.07, 0.59], # along x,y,z axes
        ))
        # cuboids.append(Cuboid(
        #     name="bottle_rack_right",
        #     pose=[*(self.rack_pos_B*np.array([-1., 1., 1.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     dims=[0.125, 0.07, 0.59], # along x,y,z axes
        # ))
        cuboids.append(Cuboid(
            name="bottle_rack_right_bottom",
            pose=[*(self.rack_pos_B*np.array([-1., 1., 1.])+np.array([0., -0.035+0.003-0.005, 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.125, 0.006, 0.59], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="bottle_rack_right_sidel",
            pose=[*(self.rack_pos_B*np.array([-1., 1., 1.])+np.array([0.0625-0.01-0.005, 0., 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.02, 0.07, 0.59], # along x,y,z axes
        ))
        cuboids.append(Cuboid(
            name="bottle_rack_right_sider",
            pose=[*(self.rack_pos_B*np.array([-1., 1., 1.])+np.array([-0.0625+0.01-0.005, 0., 0.])), 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[0.02, 0.07, 0.59], # along x,y,z axes
        ))

        # # table_height=0.93
        # table_height=0.92
        # cuboids.append(Cuboid(
        #     name="table",
        #     pose=[0.0, -self._X_World_Base[2, 3]+table_height/2., 0.36+0.35, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     dims=[0.7, table_height, 0.7], # along x,y,z axes
        # ))
        #
        # # table_height=1.126
        # table_height=1.116
        # cuboids.append(Cuboid(
        #     name="table2",
        #     pose=[-0.35-0.35, -self._X_World_Base[2, 3]+table_height/2., 0.36+0.35, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
        #     dims=[0.7, table_height, 0.7], # along x,y,z axes
        # ))
        table_height=2.0 # set default table to avoid collisions when we haven't measured the real table height yet
        table_distance=0.36
        cuboids.append(Cuboid(
            name="table",
            pose=[0.0, -self._X_World_Base[2, 3]+table_height/2., table_distance+0.35, 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[2.0, table_height, 0.7], # along x,y,z axes
        ))

        # -------------------------------- World objects
        cuboids.append(Cuboid(
            name="ground",
            pose=[0., -self._X_World_Base[2, 3], 0., 1, 0, 0, 0], # in base frame (x, y, z, qw, qx, qy, qz)
            dims=[5.0, 0.04, 5.0], # along x,y,z axes (4cm thick gives 2cm error leeway above ground)
        ))

        # ... empty for now ...

        # -------- Build WorldConfig for Curobo
        _world_config = WorldConfig(
            sphere=spheres,
            cuboid=cuboids
        )

        # To use this world in a collision checker, we need to approximate some object types
        # as cuRobo currently only provides a collision checker for cuboids and meshes.
        # Capsules, cylinders, and spheres can be approximated to cuboids using
        self._cuboid_world = WorldConfig.create_obb_world(_world_config)

    def _create_curobo_robot(
            self,
            robot_arm_name,
            robot_gripper_name,
            robot_camera_name
    ):
        yaml_path = join_path(get_robot_configs_path(), f"{robot_arm_name}_{robot_gripper_name}_{robot_camera_name}.yml")
        logger.info(f"Loading robot: arm-{robot_arm_name}\tgripper-{robot_gripper_name}\tcamera-{robot_camera_name}\n from {yaml_path}...")

        self._tensor_args = TensorDeviceType()

        self._robot_cfg_dict = load_yaml(file_path=yaml_path) # dict

        # 1. Load robot configuration from YAML (RobotConfig -> CudaRobotModelConfig (.kinematics))
        self._robot_cfg = RobotConfig.from_dict(
            self._robot_cfg_dict,
            tensor_args=self._tensor_args
        )

        # 2. CuRobo robot for kinematics and collision checking (RobotWorldConfig -> WorldCollision (.world_model) -> CudaRobotModel (.kinematics))
        collision_world_config = RobotWorldConfig.load_from_config(
            robot_config=self._robot_cfg, # (RobotConfig)
            world_model=self._cuboid_world, # (WorldConfig)
            tensor_args=self._tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            collision_activation_distance=0.0
        )
        self._robot_world = RobotWorld(config=collision_world_config) # model can be updated by updating cuboid_world and then calling _robot_world.update_world(cuboid_world)
        logger.info(" > Created cuRobo RobotworldConfig -> RobotWorld for collision checking and kinematics (CudaRobotModel).")

        # 3. Motion Generation
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=self._robot_cfg, # (RobotConfig)
            world_model=self._cuboid_world, # (WorldConfig)
            tensor_args=self._tensor_args,
            world_coll_checker=collision_world_config.world_model, # (WorldCollision)
            collision_checker_type=CollisionCheckerType.PRIMITIVE
        )
        self._motion_gen = MotionGen(config=motion_gen_config) # model can be updated by updating cuboid_world and then calling _motion_gen.update_world(cuboid_world)
        # 4. IK Solver
        self._ik_solver = self._motion_gen.ik_solver # model can be updated by updating cuboid_world and then calling _ik_solver.update_world(cuboid_world)
        self._ik_batch_size = 16
        logger.info(" > Created cuRobo MotionGen for motion generation.")
        logger.info("  > Warming up cuRobo MotionGen CUDA kernels... (this can take up to 5 seconds)")
        self._motion_gen.warmup() # Preload CUDA kernels, can take up to 5 seconds ----> COMMENT OUT FOR QUICK HOT RELOADING
        logger.success(" > Done!")

        # # All 3 WorldCollision instances are the SAME!
        # logger.debug(f"_robot_world.world_model:  (RobotWorld -> WorldCollision, id={id(self._robot_world.world_model)})\n"
        #              f"_motion_gen.world_collision:  (MotionGen -> WorldCollision, id={id(self._motion_gen.world_collision)})\n" # or self._motion_gen.world_coll_checker
        #              f"_ik_solver.world_coll_checker:  (IKSolver -> WorldCollision, id={id(self._ik_solver.world_coll_checker)})")

        # # All 4 kinematics are DIFFERENT!
        # logger.debug(f"_robot_cfg.kinematics:  (RobotConfig -> CudaRobotModel, id={id(self._robot_cfg.kinematics)})\n"
        #              f"_curobo_fn.kinematics:  (RobotWorld -> CudaRobotModel, id={id(self._robot_world.kinematics)})\n"
        #              f"_ik_solver.kinematics:  (IKSolver -> CudaRobotModel, id={id(self._ik_solver.kinematics)})\n"
        #              f"_motion_gen.kinematics:  (MotionGen -> CudaRobotModel, id={id(self._motion_gen.kinematics)})")
        # # BUT all 4 kinematics_config are the SAME!
        # logger.debug(f"_robot_cfg.kinematics:  (RobotConfig -> CudaRobotModel, id={id(self._robot_cfg.kinematics.kinematics_config)})\n"
        #              f"_curobo_fn.kinematics:  (RobotWorld -> CudaRobotModel, id={id(self._robot_world.kinematics.kinematics_config)})\n"
        #              f"_ik_solver.kinematics:  (IKSolver -> CudaRobotModel, id={id(self._ik_solver.kinematics.kinematics_config)})\n"
        #              f"_motion_gen.kinematics:  (MotionGen -> CudaRobotModel, id={id(self._motion_gen.kinematics.kinematics_config)})")

    def set_gripper_pos(self, gripper_width: float):
        """Set the gripper width in the robot configuration.
        Parameters:
            gripper_width: Desired gripper width in meters.
        """
        if gripper_width < 0.0 or gripper_width > self.max_gripper_width:
            logger.warning(f"Gripper width {gripper_width} is out of bounds [0.0, {self.max_gripper_width}]. Clipping to valid range.")
            gripper_width = np.clip(gripper_width, 0.0, self.max_gripper_width)
        # Convert to range 0.0 (open) to 1.0 (closed)
        finger_joint_value = 1.0 - (gripper_width / self.max_gripper_width)
        logger.info(f"Setting gripper width to {gripper_width} meters (finger_joint value: {finger_joint_value}).")
        self._motion_gen.update_locked_joints(
            lock_joints={"finger_joint": finger_joint_value},
            robot_config_dict=self._robot_cfg_dict
        )

    def _add_cuboid_obstacle_to_cuboid_world(self, name: str, position: list[float], quat_wxyz: list[float], scale: list[float]):
        """Add a cuboid obstacle to the collision world.
        Parameters:
            name: Name of the cuboid obstacle.
            position: [x, y, z] position of the cuboid center in the robot base frame.
            quat_wxyz: [qw, qx, qy, qz] quaternion representing the cuboid orientation in the robot base frame.
            scale: [sx, sy, sz] dimensions of the cuboid along x, y
        """
        # First check if the obstacle already exists
        existing_obstacle = self._cuboid_world.get_obstacle(name)
        if existing_obstacle is not None:
            # logger.debug(f"Cuboid obstacle '{name}' already exists in the collision world, updating its pose and scale.")
            existing_obstacle.pose = [*position, *quat_wxyz]  # (x, y, z, qw, qx, qy, qz)
            existing_obstacle.dims = scale  # along x,y,z axes
            logger.info(f"Updated cuboid obstacle '{name}' in the collision world to pos {position}, quat {quat_wxyz}, scale {scale}.")
        else:
            # logger.debug(f"Cuboid obstacle '{name}' does not exist in the collision world, adding new obstacle.")
            cuboid = Cuboid(
                name=name,
                pose=[*position, *quat_wxyz],  # (x, y, z, qw, qx, qy, qz)
                dims=scale  # along x,y,z axes
            )
            self._cuboid_world.add_obstacle(cuboid)
            logger.info(f"Added cuboid obstacle '{name}' to the collision world at pos {position}, quat {quat_wxyz}, scale {scale}.")
        
    def add_cuboid_obstacles(self, cuboid_dict: dict[str, dict]):
        """Add multiple cuboid glass_obstacles to the collision world.
        Parameters:
            cuboid_dict: List of cuboid obstacle dictionaries with keys:
                - name: Name of the cuboid obstacle.
                - position: [x, y, z] position of the cuboid center in the robot base frame.
                - quat_wxyz: [qw, qx, qy, qz] quaternion representing the cuboid orientation in the robot base frame.
                - scale: [sx, sy, sz] dimensions of the cuboid along x, y, z axes.
        """
        for name, cuboid in cuboid_dict.items():
            position = cuboid["position"]
            quat_wxyz = cuboid["quat_wxyz"]
            scale = cuboid["scale"]
            self._add_cuboid_obstacle_to_cuboid_world(name, position, quat_wxyz, scale)

        # WorldCollision instances are the same, so updating one updates all
        # self._robot_world.update_world(self._cuboid_world)
        # self._ik_solver.update_world(self._cuboid_world)
        self._motion_gen.update_world(self._cuboid_world) # this one also passes `fix_cache_reference=self.use_cuda_graph` and resets graph planner buffer

        # self.update_visualization_world()
        # self.add_cuboid_obstacles_to_visualization_world(cuboid_dict) # disabled to avoid flooding visualization cmd_queue

    def _remove_cuboid_obstacle_from_coll_world(self, name: str):
        """Remove a cuboid obstacle from the collision world by name."""
        self._cuboid_world.remove_obstacle(name)
        # for some reason Curobo's WorldConfig does not also remove the obstacle from the cuboid list, so we have to do it manually
        for i in range(len(self._cuboid_world.cuboid)):
            if self._cuboid_world.cuboid[i].name == name:
                del self._cuboid_world.cuboid[i]
                break
        logger.info(f"Removed cuboid obstacle '{name}' from the collision world.")

    def remove_cuboid_obstacle(self, name: str):
        """Remove a cuboid obstacle from the collision world by name."""
        self._remove_cuboid_obstacle_from_coll_world(name)

        # WorldCollision instances are the same, so updating one updates all
        # self._robot_world.update_world(self._cuboid_world)
        # self._ik_solver.update_world(self._cuboid_world)
        self._motion_gen.update_world(self._cuboid_world) # this one also passes `fix_cache_reference=self.use_cuda_graph` and resets graph planner buffer

        # self.update_visualization_world()
        # self.remove_cuboid_obstacles_from_visualization_world(name) # disabled to avoid flooding visualization cmd_queue

    def update_cuboid_obstacle_class(self, class_name: str, cuboid_dict: dict[str, dict]):
        """Update multiple cuboid obstacles of a given class in the collision world.
            1. Adds new obstacles if they don't exist.
            2. Updates existing obstacles if they already exist.
            3. Removes obstacles of the class that are not in the provided cuboid_dict.

        Parameters:
            class_name: Class name of the cuboid obstacles to update.
            cuboid_dict: List of cuboid obstacle dictionaries with keys:
                - name: Name of the cuboid obstacle.
                - position: [x, y, z] position of the cuboid center in the robot base frame.
                - quat_wxyz: [qw, qx, qy, qz] quaternion representing the cuboid orientation in the robot base frame.
                - scale: [sx, sy, sz] dimensions of the cuboid along x, y, z axes.
        """
        if class_name not in self._cuboid_obstacle_classes:
            self._cuboid_obstacle_classes[class_name] = set()
        else:
            # Remove obstacles of this class that are not in the new cuboid_dict
            existing_names = self._cuboid_obstacle_classes[class_name]
            new_names = set(cuboid_dict.keys())
            names_to_remove = existing_names - new_names
            for name in names_to_remove:
                self._remove_cuboid_obstacle_from_coll_world(name)
                self._cuboid_obstacle_classes[class_name].remove(name)
        # Add or update obstacles from the new cuboid_dict
        for name, cuboid in cuboid_dict.items():
            self._cuboid_obstacle_classes[class_name].add(name)
            position = cuboid["position"]
            quat_wxyz = cuboid["quat_wxyz"]
            scale = cuboid["scale"]
            self._add_cuboid_obstacle_to_cuboid_world(name, position, quat_wxyz, scale)

        # WorldCollision instances are the same, so updating one updates all
        # self._robot_world.update_world(self._cuboid_world)
        # self._ik_solver.update_world(self._cuboid_world)
        self._motion_gen.update_world(self._cuboid_world)

        # self.update_visualization_world()
        # self.update_cuboid_obstacle_class_in_visualization_world(class_name, cuboid_dict) # disabled to avoid flooding visualization cmd_queue

    def attach_obstacle_to_robot(self, q: JointConfigurationType, name: str, pose_offset=None):
        """Attach an obstacle to the robot for collision checking.
        Parameters:
            q: Joint configuration of the robot at which to attach the obstacle.
            name: Name of the obstacle to attach.
            pose_offset: [x, y, z, qw, qx, qy, qz] pose offset of the obstacle relative to the robot end-effector frame.
        """
        self.attach_obstacles_to_robot(q, [name], pose_offset)
    def attach_obstacles_to_robot(self, q: JointConfigurationType, names: list[str], pose_offset=None):
        """Attach an obstacle to the robot for collision checking.
        Parameters:
            q: Joint configuration of the robot at which to attach the obstacle.
            names: Names of the obstacles to attach.
            pose_offset: [x, y, z, qw, qx, qy, qz] pose offset of the obstacles relative to the robot end-effector frame.
        """
        if pose_offset is None:
            pose_offset = [0, 0, 0, 1, 0, 0, 0]
        for name in names:
            obstacle = self._cuboid_world.get_obstacle(name)
            if obstacle is None:
                logger.error(f"Cannot attach obstacle '{name}' to robot: obstacle not found in the collision world.")
                return
        q_t = torch.tensor(q, dtype=torch.float32, device="cuda").unsqueeze(0)
        state = JointState.from_position(q_t)

        self._motion_gen.attach_objects_to_robot(
            joint_state=state,
            object_names=names,
            world_objects_pose_offset=Pose.from_list(pose_offset, tensor_args=self._tensor_args),
            sphere_fit_type = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
        )
        logger.info(f"Attached obstacles '{names}' to the robot at joint configuration {q} with pose offset {pose_offset}.")

        # self.update_visualization_world() # this doesn't work since the attached object is still in the world, but not enabled
        # instead, we have to manually disable it in the visualization world
        for name in names:
            self.disable_cuboid_obstacles_in_visualization_world(name)
            # and keep track of attached object names
            self._attached_object_names.add(name)

    def detach_from_robot(self):
        """Detach any attached glass_obstacles from the robot."""
        self._motion_gen.detach_object_from_robot()
        logger.info(f"Detached all glass_obstacles from the robot.")

        # self.update_visualization_world()
        for name in self._attached_object_names:
            # For some reason, CuRobo does not automatically re-enable the detached objects in the world collision checker
            if name in self._motion_gen.world_coll_checker.get_obstacle_names():
                self._motion_gen.world_coll_checker.enable_obstacle(enable=True, name=name)
            # This we have to do manually in the visualization world as well
            self.enable_cuboid_obstacles_in_visualization_world(name)
        self._attached_object_names.clear()

    def disable_obstacle(self, name: str):
        """Disable an obstacle in the collision world by name."""
        self._motion_gen.world_coll_checker.enable_obstacle(enable=False, name=name)
        logger.info(f"Disabled obstacle '{name}' in the collision world.")
    def enable_obstacle(self, name: str):
        """Enable an obstacle in the collision world by name."""
        self._motion_gen.world_coll_checker.enable_obstacle(enable=True, name=name)
        logger.info(f"Enabled obstacle '{name}' in the collision world.")

    def check_is_collision_free(self, q: JointConfigurationType) -> bool:
        """Check if the given joint configuration is collision-free."""
        q_t = torch.tensor(q, dtype=torch.float32, device="cuda").unsqueeze(0)
        d_world, d_self = self._robot_world.get_world_self_collision_distance_from_joints(q_t)
        # if not bool((d_world <= 0).all() and (d_self <= 0).all()):
        #     logger.debug(f"Checking collision for joint configuration: {q}")
        #     logger.debug(f"Collision distances: d_world={d_world}, d_self={d_self}")
        #     return False
        # return True
        return bool((d_world <= 0).all() and (d_self <= 0).all())

    def check_batch_is_collision_free(self, q_batch: List[JointConfigurationType] | np.ndarray) -> ndarray[Any, dtype[bool_]]:
        """Check if the given batch of joint configurations are collision-free."""
        time_s = time.monotonic()
        q_batch_t = torch.tensor(q_batch, dtype=torch.float32, device="cuda").contiguous()
        d_world, d_self = self._robot_world.get_world_self_collision_distance_from_joints(q_batch_t)
        logger.debug(f"Checked batch collision for {len(q_batch)} configurations in {(time.monotonic()-time_s)*1000:.2f} ms.")
        return (d_world+d_self).cpu().numpy() <= 0

    def get_collision_distance(self, p_spheres: np.ndarray, r_spheres: np.ndarray) -> np.ndarray:
        """Get the collision distance between the collisionworld and the given spheres.
        Parameters:
            p_spheres: (num_spheres, 3) position of the sphere centers in the robot base frame.
            r_spheres: (num_spheres,) radius of the spheres.
        Returns:
            collision distance: (num_spheres,) the largest penetration distance to the collision world for each sphere.
        """
        # q_sph = torch.tensor(
        #     np.array([
        #         [[[*p_spheres[0], r_spheres[0]]]],
        #         [[[*p_spheres[1], r_spheres[1]]]],
        #           ...
        #     ]),  # shape = (num_spheres, 1, 1, 4)
        #     dtype=tensor_args.dtype,
        #     device=tensor_args.device
        # )
        q_sph = torch.tensor(
            np.concatenate(
                [p_spheres, r_spheres[:, np.newaxis]],
                axis=1
            ).reshape(-1, 1, 1, 4),  # shape = (num_spheres, 1, 1, 4)
            dtype=self._tensor_args.dtype,
            device=self._tensor_args.device
        )
        d = self._robot_world.get_collision_distance(q_sph).squeeze()  # shape = (num_spheres,)
        return d.cpu().numpy()

    @staticmethod
    def _get_pose_from_homogeneous_matrix(hm: HomogeneousMatrixType) -> Pose:
        pos_xyz = hm[:3, 3]
        rot_mat = hm[:3, :3]
        quat_xyzw = R.from_matrix(rot_mat).as_quat()  # xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz

        pos_xyz_t = torch.tensor(pos_xyz, dtype=torch.float32, device="cuda").unsqueeze(0)
        quat_wxyz_t = torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda").unsqueeze(0)

        return Pose(pos_xyz_t, quat_wxyz_t)

    @staticmethod
    def _get_pose_from_homogeneous_matrix_batch(hm_batch: np.ndarray) -> Pose:
        rot_mats = hm_batch[:, :3, :3]
        quats_xyzw = R.from_matrix(rot_mats).as_quat()  # (n, 4) xyzw
        quats_wxyz = np.zeros_like(quats_xyzw)
        quats_wxyz[:, 0] = quats_xyzw[:, 3]
        quats_wxyz[:, 1:] = quats_xyzw[:, :3]

        pos_xyz_t = torch.tensor(hm_batch[:, :3, 3], dtype=torch.float32, device="cuda")
        quat_wxyz_t = torch.tensor(quats_wxyz, dtype=torch.float32, device="cuda")

        return Pose(pos_xyz_t, quat_wxyz_t)

    # ur5e implementation is preferred over curobo implementation for now
    # --------------------------------------------------------------
    # def inverse_kinematics(self, tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
    #     """Compute the inverse kinematics for the given end-effector pose."""
    #     goal_pose = self._get_pose_from_homogeneous_matrix(tcp_pose)
    #
    #     result = self._ik_solver.solve_single(goal_pose)
    #     if not result.success:
    #         logger.warning(f"No IK solution found for TCP pose: {tcp_pose}")
    #         return []
    #     solutions = result.solution[0].detach().cpu().numpy().astype(np.float64)
    #     return solutions
    #
    # def inverse_kinematics_closest_to(self, tcp_pose: HomogeneousMatrixType, q_seed: JointConfigurationType):
    #     """Compute the inverse kinematics for the given end-effector pose, returning the solution closest to q_seed."""
    #     logger.debug(f"Computing IK for TCP pose:\n"
    #                  f"{tcp_pose} with seed configuration:\n"
    #                  f"{q_seed}")
    #     goal_pose = self._get_pose_from_homogeneous_matrix(tcp_pose)
    #
    #     q_seed_t = torch.tensor(q_seed, dtype=torch.float32, device="cuda").unsqueeze(0)
    #
    #     # shape (1, dof)
    #     retract_config = q_seed_t
    #     # shape (n, 1, dof)
    #     seed_config = q_seed_t.unsqueeze(0)
    #
    #     result = self._ik_solver.solve_single(goal_pose, retract_config=retract_config, seed_config=seed_config)
    #     if not result.success:
    #         logger.warning(f"No IK solution found for TCP pose:\n"
    #                        f"{tcp_pose}")
    #         return None
    #     ik_sol = result.solution[0][0].detach().cpu().numpy().astype(np.float64)
    #     logger.debug(f"IK solution found: {ik_sol}")
    #     return ik_sol
    #
    # def _inverse_kinematics_closes_to_batch_internal(self, tcp_poses: np.ndarray, q_seed: JointConfigurationType) -> List[Optional[JointConfigurationType]]:
    #     goal_pose = self._get_pose_from_homogeneous_matrix_batch(np.array(tcp_poses))
    #
    #     # shape (batch, dof)
    #     retract_config = torch.tensor(
    #         np.array([q_seed] * len(tcp_poses)),
    #         dtype=torch.float32, device="cuda"
    #     )
    #
    #     # shape (n, batch, dof)
    #     seed_config = retract_config.repeat(len(tcp_poses), 1, 1)
    #
    #     result = self._ik_solver.solve_batch(goal_pose, retract_config=retract_config, seed_config=seed_config)
    #     solutions: List[Optional[JointConfigurationType]] = []
    #     for i in range(len(tcp_poses)):
    #         if not result.success[i]:
    #             logger.warning(f"No IK solution found for TCP pose:\n"
    #                            f"{tcp_poses[i]}")
    #             raise RuntimeError("No IK solution found in batch IK")
    #         else:
    #             ik_sol = result.solution[i].detach().cpu().numpy().astype(np.float64)[0]
    #             solutions.append(ik_sol)
    #     return solutions
    # def inverse_kinematics_closes_to_batch(self, tcp_poses: np.ndarray, q_seed: JointConfigurationType) -> List[Optional[JointConfigurationType]]:
    #     """Compute the inverse kinematics for the given batch of end-effector poses, returning the solution closest to each q_seed."""
    #     logger.debug(f"Computing batch IK for {len(tcp_poses)} TCP poses with seed configurations.")
    #     batch_size = self._ik_batch_size
    #     time_s = time.monotonic()
    #     solutions: List[Optional[JointConfigurationType]] = []
    #     for i in range(0, len(tcp_poses), batch_size):
    #         time_i = time.monotonic()
    #         if i + batch_size <= len(tcp_poses):
    #             tcp_pose_batch = tcp_poses[i:i+batch_size]
    #         else:
    #             tcp_pose_batch = tcp_poses[i:]
    #             # extend to get at least batch_size elements
    #             tcp_pose_batch = np.vstack([tcp_pose_batch] * (batch_size // len(tcp_pose_batch) + 1))[:batch_size]
    #         print(len(tcp_pose_batch))
    #         solutions_batch = self._inverse_kinematics_closes_to_batch_internal(tcp_pose_batch, q_seed)
    #         if i == 0:
    #             solutions = solutions_batch
    #         else:
    #             solutions.extend(solutions_batch)
    #         logger.debug(f"  Processed batch {i//batch_size + 1}/{(len(tcp_poses)-1)//batch_size + 1} of size {len(tcp_pose_batch)} in {(time.monotonic()-time_i)*1000:.2f} ms.")
    #     logger.debug(f"Batch IK for {len(tcp_poses)} poses completed in {(time.monotonic()-time_s)*1000:.2f} ms.")
    #     return solutions
    # --------------------------------------------------------------

    def inverse_kinematics_fn(self, tcp_pose: HomogeneousMatrixType) -> List[JointConfigurationType]:
        solutions = ur5e.inverse_kinematics_with_tcp(tcp_pose, self._X_Tool0Tcp)
        solutions = [solution.squeeze() for solution in solutions]

        if len(solutions) == 0:
            logger.warning(f"No IK solution found")
            return []
        return solutions

    def forward_kinematics_fn(self, q: JointConfigurationType) -> HomogeneousMatrixType:
        X_B_Tool0 = ur5e.forward_kinematics(q[0], q[1], q[2], q[3], q[4], q[5])
        X_B_Tcp = X_B_Tool0 @ self._X_Tool0Tcp
        return X_B_Tcp

    # Preferred over curobo IK since ur5e gives all 8 solutions, and is faster
    def inverse_kinematics_fn_wrt_start_pos(self, X_B_TcpTarget: HomogeneousMatrixType, start_pos: HomogeneousMatrixType, collision_check=True, filter_on_joint4_upright=False) -> List[JointConfigurationType]:
        solutions = ur5e.inverse_kinematics_with_tcp(X_B_TcpTarget, self._X_Tool0Tcp)
        solutions = [solution.squeeze() for solution in solutions]
        # logger.info(f"Found {len(solutions)} IK solutions for start_pos: {start_pos}")

        # if a calculated solution is off by 2pi compared with the start_pos, add / subtract 2pi
        for solution in solutions:
            for i in range(6):
                if solution[i] - start_pos[i] > np.pi and solution[i] - 2 * np.pi > - 2 * np.pi:
                    solution[i] -= 2 * np.pi
                elif solution[i] - start_pos[i] < -np.pi and solution[i] + 2 * np.pi < 2 * np.pi:
                    solution[i] += 2 * np.pi

        if collision_check:
            prev_sols_len = len(solutions)
            solutions = np.array(solutions)[self.check_batch_is_collision_free(solutions)]
            if len(solutions) != prev_sols_len:
                logger.info(f"Filtered out {prev_sols_len - len(solutions)} solutions because of collisions")
                if len(solutions) == 0 and prev_sols_len > 0:
                    logger.warning(f"Filtered out all solutions because of collisions")

        if filter_on_joint4_upright:
            prev_sols = solutions.copy()
            solutions = [solution for solution in solutions if np.dot(-ur5e.forward_kinematics(solution[0], solution[1], solution[2], solution[3], solution[4], np.pi)[:3, 1], np.array([0., 0., 1.])) >= 0.0]
            if len(solutions) != len(prev_sols):
                logger.info(f"Filtered out {len(prev_sols) - len(solutions)} solutions because joint 4 is not upright")
            for prev_sol in prev_sols:
                if prev_sol not in solutions:
                    logger.info(f"Filtered out solution: {prev_sol}: {np.dot(-ur5e.forward_kinematics(prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], prev_sol[4], np.pi)[:3, 1], np.array([0., 0., 1.]))}")
                else:
                    logger.info(f"Okay solution: {prev_sol}: {np.dot(-ur5e.forward_kinematics(prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], prev_sol[4], np.pi)[:3, 1], np.array([0., 0., 1.]))}")

        solutions = sorted(solutions, key=lambda x: np.linalg.norm(x - start_pos))
        # for solution in solutions:
        #     logger.info(f"Filtered solution: {solution} (dist to start: {np.linalg.norm(solution - start_pos):.4f})")
        if len(solutions) == 0:
            # logger.warning(f"No IK solution found for start_pos: {start_pos}")
            return []
        return solutions
    def inverse_kinematics_fn_wrt_start_pos_batch(self, X_B_TcpTargets: np.ndarray, start_poses: np.ndarray, collision_check=True, filter_on_joint4_upright=False) -> List[List[JointConfigurationType]]:
        batch_solutions = []
        for X_B_TcpTarget, start_pos in zip(X_B_TcpTargets, start_poses):
            # logger.debug(f"Computing IK for target:\n{X_B_TcpTarget}\nwith start pos:\n{start_pos}")
            batch_solutions.append(
                self.inverse_kinematics_fn_wrt_start_pos(
                    X_B_TcpTarget, start_pos,
                    collision_check=False,
                    filter_on_joint4_upright=filter_on_joint4_upright
                )
            )
        if collision_check:
            # Flatten all solutions into a single array
            all_solutions = np.concatenate(batch_solutions, axis=0)
            prev_sols_len = len(all_solutions)

            # Perform ONE batched collision check
            mask = self.check_batch_is_collision_free(all_solutions)

            # Reshape mask
            mask = np.split(mask, np.cumsum([len(sols) for sols in batch_solutions])[:-1])
            # Apply mask to each batch solution
            for i in range(len(batch_solutions)):
                batch_solutions[i] = np.array(batch_solutions[i])[mask[i]]

            # Log how many got filtered out
            total_filtered = prev_sols_len - sum(len(sols) for sols in batch_solutions)
            if total_filtered > 0:
                logger.info(f"Filtered out {total_filtered} solutions because of collisions")

            # Log if all got filtered out
            if len(batch_solutions) == 0 and prev_sols_len > 0:
                logger.warning("Filtered out all solutions because of collisions")

        return batch_solutions


    def plan_to_joint_configuration_curobo(
                self,
                q_start: JointConfigurationType,
                q_target: JointConfigurationType,
                orientation_lock=None, # TCP frame
                movement_axes_lock=None, # TCP frame
                time_dilation_factor: float=1.0
        ):
        """Plan a path and trajectory from q_start to q_target using cuRobo's motion generation."""
        if orientation_lock is None:
            orientation_lock = [0, 0, 0]
        if movement_axes_lock is None:
            movement_axes_lock = [0, 0, 0]
        plan_time_start = time.monotonic()
        q_start_t = torch.tensor(q_start, dtype=torch.float32, device="cuda").unsqueeze(0)
        q_target_t = torch.tensor(q_target, dtype=torch.float32, device="cuda").unsqueeze(0)

        start_state = JointState.from_position(q_start_t)
        target_state = JointState.from_position(q_target_t)

        hold_vec_weight_t = torch.tensor([*orientation_lock, *movement_axes_lock], device="cuda").unsqueeze(0)[0]

        if np.sum(orientation_lock+movement_axes_lock) > 0:
            out = self._robot_world.kinematics.get_state(q_target_t)
            goal_pose = Pose(out.ee_position, out.ee_quaternion)

            print(hold_vec_weight_t)

            result = self._motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    timeout=.5,
                    time_dilation_factor=time_dilation_factor,
                    pose_cost_metric=PoseCostMetric(
                        hold_partial_pose=True,
                        hold_vec_weight=hold_vec_weight_t,
                        project_to_goal_frame=True
                    )
                )
            )
        else:
            result = self._motion_gen.plan_single_js(
                start_state,
                target_state,
                MotionGenPlanConfig(
                    timeout=.5,
                    time_dilation_factor=time_dilation_factor,
                )
            )
        if not result.success:
            logger.error(f"[CUROBO] Motion generation plan_single_js failed: {result.status}")
            raise NoPathFoundError(q_start, q_target, f"[CUROBO] Motion generation plan_single_js failed: {result.status}")
        plan_time = time.monotonic() - plan_time_start

        motion_time = result.motion_time.item() if torch.is_tensor(result.motion_time) else result.motion_time
        plan_time = plan_time.item() if torch.is_tensor(plan_time) else plan_time
        logger.success(f"[CUROBO] Motion generation successful in: {plan_time:.4f} seconds, trajectory duration: {motion_time:.2f} seconds")

        return CuroboTrajectoryAdapter.from_curobo_traj_result(result)

    def plan_linear_curobo(
            self,
            q_start: JointConfigurationType,
            offset_position: float,
            linear_axis: int,
            project_to_TCP: bool = True,
            time_dilation_factor: float=1.0
    ):
        """Plan a linear path and trajectory from q_start along the specified linear axis using cuRobo's motion generation."""
        plan_time_start = time.monotonic()
        q_start_t = torch.tensor(q_start, dtype=torch.float32, device="cuda").unsqueeze(0)

        start_state = JointState.from_position(q_start_t)

        out = self._robot_world.kinematics.get_state(q_start_t)

        goal_position = out.ee_position.clone()
        goal_position[0, linear_axis] += offset_position
        print(goal_position)

        # goal_pose = Pose(goal_position, out.ee_quaternion)
        offset_vector = torch.zeros_like(out.ee_position)
        offset_vector[0, linear_axis] = offset_position
        # Rotate offset into base frame using the EE quaternion
        R_ee = math.quaternion_to_matrix(out.ee_quaternion)  # shape: (1, 3, 3)
        offset_world = torch.matmul(R_ee, offset_vector.unsqueeze(-1)).squeeze(-1)
        goal_position = out.ee_position + offset_world
        print(goal_position)
        goal_pose = Pose(goal_position, out.ee_quaternion)


        hold_vec_weight = [1, 1, 1, 1, 1, 1]
        hold_vec_weight[linear_axis + 3] = 0  # allow movement along the specified linear axis
        hold_vec_weight_t = torch.tensor(hold_vec_weight, device="cuda").unsqueeze(0)[0]

        result = self._motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                timeout=.5,
                time_dilation_factor=time_dilation_factor,
                pose_cost_metric=PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=hold_vec_weight_t,
                    project_to_goal_frame=project_to_TCP
                )
            )
        )
        if not result.success:
            logger.error(f"[CUROBO] Linear motion generation plan_single_js failed: {result.status}")
            raise NoPathFoundError(q_start, offset_position, f"[CUROBO] Linear motion generation plan_single_js failed: {result.status}")
        plan_time = time.monotonic() - plan_time_start

        motion_time = result.motion_time.item() if torch.is_tensor(result.motion_time) else result.motion_time
        plan_time = plan_time.item() if torch.is_tensor(plan_time) else plan_time
        logger.success(f"[CUROBO] Linear motion generation successful in: {plan_time:.4f} seconds, trajectory duration: {motion_time:.2f} seconds")

        return CuroboTrajectoryAdapter.from_curobo_traj_result(result)

    def plan_to_joint_configuration_sequence_curobo(
            self,
            q_start: JointConfigurationType,
            q_target_seq: list[JointConfigurationType],
            time_dilation_factor: float=1.0
    ):
        """Plan a path and trajectory from q_start through each configuration in q_target_seq using cuRobo's motion generation."""
        plan_time_start = time.monotonic()
        q_start_t = torch.tensor(q_start, dtype=torch.float32, device="cuda").unsqueeze(0)
        q_target_seq_t = [torch.tensor(q_target, dtype=torch.float32, device="cuda").unsqueeze(0) for q_target in q_target_seq]

        start_state = JointState.from_position(q_start_t)
        target_seq_state = [JointState.from_position(q_target_t) for q_target_t in q_target_seq_t]

        #TODO: use some plan batch for batch planning / result.interpolated_plan to manually stitch together + toppra timing
        # return NotImplementedError("cuRobo motion generation for joint configuration sequences is not yet implemented.")

        full_plan_pos = None
        full_duration = 0.0
        interp_dt = None
        for i, target_t in enumerate(target_seq_state):
            logger.info(f"[CUROBO] Planning motion to subgoal {i+1}/{len(target_seq_state)}...")
            plan_time_start_i = time.monotonic()
            result = self._motion_gen.plan_single_js(start_state, target_t, MotionGenPlanConfig(timeout=.5, time_dilation_factor=time_dilation_factor))
            if not result.success:
                logger.error(f"[CUROBO] Motion generation failed: {result.status}")
                raise NoPathFoundError(q_start, q_target_seq, f"[CUROBO] Motion generation failed: {result.status}")
            plan_time_i = time.monotonic() - plan_time_start_i
            motion_time_i = result.motion_time.item() if torch.is_tensor(result.motion_time) else result.motion_time
            plan_time_i = plan_time_i.item() if torch.is_tensor(plan_time_i) else plan_time_i
            logger.success(f"[CUROBO] Motion generation successful in: {plan_time_i:.4f} seconds, trajectory duration: {motion_time_i:.2f} seconds")

            # Get plan for this segment
            if full_plan_pos is None:
                full_plan_pos = result.get_interpolated_plan().position
            else:
                # we get rid of the overlapping points because velocities and accelerations are zero at the start and end of each segment
                full_plan_pos = torch.cat((full_plan_pos[:-4], result.get_interpolated_plan().position[4:]), dim=0)
            full_duration += motion_time_i
            interp_dt = result.interpolation_dt

            # Update start state for next segment
            start_state = target_seq_state[i]

        return CuroboTrajectoryAdapter(full_plan_pos.detach().cpu().numpy(), interp_dt, full_duration)
    
    def plan_to_tcp_pose_curobo(
            self,
            q_start: JointConfigurationType,
            tcp_pose_target: HomogeneousMatrixType,
            time_dilation_factor: float=1.0
    ):        
        """Plan a path and trajectory from q_start to the joint configuration that achieves tcp_pose_target using cuRobo's motion generation."""
        plan_time_start = time.monotonic()
        q_start_t = torch.tensor(q_start, dtype=torch.float32, device="cuda").unsqueeze(0)

        start_state = JointState.from_position(q_start_t)
        goal_pose = Pose.from_matrix(tcp_pose_target)

        result = self._motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(timeout=.5, time_dilation_factor=time_dilation_factor))
        if not result.success:
            logger.error(f"[CUROBO] Motion generation plan_single failed: {result.status}")
            raise NoPathFoundError(q_start, tcp_pose_target, f"[CUROBO] Motion generation plan_single failed: {result.status}")
        plan_time = time.monotonic() - plan_time_start

        motion_time = result.motion_time.item() if torch.is_tensor(result.motion_time) else result.motion_time
        plan_time = plan_time.item() if torch.is_tensor(plan_time) else plan_time
        logger.success(f"[CUROBO] Motion generation successful in: {plan_time:.4f} seconds, trajectory duration: {motion_time:.2f} seconds")

        return CuroboTrajectoryAdapter.from_curobo_traj_result(result)

    def plan_to_joint_configuration_grasp_approach_curobo(
            self,
            q_start: JointConfigurationType,
            q_target: JointConfigurationType,
            offset_position: float,
            linear_axis: int, #0=x,1=y,2=z in base frame
            tstep_fraction: float, #timestep fraction to start activating this constraint
            time_dilation_factor: float=1.0
    ):
        """Plan a path and trajectory from q_start to q_target using cuRobo's motion generation."""
        plan_time_start = time.monotonic()
        q_start_t = torch.tensor(q_start, dtype=torch.float32, device="cuda").unsqueeze(0)
        q_target_t = torch.tensor(q_target, dtype=torch.float32, device="cuda").unsqueeze(0)

        start_state = JointState.from_position(q_start_t)

        out = self._robot_world.kinematics.get_state(q_target_t)
        goal_pose = Pose(out.ee_position, out.ee_quaternion)

        pose_cost_metric=PoseCostMetric.create_grasp_approach_metric(
            offset_position=offset_position,
            linear_axis=linear_axis,
            tstep_fraction=tstep_fraction,
            project_to_goal_frame=True,
            tensor_args=self._tensor_args
        )

        result = self._motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                timeout=.5,
                time_dilation_factor=time_dilation_factor,
                pose_cost_metric=pose_cost_metric
            )
        )
        if not result.success:
            logger.error(f"[CUROBO] Grasp approach motion generation plan_single_js failed: {result.status}")
            raise NoPathFoundError(q_start, q_target, f"[CUROBO] Grasp approach motion generation plan_single_js failed: {result.status}")
        plan_time = time.monotonic() - plan_time_start

        motion_time = result.motion_time.item() if torch.is_tensor(result.motion_time) else result.motion_time
        plan_time = plan_time.item() if torch.is_tensor(plan_time) else plan_time
        logger.success(f"[CUROBO] Grasp approach motion generation successful in: {plan_time:.4f} seconds, trajectory duration: {motion_time:.2f} seconds")

        return CuroboTrajectoryAdapter.from_curobo_traj_result(result)

    def stop(self):
        """Stop the robot environment and any associated processes."""
        pass

    def visualize_joint_configuration(self, q, interactive=True):
        """Visualize the robot at the given joint configuration."""
        logger.warning("Visualizing robot in non-visual environment. SKIPPING...")

    def visualize_trajectory(self, trajectory, interactive=True, choose_visualization_time=False):
        """Visualize the robot following the given trajectory."""
        logger.warning("Visualizing robot in non-visual environment. SKIPPING...")

    def update_visualization_world(self):
        """Update the visualization of the robot environment."""
        logger.warning("Updating visualization world in non-visual environment. SKIPPING...")
    
    def add_cuboid_obstacles_to_visualization_world(self, cuboid_dict: dict[dict]):
        """Add cuboid glass_obstacles to the visualization world."""
        logger.warning("Adding cuboid glass_obstacles to visualization world in non-visual environment. SKIPPING...")

    def remove_cuboid_obstacles_from_visualization_world(self, name: str):
        """Remove cuboid glass_obstacles from the visualization world by name."""
        logger.warning("Removing cuboid glass_obstacles from visualization world in non-visual environment. SKIPPING...")

    def update_cuboid_obstacle_class_in_visualization_world(self, class_name: str, cuboid_dict: dict[dict]):
        """Update cuboid obstacles of a given class in the visualization world."""
        logger.warning("Updating cuboid glass_obstacles in visualization world in non-visual environment. SKIPPING...")

    def disable_cuboid_obstacles_in_visualization_world(self, name: str):
        """Disable cuboid glass_obstacles in the visualization world by name."""
        logger.warning("Disabling cuboid glass_obstacles in visualization world in non-visual environment. SKIPPING...")
    def enable_cuboid_obstacles_in_visualization_world(self, name: str):
        """Enable cuboid glass_obstacles in the visualization world by name."""
        logger.warning("Enabling cuboid glass_obstacles in visualization world in non-visual environment. SKIPPING...")

    def visualize_B_frame(self, X_B_Frame: HomogeneousMatrixType, frame_name: str = "B_frame"):
        """Visualize a coordinate frame in the robot environment."""
        logger.warning("Visualizing B frame in non-visual environment. SKIPPING...")