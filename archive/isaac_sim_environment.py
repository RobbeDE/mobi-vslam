import time

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.viewports as viewport_utils
import numpy as np
import torch
from airo_typing import HomogeneousMatrixType
from isaacsim.core.prims import XFormPrim
from isaacsim.util.debug_draw import _debug_draw
from loguru import logger
from pxr import Gf
from scipy.spatial.transform import Rotation as R

from airo_barista.control.AbstractTrajectory import AbstractTrajectory


class IsaacsimEnvironment:
    def __init__(self, isaac_world, X_World_Base):
        self.isaac_world = isaac_world
        self.num_glasses = 0
        self.X_World_Base = X_World_Base
        # Add a default light source if none exists
        light_prim_path = "/World/DefaultLight"
        if prim_utils.is_prim_path_valid(light_prim_path):
            logger.warning(f"Default light already exists at {light_prim_path}, deleting and recreating it.")
            prim_utils.delete_prim(light_prim_path)
        if not prim_utils.is_prim_path_valid(light_prim_path):
            light_prim = prim_utils.create_prim(
                prim_path=light_prim_path,
                prim_type="RectLight",
                position=[0, 0, 2.5],
                orientation=[1, 0, 0, 0],  # wxyz quaternion
            )
            if light_prim is None:
                logger.error("Failed to add default light to Isaac Sim")
            else:
                light_prim.GetAttribute("inputs:intensity").Set(50000.0)
                light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 1.0, 1.0))
                light_prim.GetAttribute("inputs:width").Set(1.0)
                light_prim.GetAttribute("inputs:height").Set(1.0)
                logger.info("Added default light to Isaac Sim")
        # -------- SpotLight 1: Red --------
        self.create_spotlight(
            path="/World/SpotLight1",
            position=[2, -2, 2],
            orientation=[1, 0, 0, 0],
            intensity=4000000.0,
            color=[1.0, 0.2, 0.2]  # red
        )

        # -------- SpotLight 2: Green --------
        self.create_spotlight(
            path="/World/SpotLight2",
            position=[-2, -2, 2],
            orientation=[1, 0, 0, 0],
            intensity=4000000.0,
            color=[0.2, 1.0, 0.2]  # green
        )

        # -------- SpotLight 3: Blue --------
        self.create_spotlight(
            path="/World/SpotLight3",
            position=[0, 2, 2],
            orientation=[1, 0, 0, 0],
            intensity=4000000.0,
            color=[0.2, 0.2, 1.0]  # blue
        )

        # Set default camera view
        print(f"viewport: {viewport_utils.get_viewport_names()}")
        viewport_utils.set_camera_view(
            eye=[-0.75, 2.25, 1.5],
            target=[0.75, 0.0, 0.6],
        )

        self.object_names = set() # To keep track of added objects
        stage = prim_utils.get_current_stage()
        for prim in stage.TraverseAll():
            if "cuboid" in prim.GetName().lower():
                object_name = prim.GetPath().pathString
                self.object_names.add(object_name)
        for prim_path in self.object_names:
            prim_utils.delete_prim(prim_path)
        self.object_names.clear()

        self._cuboid_obstacle_classes = {}
        self._viz_frames = {}
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._draw.clear_lines()

    # Function to safely create a spotlight
    @staticmethod
    def create_spotlight(path, position, orientation, intensity, color, cone_angle=30.0):
        if prim_utils.is_prim_path_valid(path):
            logger.warning(f"Spotlight already exists at {path}, deleting and recreating it.")
            prim_utils.delete_prim(path)
        spot = prim_utils.create_prim(
            prim_path=path,
            prim_type="SphereLight",  # SpotLights in Isaac Sim can be emulated using SphereLight + angle
            position=position,
            orientation=orientation
        )
        if spot is not None:
            spot.GetAttribute("inputs:intensity").Set(intensity)
            spot.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))
            spot.GetAttribute("inputs:radius").Set(0.1)  # small radius to mimic spotlight
            logger.info(f"Added spotlight at {path} with intensity {intensity} and color {color}")
        return spot

    def _add_cuboid(self, prim_path, pos_B, quat_B, size):
        self.object_names.add(prim_path)
        # Transform to world frame
        pos_W = (self.X_World_Base @ np.hstack((pos_B, 1)))[:3]
        quat_W = (R.from_matrix(self.X_World_Base[:3, :3]) * R.from_quat(quat_B, scalar_first=True)).as_quat(scalar_first=True)
        # logger.info(f"cuboid {prim_path}: pos_W: {pos_W}, quat_W: {quat_W}, size: {size}")

        if not prim_utils.is_prim_path_valid(prim_path):
            cube_prim = prim_utils.create_prim(
                prim_path=prim_path,
                prim_type="Cube",
                position=pos_W,
                orientation=quat_W,
                scale=size
            )
            if cube_prim is None:
                logger.error(f"Failed to add cuboid {prim_path} to Isaac Sim")
            else:
                logger.info(f"Added cuboid {prim_path} to Isaac Sim at pos {pos_W}, quat {quat_W}, size {size}")
        else:
            # Update existing prim
            xform_prim = XFormPrim(prim_path)
            xform_prim.set_local_poses(translations=np.array([pos_W]), orientations=np.array([quat_W]))  # wxyz quaternion
            xform_prim.set_local_scales(np.array([size]))
            logger.info(f"Updated cuboid {prim_path} in Isaac Sim to pos {pos_W}, quat {quat_W}, size {size}")

    def _convert_cuboid_world_to_isaacsim(self, cuboid_world):
        new_object_names = set()
        old_object_names = self.object_names.copy()
        prev_object_names = self.object_names.copy()
        for i, cuboid in enumerate(cuboid_world.cuboid):
            #TODO: technically we should also check if the object is enabled for collisions or not
            pos_B = cuboid.pose[:3]
            quat_B = cuboid.pose[3:]
            size = [dim/2. for dim in cuboid.dims]
            prim_path = f"/World/cuboid_{cuboid.name}" if cuboid.name else f"/World/cuboid_{i}"
            self._add_cuboid(prim_path, pos_B, quat_B, size)  # wxyz quaternion
            new_object_names.add(prim_path)
            old_object_names.discard(prim_path)
        logger.info(f"Total objects in cuboid world: {len(new_object_names)}\n"
                    f" > {len(prev_object_names & new_object_names)} existing objects retained.\n"
                    f" > {len(new_object_names - prev_object_names)} new objects added: {new_object_names - prev_object_names}.\n")
        if len(old_object_names) > 0:
            # Remove any old objects that are no longer in the cuboid world
            logger.warning(f"Removing {len(old_object_names)} old objects from Isaac Sim: {old_object_names}")
            for old_prim_path in old_object_names:
                prim_utils.delete_prim(old_prim_path)
                if old_prim_path in self.object_names:
                    self.object_names.remove(old_prim_path)

    def _visualize_robot_collision_spheres(self, q: np.ndarray, tensor_args, curobo_fn):
        # Compute forward kinematics to get the robot base pose in the world frame
        q = torch.tensor(q, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
        cuda_robot_model_state = curobo_fn.kinematics.get_state(q)
        collision_spheres = cuda_robot_model_state.link_spheres_tensor
        ee_pos_B = cuda_robot_model_state.ee_position
        ee_quat_B = cuda_robot_model_state.ee_quaternion
        # Transform end-effector pose to world frame
        ee_pos_B_np = ee_pos_B[0].detach().cpu().numpy()
        ee_quat_B_np = ee_quat_B[0].detach().cpu().numpy()
        pos_W = (self.X_World_Base @ np.hstack((ee_pos_B_np, 1)))[:3]
        quat_W = (R.from_matrix(self.X_World_Base[:3, :3]) * R.from_quat(ee_quat_B_np, scalar_first=True)).as_quat(scalar_first=True)
        self._visualize_pose(pos_W, quat_W, "curobo_ee_frame")
        # print(f"collision_spheres: {collision_spheres}")
        # Visualize the robot collision spheres in Isaac Sim
        for i, sphere in enumerate(collision_spheres[0]):
            prim_path = f"/World/robot_sphere_{i}"
            pos_B = sphere[:3].detach().cpu().numpy()
            r = sphere[3].detach().cpu().numpy()
            if r < 1e-4:
                # logger.warning(f"Skipping robot collision sphere {i} with negligible radius {r}")
                if prim_utils.is_prim_path_valid(prim_path):
                    logger.info(f"Removing robot collision sphere {prim_path} with negligible radius {r} from Isaac Sim")
                    prim_utils.delete_prim(prim_path)
                continue
            # Transform to world frame
            pos_W = (self.X_World_Base @ np.hstack((pos_B, 1)))[:3]
            # print(f"robot sphere {i}: pos_W: {pos_W}, r: {r}")

            if not prim_utils.is_prim_path_valid(prim_path):
                # Create sphere only once
                sphere_prim = prim_utils.create_prim(
                    prim_path=prim_path,
                    prim_type="Sphere",
                    position=pos_W,
                    scale=[r]*3,
                )
                if sphere_prim is None:
                    logger.error(f"Failed to add robot collision sphere {i} to Isaac Sim")
                    continue
            else:
                # Update existing prim
                xform_prim = XFormPrim(prim_path)
                xform_prim.set_local_poses(translations=np.array([pos_W]), orientations=np.array([[1, 0, 0, 0]]))  # wxyz quaternion
                xform_prim.set_local_scales(np.array([[r]*3]))
                # print(f"Updated robot sphere {i} in Isaac Sim to pos {pos_W}, r {r}")

        # Remove any old spheres that are no longer needed
        stage = prim_utils.get_current_stage()
        for prim in stage.TraverseAll():
            if "robot_sphere_" in prim.GetName():
                sphere_index = int(prim.GetName().split("robot_sphere_")[-1])
                if sphere_index >= len(collision_spheres[0]):
                    prim_path = prim.GetPath().pathString
                    logger.info(f"Removing old robot collision sphere {prim_path} from Isaac Sim")
                    prim_utils.delete_prim(prim_path)


    def visualize_joint_configuration(self, q, tensor_args, curobo_fn):
        """Visualize the robot at the given joint configuration."""
        logger.info(f"Visualizing joint configuration:\n{q}")
        self._visualize_robot_collision_spheres(q, tensor_args, curobo_fn)

    def visualize_trajectory(self, trajectory: AbstractTrajectory, tensor_args, curobo_fn, visualization_time):
        """Visualize the robot following the given trajectory."""
        # Iterate through the trajectory and visualize each joint configuration
        logger.info(f"Visualizing trajectory with duration {trajectory.duration:.2f} seconds")

        visualization_speed = trajectory.duration / visualization_time
        start_time = time.monotonic()
        sim_time = 0.0

        while sim_time < trajectory.duration:
            # Compute elapsed wall-clock time
            elapsed_real_time = time.monotonic() - start_time
            t = min(elapsed_real_time * visualization_speed, trajectory.duration)

            q = trajectory.sample(t)
            self._visualize_robot_collision_spheres(q, tensor_args, curobo_fn)
            self.isaac_world.step(render=True)

            sim_time = t

        logger.info("Finished visualizing trajectory.")

    def update_visualization_world(self, cuboid_world):
        """Update the visualization world in Isaac Sim."""
        logger.info("Updating visualization world in Isaac Sim...")
        self._convert_cuboid_world_to_isaacsim(cuboid_world)

    def add_cuboid_obstacles_to_visualization_world(self, cuboid_dict: dict[dict]):
        """Add cuboid glass_obstacles to the visualization world in Isaac Sim."""
        logger.info(f"Adding {len(cuboid_dict)} cuboid obstacles to visualization world in Isaac Sim...")
        for i, (name, cuboid) in enumerate(cuboid_dict.items()):
            pos_B = cuboid["position"]
            quat_B = cuboid["quat_wxyz"]
            size = [dim/2. for dim in cuboid["scale"]]

            prim_path = f"/World/cuboid_{name}" if name else f"/World/cuboid_{i}"
            self._add_cuboid(prim_path, pos_B, quat_B, size)  # wxyz quaternion

    def remove_cuboid_obstacles_from_visualization_world(self, name: str):
        """Remove cuboid glass_obstacles from the visualization world in Isaac Sim."""
        logger.info(f"Removing cuboid glass_obstacles with name containing '{name}' from visualization world in Isaac Sim...")
        stage = prim_utils.get_current_stage()
        for prim in stage.TraverseAll():
            if "cuboid" in prim.GetName().lower() and name.lower() in prim.GetName().lower():
                prim_path = prim.GetPath().pathString
                logger.info(f"Removing cuboid obstacle {prim_path} from Isaac Sim")
                prim_utils.delete_prim(prim_path)
                self.object_names.discard(prim_path)
                return

    def update_cuboid_obstacle_class_in_visualization_world(self, class_name: str, cuboid_dict: dict[dict]):
        """Update cuboid obstacles of a given class in the visualization world."""
        if class_name not in self._cuboid_obstacle_classes:
            self._cuboid_obstacle_classes[class_name] = set()
        else:
            # Remove existing obstacles of this class
            existing_names = self._cuboid_obstacle_classes[class_name]
            new_names = set(cuboid_dict.keys())
            names_to_remove = existing_names - new_names
            for name in names_to_remove:
                self.remove_cuboid_obstacles_from_visualization_world(name)
                self._cuboid_obstacle_classes[class_name].remove(name)
        # Add or update obstacles of this class
        for name in cuboid_dict.keys():
            self._cuboid_obstacle_classes[class_name].add(name)
        self.add_cuboid_obstacles_to_visualization_world(cuboid_dict)

    @staticmethod
    def disable_cuboid_obstacles_in_visualization_world(name: str):
        """Disable cuboid glass_obstacles in the visualization world in Isaac Sim."""
        logger.info(f"Disabling cuboid glass_obstacles with name containing '{name}' in visualization world in Isaac Sim...")
        stage = prim_utils.get_current_stage()
        for prim in stage.TraverseAll():
            if "cuboid" in prim.GetName().lower() and name.lower() in prim.GetName().lower():
                prim_path = prim.GetPath().pathString
                logger.info(f"Disabling cuboid obstacle {prim_path} in Isaac Sim")
                cube_prim = XFormPrim(prim_path)
                cube_prim.set_visibilities(visibilities=[False])
                return
    @staticmethod
    def enable_cuboid_obstacles_in_visualization_world(name: str):
        """Enable cuboid glass_obstacles in the visualization world in Isaac Sim."""
        logger.info(f"Enabling cuboid glass_obstacles with name containing '{name}' in visualization world in Isaac Sim...")
        stage = prim_utils.get_current_stage()
        for prim in stage.TraverseAll():
            if "cuboid" in prim.GetName().lower() and name.lower() in prim.GetName().lower():
                prim_path = prim.GetPath().pathString
                logger.info(f"Enabling cuboid obstacle {prim_path} in Isaac Sim")
                cube_prim = XFormPrim(prim_path)
                cube_prim.set_visibilities(visibilities=[True])
                return

    def visualize_B_frame(self, X_B_Frame: HomogeneousMatrixType, frame_name: str = "B_frame"):
        """Visualize a coordinate frame B in Isaac Sim given its pose X_B_Frame in the robot base frame.
        """
        logger.debug(f"Visualizing frame {frame_name} in Isaac Sim.")
        # Extract position and orientation from homogeneous transformation matrix
        pos_W = (self.X_World_Base @ np.hstack((X_B_Frame[:3, 3], 1)))[:3]
        quat_W = (R.from_matrix(self.X_World_Base[:3, :3]) * R.from_matrix(X_B_Frame[:3, :3])).as_quat(scalar_first=True)
        self._visualize_pose(pos_W, quat_W, frame_name)
    def _visualize_pose(self, pos_W: np.ndarray, quat_W: np.ndarray, frame_name: str):
        prim_path = f"/World/{frame_name}"
        if not prim_utils.is_prim_path_valid(prim_path):
            frame_prim = prim_utils.create_prim(
                prim_path=prim_path,
                prim_type="Xform",
                position=pos_W,
                orientation=quat_W,
                scale=[0.1, 0.1, 0.1]
            )
            if frame_prim is None:
                logger.error(f"Failed to add coordinate frame {frame_name} to Isaac Sim")
            else:
                logger.info(f"Added coordinate frame {frame_name} to Isaac Sim at pos {pos_W}, quat {quat_W}")
        else:
            # Update existing prim
            xform_prim = XFormPrim(prim_path)
            xform_prim.set_local_poses(translations=np.array([pos_W]), orientations=np.array([quat_W]))

        if frame_name not in self._viz_frames.keys():
            self._draw_debug_lines_frame(pos_W, quat_W)
            self._viz_frames[frame_name] = (pos_W, quat_W)
        else:
            # Clear and redraw
            self._viz_frames[frame_name] = (pos_W, quat_W)
            self._draw.clear_lines()
            for frame, (pos_W, quat_W) in self._viz_frames.items():
                self._draw_debug_lines_frame(pos_W, quat_W)
    def _draw_debug_lines_frame(self, pos_W, quat_W):
        start_list = [pos_W]*3
        end_list = [
            pos_W + R.from_quat(quat_W, scalar_first=True).as_matrix()[:, 0]*0.1,
            pos_W + R.from_quat(quat_W, scalar_first=True).as_matrix()[:, 1]*0.1,
            pos_W + R.from_quat(quat_W, scalar_first=True).as_matrix()[:, 2]*0.1,
            ]
        color_list = [
            Gf.Vec4f(1.0, 0.0, 0.0, 0.8),  # X - Red
            Gf.Vec4f(0.0, 1.0, 0.0, 0.8),  # Y - Green
            Gf.Vec4f(0.0, 0.0, 1.0, 0.8),  # Z - Blue
        ]
        size_list = [2.0]*3
        self._draw.draw_lines(start_list, end_list, color_list, size_list)
