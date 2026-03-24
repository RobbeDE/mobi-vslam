# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

# General imports
import time
import numpy as np
# import torch
# import torch.utils.dlpack
import copy
import multiprocessing
from visual_slam.InteractiveScreen import InteractiveScreen
from visual_slam.shared_variables import shared_zed_data
from visual_slam.map_visualization import OccupancyGridViewer

# from viztracer import VizTracer

# Custom imports
from visual_slam.settings import GRID_DIMS, VOXEL_SIZE, MAX_VOXELS
from visual_slam.utils import filter_points, get_updated_points, points_to_grid_map, points_to_world, remove_floor_plane, timestamp
from visual_slam.shared_variables import shared_zed_data, occupancy_grid, occupancy_grid_lock
from visual_slam.zed_capture import ZedThread
from visual_slam.rerun_visualization import run_rerun_visualization
# from visual_slam.isaac_sim_visualization import run_isaac_sim_visualization

# # CuRobo imports
# from curobo.geom.types import WorldConfig, Mesh, VoxelGrid, Cuboid
# from curobo.types.base import TensorDeviceType
# from curobo.util_file import join_path, get_robot_configs_path
# from curobo.wrap.model.robot_world import RobotWorldConfig, RobotWorld
# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

def main_loop(screen: InteractiveScreen) -> None:

    # tracer = VizTracer()
    # tracer.start()

    def on_exit() -> None:
        # tracer.stop()
        # tracer.save()
        print("Exiting main loop...")
        zed_thread.stop()
        zed_thread.join()
        visualization_shutdown_event.set()
        viz_process.join()
        print("Cleaned up processes. Exiting.")

    screen.set_exit_handler(on_exit)


    # curobo_fn = RobotWorld(config)

    # q_sph = torch.randn((100000, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype) # Test collision query
    # q_sph[...,3] = 0.2

    # # Voxel Pool
    # voxel_pool = VoxelPool(voxel_size=VOXEL_SIZE, max_size=MAX_VOXELS)

    print("Starting capture...")

    # 1. Setup Communication Queue
    # maxsize=1 is crucial: if Viz is slow, we drop frames rather than filling RAM
    viz_queue = multiprocessing.Queue(maxsize=1)

    # 2. Start Visualization Process
    visualization_shutdown_event = multiprocessing.Event()
    viz_process = multiprocessing.Process(
        target=run_rerun_visualization, 
        args=(viz_queue, VOXEL_SIZE, visualization_shutdown_event)
    )
    viz_process.start()

    # 3. Start map visualization thread
    grid_viewer = OccupancyGridViewer(occupancy_grid, occupancy_grid_lock)
    grid_viewer.start()

    print("[Main Process] Starting Capture & Logic...")

    # Initialize ZED Camera
    zed_thread = ZedThread(daemon=True, verbose=True)
    zed_thread.start()

    while True:
        pass

        shared_zed_data.wait_for_all_updates()

        timestamp("[Main Process] New Data Available.")
        
        # Make deep copy of all shared data
        pose = copy.deepcopy(shared_zed_data.get_live_pose())
        if pose is None:
            continue
        dynamic_map = points_to_world(copy.deepcopy(shared_zed_data.get_live_point_cloud()), pose)
        static_map = copy.deepcopy(shared_zed_data.get_spatial_map())
        updated_static_map = get_updated_points(static_map)
        floor_equation = copy.deepcopy(shared_zed_data.get_floor_plane())
        image = copy.deepcopy(shared_zed_data.get_rgb_image())[:,:, :3] # Drop alpha channel if exists
        image = image[:, :, ::-1]
        global_map = np.concatenate([dynamic_map, updated_static_map], axis=0)

        print(f"Dynamic map size: {dynamic_map.shape[0]}")
        print(f"Updated static map size: {updated_static_map.shape[0]}")
        print(f"Global map size: {global_map.shape[0]}")

        non_points_floor = remove_floor_plane(global_map, floor_height=-1.45, threshold=0.1)

        viz_queue.put({"type": "voxels", "data": global_map})
        viz_queue.put({"type": "camera", "data": pose})

        with occupancy_grid_lock:
            points_to_grid_map(occupancy_grid, non_points_floor[:,:2], VOXEL_SIZE)
        arr = (occupancy_grid.astype(np.uint8)) * 255   # convert to grayscale

        # Convert to 3-channel (pygame needs RGB)
        rgb = np.stack((arr, arr, arr), axis=-1)
        screen.imshow(rgb)


        # # If we don't have data yet, skip this iteration
        # if pose is None or global_map is None:
        #     continue
        # timestamp("[Main Process] New data retrieved")

        # # Visualization
        # viz_queue.put({"type": "camera", "data": pose})
        # viz_queue.put({"type": "voxels", "data": global_map, "pose": pose})
        # timestamp("[Main Process] Sent data to visualization.")

        # # Update CuRobo Voxel Pool
        # active_cuboids = voxel_pool.update(global_map)
        # print(f"Active Voxels: {len(active_cuboids)}")
        # timestamp("[Main Process] Updated CuRobo Voxel Pool.")

        # # 3. Send to CuRobo
        # # Because we used the 'cache' config, CuRobo reuses GPU memory
        # global_map = filter_points(global_map)



        # # 4. Update World
        # # We reuse the voxel_grid object, but update the wrapper config
        # world_cfg = WorldConfig(mesh=[], cuboid=active_cuboids, sphere=[], voxel=[])
        # curobo_fn.update_world(world_cfg)

        # # 4. Check Collision
        # d = curobo_fn.get_collision_distance(q_sph)
        # timestamp("[Main Process] Collision query done.")