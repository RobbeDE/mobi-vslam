import multiprocessing
import os
import numpy as np
from curobo.geom.types import WorldConfig

def run_isaac_sim_visualization(data_queue: multiprocessing.Queue, voxel_size: float, shutdown_event: multiprocessing.Event) -> None:
    """
    This function runs in a completely separate OS process.
    All Isaac Sim imports and initialization must happen HERE.
    """
    print("Starting Isaac Sim...")

    import os
    os.environ["OMNI_LOG_LEVEL"] = "ERROR"

    # --- ISAAC SIM IMPORTS (Must be inside the process) ---
    from isaacsim.simulation_app import SimulationApp

    # Initialize the App
    simulation_app = SimulationApp( {"headless": False, "width": 1920, "height": 1080} )

    # Omni imports must happen AFTER SimulationApp starts
    from omni.isaac.core import World
    from omni.isaac.core.utils import stage as stage_utils
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.prims import XFormPrim
    from pxr import UsdGeom, Gf, Vt, Usd  # <--- NEW IMPORTS FOR USD

    # --- HELPER CLASSES AND FUNCTIONS (Moved inside to access scoped imports) ---
    class VoxelInstancer:
        def __init__(self, prim_path: str, voxel_size: float) -> None:
            self.stage = stage_utils.get_current_stage()
            self.voxel_size = voxel_size
            
            # 1. Create the PointInstancer
            self.instancer = UsdGeom.PointInstancer.Define(self.stage, prim_path)
            
            # 2. Create the Prototype (One single Cube mesh)
            proto_path = f"{prim_path}/voxel_proto"
            self.cube = UsdGeom.Cube.Define(self.stage, proto_path)
            
            # Set size (Isaac Cube default is 2.0 (-1 to 1), we scale it)
            # We set the base size to 1.0, then scale it by voxel_size
            self.cube.GetSizeAttr().Set(1.0)
            
            # Apply scaling to the prototype to match voxel size
            # This ensures the visual cubes touch exactly at grid lines
            xform = UsdGeom.Xformable(self.cube)
            xform.ClearXformOpOrder()
            xform.AddScaleOp().Set(Gf.Vec3f(voxel_size, voxel_size, voxel_size))

            # 3. Link Prototype to Instancer
            self.instancer.GetPrototypesRel().AddTarget(proto_path)

        def update(self, points_np: np.ndarray) -> None: # type: ignore
            """
            Takes raw point cloud (N, 3), snaps to grid, and renders.
            """
            if len(points_np) == 0:
                self.instancer.GetPositionsAttr().Set(Vt.Vec3fArray([]))
                self.instancer.GetProtoIndicesAttr().Set(Vt.IntArray([]))
                return
            
            # 1. Ensure we only use X,Y,Z (Drop the 4th column if it exists)
            # ZED cameras often return (N, 4) where 4 is color/confidence.
            # If we don't slice this, the memory alignment breaks in USD.
            points_xyz = points_np[:, :3]

            # 2. Perform Grid Snapping
            # We do the math here. Note: this might promote types to float64 automatically.
            indices = np.rint(points_xyz / self.voxel_size)
            snapped_pos = indices * self.voxel_size
            
            # 3. Remove duplicates
            unique_pos = np.unique(snapped_pos, axis=0)
            
            # 4. EXPLICIT CAST TO FLOAT32
            # USD (Vt.Vec3fArray) assumes the memory buffer is 32-bit floats.
            # If unique_pos is float64, you get the 10^32 coordinate bug.
            unique_pos = unique_pos.astype(np.float32)
            
            # --- Update USD Attributes ---
            num_points = len(unique_pos)
            
            # Convert to USD types (Zero-copy where possible)
            usd_positions = Vt.Vec3fArray.FromNumpy(unique_pos)
            # All points use prototype index 0 (the cube)
            proto_indices = Vt.IntArray(num_points, 0)
            
            self.instancer.GetPositionsAttr().Set(usd_positions)
            self.instancer.GetProtoIndicesAttr().Set(proto_indices)


    def convert_cuboid_world_to_isaacsim(cuboid_world: WorldConfig) -> None:
        if (type(cuboid_world) != WorldConfig) or (cuboid_world.cuboid is None):
            print("Invalid cuboid world provided for conversion to Isaac Sim.")
            return

        for i, cuboid in enumerate(cuboid_world.cuboid):
            if cuboid.pose is None:
                print(f"Skipping cuboid {i} due to missing pose or dims")
                continue

            pos = cuboid.pose[:3]
            quat = cuboid.pose[3:]
            size = [dim/2. for dim in cuboid.dims]

            if i % 1000 == 0:
                print(f"cuboid {i}: pos: {pos}, quat: {quat}, size: {size}")

            prim_path = f"/World/cuboid_{i}_{cuboid.name}" if cuboid.name else f"/World/cuboid_{i}"
            if not prim_utils.is_prim_path_valid(prim_path):
                cube_prim = prim_utils.create_prim(
                    prim_path=prim_path,
                    prim_type="Cube",
                    position=pos,
                    orientation=quat,
                    scale=size
                )
                if cube_prim is None:
                    print(f"Failed to add cuboid {i} to Isaac Sim")
                    continue
            else:
                # Update existing prim
                xform_prim = XFormPrim(prim_path)
                xform_prim.set_local_poses(translations=np.array([pos]), orientations=np.array([quat]))  # wxyz quaternion
                xform_prim.set_local_scales(np.array([size]))
                print(f"Updated cuboid {prim_path} in Isaac Sim to pos {pos}, quat {quat}, size {size}")

    # --- SETUP SCENE ---
    isaac_world = World(stage_units_in_meters=1.0)
    isaac_world.scene.add_default_ground_plane() # Ground plane for visualization
    voxel_viz = VoxelInstancer("/World/PointBytes", voxel_size) # Initialize the Visualizer

    print("Visualization setup Ready. Waiting for data...")

    # --- RENDER LOOP ---
    while not shutdown_event.is_set():
        # Check if new data is available without blocking
        if not data_queue.empty():
            try:
                # get_nowait coupled with the queue size of 1 ensures 
                # we just processed the absolute latest frame
                points = data_queue.get_nowait()
                voxel_viz.update(points)

            except:
                pass
        
        # Step Physics/Render
        simulation_app.update()
    
    print("Shutting down Isaac Sim visualization...")
    simulation_app.close()