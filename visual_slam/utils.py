# from curobo.geom.types import Cuboid
import numpy as np
import numpy.typing as npt
from visual_slam.shared_variables import start_time
import time

# class VoxelPool:
#     def __init__(self, voxel_size: float, max_size: int) -> None:
#         self.voxel_size = voxel_size
#         self.max_size = max_size
        
#         # 1. Pre-allocate a pool of Cuboid objects (Hidden initially)
#         # We reuse these objects instead of creating new ones
#         self.inactive_pool = []
#         for i in range(max_size):
#             c = Cuboid(
#                 name=f"vox_{i}", 
#                 dims=[voxel_size, voxel_size, voxel_size], 
#                 pose=[0, 0, 0, 1, 0, 0, 0], # Hidden far away
#                 color=[0.0, 0.0, 1.0, 1.0]
#             )
#             self.inactive_pool.append(c)
            
#         # 2. Tracking Dictionary: Maps (x,y,z) integers -> Cuboid Object
#         self.active_voxels = {} 

#     def update(self, points_np: np.ndarray) -> list[Cuboid]: # type: ignore
#         """
#         Args:
#             points_np: (N, 3) raw points from ZED
#         Returns:
#             List of Cuboid objects to send to CuRobo
#         """
#         # 1. Quantize to Integer Indices (Key generation)
#         # We use integers for dictionary keys because they are exact and fast
#         if points_np.shape[0] == 0:
#             return []

#         # Snap to grid and convert to int for hashing
#         indices = np.rint(points_np / self.voxel_size).astype(int)
        
#         # Remove duplicates in the input data immediately
#         unique_indices = np.unique(indices, axis=0)
        
#         # Convert to set of tuples for fast set operations
#         # current_frame_keys = {(1,2,3), (5,5,5), ...}
#         current_frame_keys = set(map(tuple, unique_indices))
        
#         # Get keys from previous frame
#         prev_keys = set(self.active_voxels.keys())
        
#         # 2. Calculate Set Differences (The Optimization)
#         # Voxels that exist now but didn't before
#         to_add = current_frame_keys - prev_keys

#         # 3. Process Additions
#         for key in to_add:
#             if not self.inactive_pool:
#                 # Pool exhausted: In a real app, log a warning or expand pool
#                 continue 
                
#             cuboid = self.inactive_pool.pop()
            
#             # Update the recycled object's pose
#             # Convert integer key back to float world position
#             x, y, z = key
#             pos_x = x * self.voxel_size
#             pos_y = y * self.voxel_size
#             pos_z = z * self.voxel_size
            
#             # Update mutable pose attribute [x, y, z, qw, qx, qy, qz]
#             # Note: We assume Identity rotation for voxels
#             cuboid.pose = [pos_x, pos_y, pos_z, 1.0, 0.0, 0.0, 0.0]
            
#             # Register it
#             self.active_voxels[key] = cuboid
            
#         # 5. Return the list of currently active objects
#         return list(self.active_voxels.values())
    

def points_to_world(points: npt.NDArray, pose: npt.NDArray) -> npt.NDArray:
    N = points.shape[0]
    homogenous_points = np.hstack((points, np.ones((N, 1), dtype=points.dtype))) # (N, 4)
    world_points = (pose @ homogenous_points.T).T # (N, 4)
    return world_points[:, :3] # (N, 3)

def get_updated_points(spatial_map: list[tuple[npt.NDArray, int]]) -> npt.NDArray:
    updated_points = []
    for vertices, has_been_updated in spatial_map:
        if has_been_updated:
            updated_points.append(vertices)
    if updated_points:
        return np.vstack(updated_points)
    else:
        return np.empty((0, 3), dtype=np.float32)

def filter_points(points: npt.NDArray) -> npt.NDArray:
    # Simple filter: Remove points that are too close or too far
    dists = np.linalg.norm(points, axis=1)
    mask = (dists > 0.1) & (dists < 10.0) # Keep points between 0.1m and 10m
    return points[mask]

def remove_floor_plane(points: npt.NDArray, floor_height: float, threshold: float) -> npt.NDArray:
    # Remove points close to the floor plane (z = floor_height)
    z_coords = points[:, 2]
    mask = np.abs(z_coords - floor_height) > threshold
    non_floor_points = points[mask]

    return non_floor_points

def points_to_grid_map(grid_map: npt.NDArray, points: npt.NDArray, voxel_size: float) -> npt.NDArray:
    # Convert points → integer grid coordinates
    indices = np.rint(points / voxel_size).astype(int)

    # Center the grid around (0,0)
    grid_dims = grid_map.shape
    center_offsets = np.array(grid_dims) // 2
    indices = indices + center_offsets  # shift

    # ------------------------------------------------------------------
    # FILTER OUT POINTS THAT FALL OUTSIDE THE GRID
    # ------------------------------------------------------------------
    valid_mask = np.all(
        (indices >= 0) &
        (indices < np.array(grid_dims)),
        axis=1
    )
    valid_indices = indices[valid_mask]

    # ------------------------------------------------------------------
    # UPDATE GRID: set occupied cells to True
    # valid_indices is shape (N, 2)
    # ------------------------------------------------------------------
    grid_map[:, :] = True
    if valid_indices.size > 0:
        grid_map[valid_indices[:, 0], valid_indices[:, 1]] = False

    


def timestamp(msg: str) -> None:
    timestamp_ms = (time.perf_counter() - start_time) * 1000.0
    print(f"[{timestamp_ms:.3f} ms] {msg}")