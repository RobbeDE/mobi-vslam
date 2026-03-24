import sys
import time
import pyzed.sl as sl
import argparse
import numpy as np


# keep this at module scope so it persists across frames
prev_vertices_map = {}  # key -> numpy array of previous-frame vertices

def compute_avg_shifts_per_chunk(mesh):
    """
    Returns {key: avg_shift} and updates prev_vertices_map for the next frame.
    Key tries chunk.id if available, otherwise uses id(chunk).
    """
    shifts = {}
    for chunk in mesh.chunks:
        key = getattr(chunk, "id", None) or id(chunk)

        # convert current vertices to numpy array (N,3)
        curr = np.array(chunk.vertices, dtype=float)

        prev = prev_vertices_map.get(key)
        if prev is None:
            # first time we see this chunk -> no displacement info yet
            avg_shift = 0.0
        else:
            # if vertex counts changed, compare up to the min length
            n = min(prev.shape[0], curr.shape[0])
            if n == 0:
                avg_shift = 0.0
            else:
                print(curr[:n])
                print(prev[:n])

                diffs = np.linalg.norm(curr[:n] - prev[:n], axis=1)
                avg_shift = float(diffs.mean())

        shifts[key] = avg_shift

        # store a copy for the next frame
        prev_vertices_map[key] = curr.copy()

    # optionally, remove entries for chunks that no longer exist
    existing_keys = {getattr(c, "id", None) or id(c) for c in mesh.chunks}
    for k in list(prev_vertices_map.keys()):
        if k not in existing_keys:
            prev_vertices_map.pop(k, None)

    return shifts


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
                                                use_chunk_only = True,
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

        # Request an update of the spatial map every 30 frames
        if timer % 30 == 0 :
            zed.request_spatial_map_async()

        # Retrieve spatial_map when ready
        if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and timer > 0:
            zed.retrieve_spatial_map_async(mesh)

            # Example usage in your frame loop
            shifts = compute_avg_shifts_per_chunk(mesh)
            # for key, avg in shifts.items():
                # print(f"Chunk {key}: avg shift = {avg:.6f}")

zed.disable_spatial_mapping()
zed.disable_positional_tracking()
zed.close()