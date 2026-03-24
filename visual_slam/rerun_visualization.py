import multiprocessing
import numpy as np
import numpy.typing as npt


def run_rerun_visualization(data_queue: multiprocessing.Queue, voxel_size: float, shutdown_event: multiprocessing.Event) -> None:
    """
    Runs Rerun in a separate process. 
    Expects data_queue to receive dicts: 
    {'type': 'voxels', 'data': np.array} OR {'type': 'camera', 'data': np.array}
    """
    print("Starting Rerun Visualization...")

    import rerun as rr

    # Initialize Rerun
    rr.init("Curobo_Voxel_Viz", spawn=True)
    
    # 1. Setup World Coordinate System (Z-Up)
    rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # 2. Setup a "dummy" camera frustum so we can actually SEE the camera object
    # We log this once (static=True) or every frame if focal length changes.
    # Without this, the camera is just an invisible coordinate point.
    rr.log(
        "World/Camera/Pinhole",
        rr.Pinhole(
            resolution=[1280, 720], # Arbitrary resolution for visualization
            focal_length=700,       # Arbitrary FOV for visualization
        ),
        static=True
    )

    # --- HELPER FUNCTIONS ---

    def update_voxel_viz(points_np: np.ndarray) -> None:
        if len(points_np) == 0:
            rr.log("World/PointBytes", rr.Clear(recursive=False))
            return
        
        points_xyz = points_np[:, :3]
        indices = np.rint(points_xyz / voxel_size)
        snapped_pos = indices * voxel_size
        unique_pos = np.unique(snapped_pos, axis=0)

        rr.log(
            "World/PointBytes",
            rr.Boxes3D(
                centers=unique_pos,
                sizes=[voxel_size, voxel_size, voxel_size],
                colors=[200, 200, 200],
                fill_mode="solid"
            )
        )

    def update_camera_viz(pose_matrix: np.ndarray) -> None:
        """
        Updates the camera position in the world.
        pose_matrix: 4x4 Homogeneous Matrix (numpy)
        """
        # Extract Translation (first 3 rows, 4th column)
        translation = pose_matrix[:3, 3]
        
        # Extract Rotation Matrix (3x3 top-left)
        rotation_mat = pose_matrix[:3, :3]

        R_x_90 = np.array([
        [1, 0,  0],
        [0, 0, 1],
        [0, -1,  0],
        ], dtype=float)

        rotation_mat = rotation_mat @ R_x_90

        # Log the transform. 
        # This moves "World/Camera" (and its child "Pinhole") to the new location.
        rr.log(
            "World/Camera",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_mat
            )
        )

    print("Visualization setup Ready. Waiting for data...")

    # --- RENDER LOOP ---
    while not shutdown_event.is_set():
        if not data_queue.empty():
            try:
                # Expecting a dict now: { "type": "...", "data": ... }
                msg = data_queue.get_nowait()
                
                if isinstance(msg, dict):
                    if msg.get("type") == "voxels":
                        update_voxel_viz(msg["data"])
                    elif msg.get("type") == "camera":
                        update_camera_viz(msg["data"])
                else:
                    # Legacy support if you accidentally send just arrays
                    # Assuming raw arrays are voxels
                    update_voxel_viz(msg)

            except Exception as e:
                pass
        
        import time
        time.sleep(0.01)
    
    print("Shutting down Rerun visualization...")
    pass