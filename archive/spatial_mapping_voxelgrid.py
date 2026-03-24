import torch
import numpy as np
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import WorldConfig, VoxelGrid
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# ==========================================
# PART 1: The Point Cloud -> SDF Converter
# ==========================================
class PointCloudToSDF:
    def __init__(self, workspace_limits, voxel_size=0.02, device="cuda"):
        """
        Converts raw points to CuRobo VoxelGrid.
        workspace_limits: [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        self.device = device
        self.voxel_size = voxel_size
        
        # Setup Grid Coordinates
        self.min_bound = torch.tensor(workspace_limits[:3], device=device)
        self.max_bound = torch.tensor(workspace_limits[3:], device=device)
        dims = ((self.max_bound - self.min_bound) / voxel_size).ceil().int()
        self.dims = dims.tolist()
        
        # Create meshgrid of voxel centers
        x = torch.linspace(self.min_bound[0], self.max_bound[0], self.dims[0], device=device)
        y = torch.linspace(self.min_bound[1], self.max_bound[1], self.dims[1], device=device)
        z = torch.linspace(self.min_bound[2], self.max_bound[2], self.dims[2], device=device)
        self.grid_coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        
        # Initialize SDF with Infinity
        self.current_sdf = torch.full((self.grid_coords.shape[0],), float('inf'), device=device)

    def reset(self):
        self.current_sdf.fill_(float('inf'))

    def update(self, points_np, obstacle_radius=0.03):
        """
        Updates the internal grid with new points (chunks).
        obstacle_radius: Radius to expand points by (meters).
        """
        if points_np is None or len(points_np) == 0: return
        
        points = torch.from_numpy(points_np).float().to(self.device)
        
        # Process in batches to prevent OOM
        batch_size = 50000 
        num_voxels = self.grid_coords.shape[0]
        
        for i in range(0, num_voxels, batch_size):
            end = min(i + batch_size, num_voxels)
            voxels = self.grid_coords[i:end]
            
            # Distance: Voxel Centers <-> Point Cloud
            # shape: (Batch, N_Points)
            dists = torch.cdist(voxels, points)
            min_dist, _ = dists.min(dim=1)
            
            # SDF: dist - radius. (Negative = Collision)
            new_sdf = min_dist - obstacle_radius
            
            # Keep the minimum distance (union of obstacles)
            self.current_sdf[i:end] = torch.minimum(self.current_sdf[i:end], new_sdf)

    def get_voxel_grid(self):
        # Reshape to (D, H, W)
        sdf_3d = self.current_sdf.view(self.dims[0], self.dims[1], self.dims[2])
        
        # Pose is the center of the grid volume
        center = (self.min_bound + self.max_bound) / 2.0
        pose = [center[0].item(), center[1].item(), center[2].item(), 1, 0, 0, 0]
        
        return VoxelGrid(
            name="zed_sdf",
            pose=pose,
            dims=self.dims,
            voxel_size=self.voxel_size,
            feature_tensor=sdf_3d
        )

# ==========================================
# PART 2: Setup Robot & Collision Checker
# ==========================================

# 1. Load Robot Config (Using Franka as example, replace with yours)
robot_name = "franka.yml"
robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_name))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args=TensorDeviceType())

# 2. Initialize Kinematics (Needed to calculate sphere positions from joint angles)
kinematics = CudaRobotModel(robot_cfg.kinematics)

# 3. Initialize the Voxel Collision Checker
# We must initialize it with a dummy config that matches our workspace size
workspace_limits = [-1.5, -1.5, -0.5, 1.5, 1.5, 2.0] # [x_min, y_min, z_min, ...]
voxel_size = 0.02

# Define the initial empty world
world_config = WorldConfig.from_dict({
    "voxel": {
        "base": {
            "dims": [3.0, 3.0, 2.5], # Must match workspace size roughly
            "pose": [0, 0, 0.75, 1, 0, 0, 0], 
            "voxel_size": voxel_size,
            "feature_dtype": torch.float32,
        }
    }
})

# THE STANDALONE CHECKER
collision_checker = WorldVoxelCollision(world_config)

# Initialize the SDF Calculator
sdf_calc = PointCloudToSDF(workspace_limits, voxel_size)


# ==========================================
# PART 3: Main Loop (ZED Data -> Check)
# ==========================================

def run_collision_check_loop(zed_points_chunk, current_joint_angles):
    """
    zed_points_chunk: numpy array (N, 3)
    current_joint_angles: numpy array or list of joint angles
    """
    
    # A. Update SDF --------------------------------------------------
    # You can call update() multiple times if you have chunks
    sdf_calc.reset() # Clear previous frame
    sdf_calc.update(zed_points_chunk)
    
    # Get the generic VoxelGrid object
    voxel_grid = sdf_calc.get_voxel_grid()
    
    # B. Feed to CuRobo Collision Checker ----------------------------
    collision_checker.update_voxel_data(voxel_grid)
    
    # C. Calculate Robot Sphere Positions ----------------------------
    # Convert joints to torch tensor (Batch=1, Joints)
    q_tensor = torch.tensor([current_joint_angles], dtype=torch.float32, device="cuda")
    
    # Forward Kinematics to get spheres
    kin_state = kinematics.get_state(q_tensor)
    robot_spheres = kin_state.link_spheres_tensor # Shape: (1, N_Spheres, 4) [x,y,z,radius]
    
    # D. Check Collision ---------------------------------------------
    # get_sphere_collision returns distance. 
    # Positive = Safe, Negative = In Collision
    dist = collision_checker.get_sphere_collision(robot_spheres)
    
    # Output results
    min_dist = torch.min(dist).item()
    
    print(f"Nearest Obstacle Distance: {min_dist:.4f} m")
    
    if min_dist < 0:
        print(">>> COLLISION DETECTED <<<")
        return True
    else:
        print("Path Clear")
        return False

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # 1. Fake ZED Data (A wall of points at x = 0.5)
    fake_points = np.random.uniform(-0.5, 0.5, (10000, 3))
    fake_points[:, 0] = 0.5 # Flatten X to make a wall
    
    # 2. Fake Robot Joints (Home position)
    joints = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78, 0.0, 0.0] # Franka 9 DoF (incl gripper)
    
    # 3. Run Check
    run_collision_check_loop(fake_points, joints)