from utils import load_occupancy_grid, visualize_occupancy_grid

FILE_NAME = "occupancy_grid5.npz"

occupancy_grid = load_occupancy_grid(FILE_NAME)
visualize_occupancy_grid(occupancy_grid)
