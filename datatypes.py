import numpy as np
from scipy.ndimage import distance_transform_edt

class OccupancyGrid:
    def __init__(self, map_size: float, cell_size: float, grid: np.ndarray):
        self.map_size = map_size
        self.cell_size = cell_size
        self.grid = grid  # (0,0,0) =Occupied, (128,128,128) =Unknown, (255,255,255) =Free

    @property
    def map_size_cells(self) -> int:
        return int(self.map_size / self.cell_size)
    
    def get_risk_map(self, sigma: float) -> np.ndarray:
        """Computes a risk map using distance transform and Gaussian decay.
        Goes from 0 to 1 (with 1 being occupied or unknown) with a smooth gradient in between."""
        
        free_spaces = np.all(self.grid == (255, 255, 255), axis=2)

        distances = np.array(distance_transform_edt(free_spaces)) * self.cell_size
        risk_map = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        return risk_map