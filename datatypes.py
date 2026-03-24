import numpy as np

class OccupancyGrid:
    def __init__(self, map_size: float, cell_size: float, grid: np.ndarray):
        self.map_size = map_size
        self.cell_size = cell_size
        self.grid = grid  # 0=Occupied, 128=Unknown, 255=Free

    @property
    def map_size_cells(self) -> int:
        return int(self.map_size / self.cell_size)

    