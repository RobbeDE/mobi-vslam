import cv2
from utils import load_occupancy_grid, draw_occupancy_grid
import numpy as np
from loguru import logger
from constants import *
from utils import *

FILE_NAME = "occupancy_grids/edited_occupancy_grid15.npz"

# Mouse Callback for clicking on the grid
def on_mouse_click(event, x, y, flags, param: OccupancyGrid):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(points_G_to_Rw_2d(np.array([x, y]), param.cell_size, param.map_size_cells))



occupancy_grid = load_occupancy_grid(FILE_NAME)
cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
# Attach the mouse callback to the window
cv2.setMouseCallback("Occupancy Grid", on_mouse_click, param=occupancy_grid)
draw_occupancy_grid("Occupancy Grid", occupancy_grid)
cv2.waitKey(0)
