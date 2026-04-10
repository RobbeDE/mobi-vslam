import cv2
from utils import load_occupancy_grid, draw_occupancy_grid
import numpy as np
from constants import *
from utils import *

FILE_NAME = "occupancy_grids/edited_occupancy_grid15.npz"

# Mouse Callback for clicking on the grid
def on_mouse_click(event, x, y, flags, param: OccupancyGrid):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(points_G_to_Rw_2d(np.array([x, y]), param.cell_size, param.map_size_cells))



# Visualize the occupancy grid
occupancy_grid = load_occupancy_grid(FILE_NAME)
cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
# Attach the mouse callback to the window
cv2.setMouseCallback("Occupancy Grid", on_mouse_click, param=occupancy_grid)
draw_occupancy_grid("Occupancy Grid", occupancy_grid)


# Visualize the risk map
risk_map = occupancy_grid.get_risk_map(sigma=0.4)
vis = cv2.normalize(risk_map, None, 0, 255, cv2.NORM_MINMAX)
vis = vis.astype(np.uint8)
vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET) # Comment to get a grayscale visualization where the values represent the risk
cv2.imshow("Risk Map", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
