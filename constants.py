MAP_SIZE_METERS = 50.0      # The map will cover 50x50 meters
CELL_SIZE = 0.1             # Cells are 10cm x 10cm
MAP_SIZE_CELLS = int(MAP_SIZE_METERS / CELL_SIZE)  # 500
CAMERA_HEIGHT = 0.45        # The height of the ZED camera in meters
GROUND_MARGIN = 0.25        # Margin around the camera height to consider as "ground" (in meters)
MAX_OBSTACLE_HEIGHT = 1.5   # Maximum height to consider for obstacles (in meters)