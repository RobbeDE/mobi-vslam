import cv2
from utils import load_occupancy_grid, save_occupancy_grid

ORIGINAL_FILE_NAME = "occupancy_grids/occupancy_grid15.npz"  # The .npz file containing the saved spatial map data
EDITED_FILE_NAME = "occupancy_grids/edited_occupancy_grid15.npz"  # The occupancy grid file to save after editing
BRUSH_SIZE = 1              # Set to 2 or 3 if you want a wider brush when painting


def edit_occupancy_grid(file_name: str):

    occupancy_grid = load_occupancy_grid(file_name)

    # --- EDITOR & SAVING LOGIC ---
    cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Occupancy Grid", 800, 800)

    # Track current mouse position continuously
    mouse_x, mouse_y = -1, -1

    # Variables to track drawing state & draw smooth lines over fast mouse movements
    last_painted_pos = None
    last_painted_key = None
    key_timeout = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        # Only track the position. No clicks are processed here.
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y

    cv2.setMouseCallback("Occupancy Grid", mouse_callback)

    print("\n--- Map Loaded Successfully ---")
    print("Editor Controls:")
    print(" Hover your mouse and HOLD one of the following keys to paint:")
    print("  [O] - Paint as Occupied (Black)")
    print("  [F] - Paint as Free (White)")
    print("[U] - Paint as Unknown (Gray)")
    print("---------------------------------------------------------------")
    print("  [S]          - Save map to disk")
    print("  [Q] or [ESC] - Exit program\n")

    while True:
        # Create a fresh display image so markers/text don't bleed into grid_data
        display_img = occupancy_grid.grid.copy()

        # Draw a small red crosshair where the cursor is currently hovering
        if 0 <= mouse_x < occupancy_grid.map_size_cells and 0 <= mouse_y < occupancy_grid.map_size_cells:
            cv2.drawMarker(display_img, (mouse_x, mouse_y), (0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1)

        cv2.putText(display_img, "Hold:[O]ccupied | [F]ree | [U]nknown    Action: [S]ave |[Q]uit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Occupancy Grid", display_img)
        
        # Read keyboard input every 10ms
        key = cv2.waitKey(10) & 0xFF
        
        # Determine active drawing color based on held key
        active_color = None
        if key == ord('o'):
            active_color = (0, 0, 0)         # Occupied
        elif key == ord('f'):
            active_color = (255, 255, 255)   # Free
        elif key == ord('u'):
            active_color = (128, 128, 128)   # Unknown

        # If a painting key is currently pressed...
        if active_color is not None:
            if 0 <= mouse_x < occupancy_grid.map_size_cells and 0 <= mouse_y < occupancy_grid.map_size_cells:
                if last_painted_pos is not None and last_painted_key == key:
                    # Draw a line from the last pos to current pos to prevent gaps
                    cv2.line(occupancy_grid.grid, last_painted_pos, (mouse_x, mouse_y), active_color, BRUSH_SIZE)
                else:
                    # If this is the start of a stroke, just color the single pixel area
                    cv2.line(occupancy_grid.grid, (mouse_x, mouse_y), (mouse_x, mouse_y), active_color, BRUSH_SIZE)
                
                # Store the state for the next frame
                last_painted_pos = (mouse_x, mouse_y)
                last_painted_key = key
                key_timeout = 0
        else:
            # If no key is pressed, allow a short timeout before breaking the drawing sequence.
            # This masks the OS key-repeat stutter that happens when you first hold down a key.
            key_timeout += 1
            if key_timeout > 15:  
                last_painted_pos = None
                last_painted_key = None
        
        # Standard actions
        if key == 27 or key == ord('q'):  
            break
        elif key == ord('s'):            
            save_occupancy_grid(occupancy_grid, EDITED_FILE_NAME)
            
            print("Occupancy grid saved to", EDITED_FILE_NAME)
            
            cv2.putText(display_img, "SAVED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Occupancy Grid", display_img)
            cv2.waitKey(500) 

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading spatial map...")

    edit_occupancy_grid(ORIGINAL_FILE_NAME)