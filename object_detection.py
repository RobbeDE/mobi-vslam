import pyzed.sl as sl
import cv2
import numpy as np
import math

def create_2d_map(cam_pose, obj_array, map_size=600, scale=40):
    """
    Creates a 2D top-down map of the camera and detected objects.
    - map_size: Size of the square window in pixels.
    - scale: Pixels per meter (e.g., 40 means 40 pixels = 1 meter).
             Map covers roughly (map_size/scale) meters in total.
    """
    # Create a dark gray background
    grid_map = np.ones((map_size, map_size, 3), dtype=np.uint8) * 30

    # The center of the window acts as the World Origin (0,0,0)
    cx, cy = map_size // 2, map_size // 2

    # 1. Draw the Grid (Every 1 meter)
    for i in range(0, map_size, scale):
        cv2.line(grid_map, (i, 0), (i, map_size), (60, 60, 60), 1)
        cv2.line(grid_map, (0, i), (map_size, i), (60, 60, 60), 1)

    # 2. Draw World Coordinate Frame (Axes)
    # X-axis (Red) points right
    cv2.arrowedLine(grid_map, (cx, cy), (cx + scale, cy), (0, 0, 255), 2, tipLength=0.2)
    # Z-axis (Blue) points forward (represented as UP on our screen)
    cv2.arrowedLine(grid_map, (cx, cy), (cx, cy - scale), (255, 0, 0), 2, tipLength=0.2)
    cv2.putText(grid_map, "World Origin", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 3. Draw Camera Position & Orientation
    cam_trans = cam_pose.get_translation().get()
    cam_x, cam_z = cam_trans[0], cam_trans[2]

    # Map real-world X, Z to pixel coordinates
    cam_px = int(cx + cam_x * scale)
    cam_py = int(cy - cam_z * scale)

    # Get camera rotation to draw the direction it's facing
    cam_rot = cam_pose.get_rotation_matrix().r
    forward_v = cam_rot[:, 2] # The Z-axis of the camera in world frame

    # Draw camera as a Cyan circle and its forward vector as a line
    cv2.circle(grid_map, (cam_px, cam_py), 6, (255, 255, 0), -1)
    cv2.line(grid_map, (cam_px, cam_py), 
             (int(cam_px + forward_v[0] * 20), int(cam_py - forward_v[2] * 20)), 
             (255, 255, 0), 2)
    cv2.putText(grid_map, "Camera", (cam_px + 10, cam_py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # 4. Draw Detected Objects
    for obj in obj_array:
        pos = obj.position
        x, z = pos[0], pos[2] # Extract X and Z for top-down 2D mapping
        
        if not math.isnan(x):
            obj_px = int(cx + x * scale)
            obj_py = int(cy - z * scale)
            
            # Draw object as a bright Green circle
            cv2.circle(grid_map, (obj_px, obj_py), 5, (0, 255, 0), -1)
            
            # Label the object with its ID
            label = f"{str(obj.label).split('.')[-1]}[{obj.id}]"
            cv2.putText(grid_map, label, (obj_px + 8, obj_py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return grid_map

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_serial_number("31733653")
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER # Real-world tracking in meters
    init_params.sdk_verbose = 1

    # runtime parameters
    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {err}. Exit program.")
        exit()

    # Enable Positional Tracking
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False 
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_gravity_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Enable object detection : {err}. Exit program.")
        zed.close()
        exit()

    # Variables for runtime
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    image_matrix = sl.Mat()
    cam_pose = sl.Pose()

    print("Running... Press 'q' to exit.")

    # Loop continuously until 'q' is pressed
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            
            # Retrieve camera feed, objects, and current world pose
            zed.retrieve_image(image_matrix, sl.VIEW.LEFT)
            zed.retrieve_objects(objects, obj_runtime_param)
            zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)

            image_ocv = image_matrix.get_data()

            if objects.is_new:
                obj_array = objects.object_list
                
                # Terminal Print & 2D Box Drawing
                for obj in obj_array:
                    bbox_2d = obj.bounding_box_2d
                    pos = obj.position 
                    x, y, z = pos[0], pos[1], pos[2]
                    
                    label_str = str(obj.label).split('.')[-1]

                    print(cam_pose.get_translation())

                    if not math.isnan(x):
                        print(f"[Frame] ID: {obj.id} | Label: {label_str} | Pose (m): X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

                    if len(bbox_2d) == 4:
                        top_left = (int(bbox_2d[0][0]), int(bbox_2d[0][1]))
                        bottom_right = (int(bbox_2d[2][0]), int(bbox_2d[2][1]))

                        cv2.rectangle(image_ocv, top_left, bottom_right, (0, 255, 0), 2)
                        label_text = f"{label_str} {int(obj.confidence)}% [ID: {obj.id}]"

                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(image_ocv, (top_left[0], top_left[1] - text_h - 10), 
                                      (top_left[0] + text_w, top_left[1]), (0, 255, 0), -1)
                        cv2.putText(image_ocv, label_text, (top_left[0], top_left[1] - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Generate and retrieve the 2D Top-Down grid map
            top_down_map = create_2d_map(cam_pose, objects.object_list)

            # Display Both Windows
            cv2.imshow("ZED - Live Camera feed", image_ocv)
            cv2.imshow("ZED - 2D Top-Down Map", top_down_map)

            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                break

    # Clean up
    cv2.destroyAllWindows()
    zed.disable_object_detection()
    zed.close()

if __name__ == "__main__":
    main()