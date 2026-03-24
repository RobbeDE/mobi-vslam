from threading import Lock, Event
import numpy as np
from numpy.typing import NDArray
import time

start_time = time.perf_counter()

class SharedZedData:
    _lock: Lock = Lock()
    _resolution: tuple[int, int]
    _map_max_size: int

    spatial_map: list[tuple[NDArray, int]]
    rgb_image: np.ndarray
    depth_image: np.ndarray
    live_point_cloud: np.ndarray
    live_pose: np.ndarray
    floor_plane: np.ndarray

    _spatial_map_lock: Lock
    _rgb_image_lock: Lock
    _depth_image_lock: Lock
    _live_point_cloud_lock: Lock
    _live_pose_lock: Lock
    _floor_plane_lock: Lock

    _spatial_map_event: Event
    _rgb_image_event: Event
    _depth_image_event: Event
    _live_point_cloud_event: Event
    _live_pose_event: Event
    _floor_plane_event: Event

    
    def __init__(self, resolution: tuple[int, int], map_max_size: int) -> None:
        self._resolution = resolution
        self._map_max_size = map_max_size

        self.spatial_map = None
        self.rgb_image = None
        self.depth_image = None
        self.live_point_cloud = None
        self.live_pose = None
        self.floor_plane = None

        self._spatial_map_lock = Lock()
        self._rgb_image_lock = Lock()
        self._depth_image_lock = Lock()
        self._live_point_cloud_lock = Lock()
        self._live_pose_lock = Lock()
        self._floor_plane_lock = Lock()

        self._spatial_map_event = Event()
        self._rgb_image_event = Event()
        self._depth_image_event = Event()
        self._live_point_cloud_event = Event()
        self._live_pose_event = Event()
        self._floor_plane_event = Event()

        self._full_update_event = Event()

    def __new__(cls, *args, **kwargs) -> 'SharedZedData':
        with cls._lock:
            if not hasattr(cls, 'instance'):
                cls.instance = super(SharedZedData, cls).__new__(cls)
        return cls.instance
    
    def get_spatial_map(self) -> list[tuple[NDArray, int]]:
        with self._spatial_map_lock:
            return self.spatial_map
        
    def set_spatial_map(self, data: list[tuple[NDArray, int]]) -> None:
        with self._spatial_map_lock:
            self.spatial_map = data
        self._spatial_map_event.set()

    def wait_for_spatial_map_update(self, timeout: float = None) -> bool:
        self._spatial_map_event.wait(timeout)
        self._spatial_map_event.clear()
    
    def get_rgb_image(self) -> NDArray:
        with self._rgb_image_lock:
            return self.rgb_image
    
    def set_rgb_image(self, data: NDArray) -> None:
        with self._rgb_image_lock:
            self.rgb_image = data

    def wait_for_rgb_image_update(self, timeout: float = None) -> bool:
        self._rgb_image_event.wait(timeout)
        self._rgb_image_event.clear()

    def get_depth_image(self) -> NDArray:
        with self._depth_image_lock:
            return self.depth_image
        
    def set_depth_image(self, data: NDArray) -> None:
        with self._depth_image_lock:
            self.depth_image = data

    def wait_for_depth_image_update(self, timeout: float = None) -> bool:
        self._depth_image_event.wait(timeout)
        self._depth_image_event.clear()

    def get_live_point_cloud(self) -> NDArray:
        with self._live_point_cloud_lock:
            return self.live_point_cloud
        
    def set_live_point_cloud(self, data: NDArray) -> None:
        with self._live_point_cloud_lock:
            self.live_point_cloud = data

    def wait_for_live_point_cloud_update(self, timeout: float = None) -> bool:
        self._live_point_cloud_event.wait(timeout)
        self._live_point_cloud_event.clear()

    def get_live_pose(self) -> NDArray:
        with self._live_pose_lock:
            return self.live_pose
        
    def set_live_pose(self, data: NDArray) -> None:
        with self._live_pose_lock:
            self.live_pose = data

    def wait_for_live_pose_update(self, timeout: float = None) -> bool:
        self._live_pose_event.wait(timeout)
        self._live_pose_event.clear()

    def get_floor_plane(self) -> NDArray:
        with self._floor_plane_lock:
            return self.floor_plane
        
    def set_floor_plane(self, data: NDArray) -> None:
        with self._floor_plane_lock:
            self.floor_plane = data
        self._floor_plane_event.set()

    def wait_for_floor_plane_update(self, timeout: float = None) -> bool:
        self._floor_plane_event.wait(timeout)
        self._floor_plane_event.clear()

    def mark_full_update(self) -> None:
        self._full_update_event.set()
    
    def wait_for_all_updates(self, timeout: float = None) -> bool:
        self._full_update_event.wait(timeout)
        self._full_update_event.clear()
    
# # ---------------------------------------------------------
# # SHARED VARIABLES
# # ---------------------------------------------------------

shared_zed_data = SharedZedData(resolution=(1280, 720), map_max_size=1000000)

occupancy_grid = np.ones((1000, 1000), dtype=np.bool)
occupancy_grid_lock = Lock()


