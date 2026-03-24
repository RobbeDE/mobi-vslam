import ctypes
import importlib
import inspect
import queue
import sys
import threading
import time
import types
from collections import defaultdict
from functools import partial

import pygame
from evdev import InputDevice, list_devices, ecodes, categorize
from loguru import logger


def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising an exception in the thread from
       another thread.
    '''
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL: this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.is_alive(): # Note: self.isAlive() on older version of Python
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do: self.ident

        raise AssertionError("could not determine the thread's id")

    def raise_exc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raise_exc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raise_exc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL: this function is executed in the context of the
        caller thread, to raise an exception in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )

# NOT RELOADABLE
class InteractiveScreen:
    AXIS_MIN = 0        # or -32768
    AXIS_MAX = 255      # or 32767
    midpoint = 0.0
    threshold = 0.5

    def __init__(self, exit_handler=lambda: None):
        self.exit_handler = exit_handler
        pygame.init()
        self._clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((960, 540), flags=pygame.RESIZABLE)
        pygame.display.set_caption("Pygame Viz")
        self.last_key = None
        self.last_image = None
        self.handlers = {}
        self.image_queue = queue.Queue()
        self._running = True
        self._demo_loop_fn_thread = None
        self._demo_loop_kwargs = None
        self._demo_loop_args = None
        self._demo_loop_fn = None
        self.camera = None
        self._camera_timer = time.monotonic()

        # Thread-safe queue to pass events from evdev thread to main pygame thread
        self.event_queue = queue.Queue()

        self.last_axis_values = {}

        # Start evdev listener in background thread
        # threading.Thread(target=self.evdev_listener, daemon=True).start()

        # For custom input handling
        self._input_active = False
        self._input_buffer = []
        self._input_done = threading.Event()
        self._input_lock = threading.Lock()
        self._touchpad_pressed = False

        # Pause / resume signal
        self.pause_event = threading.Event()
        self.pause_event.set()  # Initially not paused
        self.add_event_handler(
            lambda: self.pause_event.clear(), # Pause on Ctrl+P
            key='p',
            modifiers=['ctrl'],
            controller_button='BTN_EAST'
        )
        self.add_event_handler(
            lambda: self.pause_event.set(),  # Resume on Ctrl+O
            key='o',
            modifiers=['ctrl'],
            controller_button='BTN_SOUTH'
        )

    def start_demo_loop(self, demo_loop_fn, *args, **kwargs):
        """Start the demo loop in a background thread."""
        self._demo_loop_fn = demo_loop_fn
        self._demo_loop_args = args
        self._demo_loop_kwargs = kwargs

        self._start_demo_thread()

        # Ctrl+D restarts and reloads the function’s module
        self.add_event_handler(
            self._restart_demo_loop,
            args=[demo_loop_fn],
            key='d',
            modifiers=['ctrl'],
            controller_button='BTN_TR'
        )

        # Flush controller event queue on start
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        self._event_loop()

    def _start_demo_thread(self):
        self.pause_event.set()  # Initially not paused
        self._demo_loop_fn_thread = ThreadWithExc(
            target=self._demo_loop_fn,
            args=[self, *self._demo_loop_args],
            kwargs=self._demo_loop_kwargs,
            daemon=True
        )
        self._demo_loop_fn_thread.start()
        logger.info("[DemoLoop] Started demo loop thread.")


    @staticmethod
    def deep_reload(module):
        """
        Recursively reload the given module and all its submodules.
        Returns the reloaded top-level module.
        """
        def _toposort(nodes, edges):
            """Return a list of nodes sorted so that all edges A→B appear before B."""
            incoming = {n: set() for n in nodes}
            for n, deps in edges.items():
                for d in deps:
                    incoming[n].add(d)

            sorted_list = []
            while incoming:
                # modules with no dependencies
                ready = [n for n, deps in incoming.items() if not deps]
                if not ready:
                    # cycle detected, just append arbitrarily
                    sorted_list.extend(incoming.keys())
                    break
                for r in sorted(ready):
                    sorted_list.append(r)
                    del incoming[r]
                    for deps in incoming.values():
                        deps.discard(r)
            return sorted_list
        if not isinstance(module, types.ModuleType):
            raise TypeError(f"Expected a module, got {type(module)}")

        name = module.__name__
        logger.warning(f"Deep reloading module tree: {name}")

        # --- 1. Gather all relevant submodules ---
        submodules = {
            n: m for n, m in sys.modules.items()
            if n == name or n.startswith(name + ".")
        }

        # --- 2. Build dependency graph based on class inheritance ---
        dependencies = defaultdict(set)  # module -> set(of modules it depends on)

        for mod_name, mod in submodules.items():
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                for base in obj.__bases__:
                    base_mod = getattr(base, "__module__", None)
                    if base_mod and base_mod in submodules and base_mod != mod_name:
                        dependencies[mod_name].add(base_mod)

        # --- 3. Topologically sort the modules ---
        sorted_modules = _toposort(submodules.keys(), dependencies)

        # --- 4. Reload in topological order ---
        for mod_name in sorted_modules:
            mod = submodules[mod_name]
            try:
                logger.warning(f"Reloading submodule: {mod_name}")
                importlib.reload(mod)
            except Exception as e:
                logger.error(f"Failed to reload submodule {mod_name}: {e}")

        logger.warning(f"Finished deep reload for {name}")
        return importlib.reload(module)

    def _restart_demo_loop(self, demo_loop_fn):
        """Stop, reload, and restart the demo loop."""
        if self._demo_loop_fn_thread and self._demo_loop_fn_thread.is_alive():
            logger.warning("Stopping existing demo loop...")
            try:
                self._demo_loop_fn_thread.raise_exc(SystemExit)
                # Also send enter key to unblock input()
                with self._input_lock:
                    self._input_active = False
                    self._input_done.set()
                # And unpause the demo
                self.pause_event.set()
                while self._demo_loop_fn_thread.is_alive():
                    logger.warning("Waiting for demo loop to finish...")
                    self._demo_loop_fn_thread.raise_exc(SystemExit)
                    # Also send enter key to unblock input()
                    with self._input_lock:
                        self._input_active = False
                        self._input_done.set()
                    # And unpause the demo
                    self.pause_event.set()
                    time.sleep(0.1)
            except Exception as e:
                logger.exception(f"Error while stopping demo loop: {e}")
            logger.warning("Existing demo loop stopped.")
        # Re-import visual_slam
        import visual_slam
        self.deep_reload(visual_slam)

        # Hot reload the module that defines demo_loop_fn
        module = inspect.getmodule(demo_loop_fn)
        if module is not None:
            logger.warning(f"Deep reloading: {module.__name__}")
            reloaded_module = self.deep_reload(module)
            self._demo_loop_fn = getattr(reloaded_module, demo_loop_fn.__name__)
        else:
            logger.error("Could not determine module of demo_loop_fn; skipping reload.")

        # Restart the loop
        logger.warning("Restarting demo loop...")
        self._start_demo_thread()

    def set_exit_handler(self, exit_handler, exit_args=None) -> None:
        self.exit_handler = exit_handler
        self.exit_args = exit_args

    def _event_loop(self) -> None:
        self.pressed_keys = set()
        while self._running:
            if self.camera is not None and time.monotonic() - 0.2 > self._camera_timer:
                image = self.camera.get_rgb_image_as_int()
                self._do_imshow(image)
                self._camera_timer = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Quitting...")
                    if self.exit_args is not None:
                        self.exit_handler(*self.exit_args)
                    else:
                        self.exit_handler()
                    if self._demo_loop_fn_thread: self._demo_loop_fn_thread.raise_exc(SystemExit)
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.VIDEORESIZE:
                    if self.last_image is not None:
                        self._do_imshow(self.last_image)
                elif event.type == pygame.KEYDOWN:
                    ch = event.unicode
                    self.pressed_keys.add(event.unicode)
                    self.last_key = ch

                    if self._input_active:
                        if ch == '\r' or ch == '\n':
                            with self._input_lock:
                                self._input_active = False
                                self._input_done.set()
                        elif ch == '\b':
                            with self._input_lock:
                                if self._input_buffer:
                                    self._input_buffer.pop()
                                print('\b \b', end='', flush=True)
                        else:
                            with self._input_lock:
                                self._input_buffer.append(ch)
                                print(ch, end='', flush=True)


                    # logger.info(f"Key pressed: {pygame.key.name(event.key)}")
                    mods = pygame.key.get_mods()
                    mod_list = []
                    if mods & pygame.KMOD_CTRL:
                        mod_list.append('ctrl')
                    if mods & pygame.KMOD_SHIFT:
                        mod_list.append('shift')
                    if mods & pygame.KMOD_ALT:
                        mod_list.append('alt')
                    handler = self.handlers.get((pygame.key.name(event.key), tuple(sorted(mod_list))))
                    if handler:
                        try:
                            logger.info(f"Invoking event handler for key {pygame.key.name(event.key)} with modifiers {mod_list} -> {handler}")
                            threading.Thread(target=handler).start()
                        except Exception as e:
                            logger.exception(f"Error in event handler for key {pygame.key.name(event.key)} with modifiers {mod_list}: {e}")
                elif event.type == pygame.KEYUP:
                    # logger.info(f"Key released: {pygame.key.name(event.key)}")
                    self.pressed_keys.discard(event.unicode)

            # Handle evdev events even when pygame window unfocused
            while not self.event_queue.empty():
                evdev_event = self.event_queue.get()
                if evdev_event.type == ecodes.EV_KEY:
                    key_event = categorize(evdev_event)
                    key_code = key_event.keycode
                    # Normalize keycodes to a list of strings
                    if isinstance(key_code, str):
                        keycodes = [key_code]
                    elif isinstance(key_code, (tuple, list)):
                        keycodes = list(key_code)
                    else:
                        # fallback to string conversion
                        keycodes = [str(key_code)]
                    for key_code in keycodes:
                        if key_event.keystate == key_event.key_down:
                            if key_code == 'BTN_MOUSE':
                                self._touchpad_pressed = True
                                if self._input_active:
                                    with self._input_lock:
                                        # if buffer is empty, then add '0'
                                        if not self._input_buffer:
                                            self._input_buffer.append('0')
                                            print('0', end='', flush=True)
                            elif key_code == 'BTN_SOUTH':
                                if self._input_active:
                                    with self._input_lock:
                                        self._input_active = False
                                        self._input_done.set()
                            # logger.info(f"Controller button {key_code} pressed.")
                            handler = self.handlers.get((f'controller_{key_code}', ()))
                            if handler:
                                try:
                                    logger.info(f"Invoking event handler for controller button {key_code} -> {handler}")
                                    threading.Thread(target=handler).start()
                                except Exception as e:
                                    logger.exception(f"Error in event handler for controller button {key_code}: {e}")
                        elif key_event.keystate == key_event.key_up:
                            if key_code == 'BTN_MOUSE':
                                self._touchpad_pressed = False
                            # logger.info(f"Controller button {key_code} released.")

                elif evdev_event.type == ecodes.EV_ABS:
                    abs_event = categorize(evdev_event)
                    abs_code = ecodes.ABS[abs_event.event.code]
                    value = abs_event.event.value
                    if abs_code == 'ABS_HAT0X' or abs_code == 'ABS_HAT0Y':
                        # value will be -1, 0, or 1
                        if self._input_active and self._touchpad_pressed:
                            with self._input_lock:
                                # Map hat directions to increasing/decreasing a number in the input buffer
                                if self._input_buffer:
                                    try:
                                        current_value = int(''.join(self._input_buffer))
                                    except ValueError:
                                        current_value = 0
                                else:
                                    current_value = 0
                                current_value = current_value + (-1 if value == 1 else 1 if value == -1 else 0) * (1 if abs_code == 'ABS_HAT0Y' else -5)
                                self._input_buffer = list(str(current_value))
                                print(f'\r{current_value} ', end='', flush=True)
                            continue

                        # logger.info(f"Hat {abs_code} moved to {value}")
                        handler = self.handlers.get((f'controller_{abs_code}_{value}', ()))
                        if handler:
                            try:
                                logger.info(f"Invoking event handler for hat {abs_code} value {value} -> {handler}")
                                threading.Thread(target=handler).start()
                            except Exception as e:
                                logger.exception(f"Error in event handler for hat {abs_code} value {value}: {e}")
                        continue

                    norm_value = self.normalize_axis(value, self.AXIS_MIN, self.AXIS_MAX)
                    last_norm_value = self.last_axis_values.get(abs_code, 0.0)
                    self.last_axis_values[abs_code] = norm_value


                    if norm_value < self.midpoint - self.threshold < last_norm_value:
                        logger.info(f"Controller axis {abs_code} moved to negative direction.")
                        handler = self.handlers.get((f'controller_{abs_code}_move_negative', ()))
                        if handler:
                            try:
                                logger.info(f"Invoking event handler for controller axis {abs_code} negative move -> {handler}")
                                threading.Thread(target=handler).start()
                            except Exception as e:
                                logger.exception(f"Error in event hasegsdfndler for controller axis {abs_code} negative move: {e}")
                    elif norm_value > self.midpoint + self.threshold > last_norm_value:
                        logger.info(f"Controller axis {abs_code} moved to positive direction.")
                        handler = self.handlers.get((f'controller_{abs_code}_move_positive', ()))
                        if handler:
                            try:
                                logger.info(f"Invoking event handler for controller axis {abs_code} positive move -> {handler}")
                                threading.Thread(target=handler).start()
                            except Exception as e:
                                logger.exception(f"Error in event handler for controller axis {abs_code} positive move: {e}")
                    elif abs(norm_value) <= self.threshold < abs(last_norm_value):
                        logger.info(f"Controller axis {abs_code} returned to neutral position.")

            # Display the most recent image in the queue
            try:
                image = self.image_queue.get_nowait()
                self._do_imshow(image)
            except queue.Empty:
                pass

            # Cap the frame rate to 60 FPS
            self._clock.tick(60)

    def is_key_pressed(self, key: str):
        return key in self.pressed_keys

    def get_last_key(self):
        key = self.last_key
        self.last_key = ''
        return key

    def add_event_handler(self, handler, key=None, args=None, kwargs=None, modifiers=None, controller_button=None):
        if modifiers is None:
            modifiers = []
        if key is not None:
            self.handlers[key, tuple(sorted(modifiers))] = partial(handler, *(args or []), **(kwargs or {}))
        if controller_button is not None:
            self.handlers[f'controller_{controller_button}', ()] = partial(handler, *(args or []), **(kwargs or {}))

    def set_camera(self, camera):
        self.camera = camera

    def imshow(self, image):
        """Thread-safe: called from any thread."""
        # If there is a previous image in the queue, discard it
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        self.image_queue.put(image)
        # logger.info(f"Queued image for display (currently {self.image_queue.qsize()} images in queue)")

    def _do_imshow(self, image):
        """Must be called from the main thread."""
        # logger.info("Displaying image on screen")
        sw, sh = self.screen.get_size()
        screen_aspect_ratio = sw / sh
        ih, iw = image.shape[:2]
        img_aspect_ratio = iw / ih

        if screen_aspect_ratio > img_aspect_ratio:
            new_height = sh
            new_width = int(sh * img_aspect_ratio)
        else:
            new_width = sw
            new_height = int(sw / img_aspect_ratio)

        resized_image = pygame.transform.smoothscale(
            pygame.surfarray.make_surface(image.swapaxes(0, 1)), (new_width, new_height)
        )

        self.screen.fill((0, 0, 0))
        self.screen.blit(resized_image, ((sw - new_width) // 2, (sh - new_height) // 2))
        pygame.display.flip()

        self.last_image = image

    @staticmethod
    def find_controller():
        devices = [InputDevice(path) for path in list_devices()]
        for device in devices:
            if "Wireless Controller" in device.name:
                return device
        return None

    @staticmethod
    def find_touchpad():
        devices = [InputDevice(path) for path in list_devices()]
        for device in devices:
            if "Touchpad" in device.name or "Trackpad" in device.name:
                return device
        return None

    def evdev_listener(self):
        dev = self.find_controller()
        if dev is None:
            logger.warning("No controller found!")
            return
        logger.success(f"Listening on {dev.path} ({dev.name})")

        dev_touchpad = self.find_touchpad()
        if dev_touchpad is not None:
            logger.success(f"Also listening on touchpad {dev_touchpad.path} ({dev_touchpad.name})")
            threading.Thread(target=self.evdev_touchpad_listener, args=(dev_touchpad,), daemon=True).start()

        for event in dev.read_loop():
            if event.type in (ecodes.EV_KEY, ecodes.EV_ABS):
                self.event_queue.put(event)

    def evdev_touchpad_listener(self, dev_touchpad):
        for event in dev_touchpad.read_loop():
            # if event.type in (ecodes.EV_KEY, ecodes.EV_ABS):
            if event.type == ecodes.EV_KEY: # ignore ABS events from touchpad for now
                self.event_queue.put(event)
                # BTN_TOUCH / BTN_TOOL_FINGER can be used to detect touch events
                # BTN_LEFT / BTN_MOUSE can be used to detect clicks

    @staticmethod
    def normalize_axis(value, min_val=AXIS_MIN, max_val=AXIS_MAX):
        """
        Normalize raw axis value to range [-1, 1].
        """
        # If your raw range is symmetric around zero (e.g. -32768 to 32767), use this:
        if min_val < 0:
            norm = value / max(abs(min_val), abs(max_val))
        else:
            # Otherwise map [min_val, max_val] to [-1, 1]
            norm = 2 * (value - min_val) / (max_val - min_val) - 1
        return max(-1, min(norm, 1))  # clamp between -1 and 1

    def input(self, prompt=""):
        logger.warning(f"Prompting for user input (Controller OR Pygame Viz screen, NOT in terminal):\n"
                    f"{prompt} ")
        with self._input_lock:
            self._input_active = True
            self._input_buffer = []
            self._input_done.clear()

        self._input_done.wait()

        with self._input_lock:
            user_input = ''.join(self._input_buffer)
            self._input_active = False
            self._input_buffer = []

        print()  # Newline after input

        return user_input