import threading
import numpy as np
import time
import matplotlib.pyplot as plt

class OccupancyGridViewer(threading.Thread):
    def __init__(self, shared_grid, lock, update_rate=10):
        super().__init__()
        self.shared_grid = shared_grid      # 2D numpy array reference
        self.lock = lock                     # threading.Lock for safety
        self.update_rate = update_rate       # Hz
        self.running = True                  # control flag

    def stop(self):
        self.running = False

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()

        # Initialize the image
        with self.lock:
            img = ax.imshow(self.shared_grid.copy(),
                            cmap='gray',
                            vmin=0,
                            vmax=1)
        plt.show()

        # Update loop
        while self.running:
            with self.lock:
                img.set_data(self.shared_grid.copy())

            fig.canvas.draw_idle()
            plt.pause(1.0 / self.update_rate)

        plt.close(fig)