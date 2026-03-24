from visual_slam.main_loop import main_loop
from visual_slam.InteractiveScreen import InteractiveScreen




if __name__ == "__main__":

    screen = InteractiveScreen()
    screen.start_demo_loop(main_loop)
