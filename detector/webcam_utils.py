import cv2 as cv
import time


def calculate_fps(prev_time):
    """
    Calculate FPS and return:
    (fps_value, current_time)
    """
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    return int(fps), curr_time


def save_screenshot(frame):
    """
    Save screenshot with current timestamp.
    """
    file_name = f"Screenshot_{int(time.time())}.png"
    cv.imwrite(file_name, frame)
    print(f"Saved screenshot: {file_name}")