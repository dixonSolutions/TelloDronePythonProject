import os
import cv2
import random

DEFAULT_DIR = "/var/home/ratrad/Pictures/Tello"

def save_picture_from_frame(frame, directory: str = DEFAULT_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    fname = f"picture_{random.randint(0, 1000000)}.png"
    path = os.path.join(directory, fname)
    cv2.imwrite(path, frame)
    return path





