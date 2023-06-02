import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

def setup(master):
    root = ttk.Frame(master, padding=10)
    root.pack(fill=BOTH, expand=YES)

    frame_left = ttk.Frame(root, padding=5)
    frame_left.pack(side=LEFT, fill=BOTH, expand=YES)
    frame_right = ttk.Frame(root, padding=5)
    frame_right.pack(side=RIGHT, fill=BOTH, expand=YES)

    color_group = ttk.Labelframe(master=frame_left, text="Camera Live", padding=10)
    color_group.pack(fill=X, side=TOP)

if __name__ == "__main__":
    app = ttk.Window(
        title="Face Recognition",
        themename="solar",
        resizable=(False, False),
    )
    setup(app)
    app.mainloop()