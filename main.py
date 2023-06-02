import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

def setup(master):
    global gbr
    root = ttk.Frame(master, padding=10)
    root.pack(fill=BOTH, expand=YES)

    frame_left = ttk.Frame(root, padding=5)
    frame_left.pack(side=LEFT, fill=BOTH, expand=YES)
    frame_right = ttk.Frame(root, padding=5)
    frame_right.pack(side=RIGHT, fill=BOTH, expand=YES)

    color_group = ttk.Labelframe(master=frame_left, text="Camera Live", padding=10)
    color_group.pack(fill=X, side=TOP)
    gbr = ttk.Label(color_group)
    gbr.grid(column=0, row=1, columnspan=3)

    control_group = ttk.Labelframe(master=frame_right, text="Pengenalan Wajah", padding=(10, 5))
    control_group.pack(side=TOP,fill=X)
    face = ttk.Label(master=control_group)
    face.grid(column=0, row=0, padx=10)

    ktp_group = ttk.Labelframe(master=frame_right, text="Identifikasi KTP", padding=(10, 5))
    ktp_group.pack(side=TOP, fill=X)
    ktp = ttk.Label(master=ktp_group)
    ktp.grid(column=0, row=0, padx=10)

    btn_group = ttk.Labelframe(master=frame_right, text="Data Reader", padding=(10, 5))
    btn_group.pack(side=TOP, fill=X)

    btn_play = ttk.Button(master=btn_group,
                          text="ANPR Start",
                          bootstyle="btn-info",
                          command=run_camera)
    btn_play.grid(column=0, row=0, padx=5, pady=5, sticky="w")

def face_detection():
    global cap, frame,im,camera
    success, frame = cap.read()
    if success ==True:
        frame = cv2.resize(frame, (470, 312))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        camera = ImageTk.PhotoImage(image=im)
        gbr.configure(image=camera)
        gbr.after(1, face_detection)
    else:
        cap.release()
        cv2.destroyAllWindows()

def run_camera():
    global camscl, camres, camera, cap, runct, gbr
    if runct ==1:
        cap = cv2.VideoCapture("video/widodo.mp4")
    else:
        camscl = Image.open("bg.jpg")
        camres = camscl.resize((470, 312), Image.LANCZOS)
        camera = ImageTk.PhotoImage(camres)
        gbr.configure(image=camera)
        cap =None
        runct = 1
    # -----------------------------------------------------
    if cap is not None:
        runct =2
        face_detection()

# Variable GLobal
runct = 0

if __name__ == "__main__":
    app = ttk.Window(
        title="Face Recognition",
        themename="solar",
        resizable=(False, False),
    )
    setup(app)
    run_camera()
    app.mainloop()