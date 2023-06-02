import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

def setup(master):
    global gbr, face,ktp
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

def get_recognition():
    global model, frame, names
    global capture, save_wajah, save_ktp
    global cap_face, arr_face, face_rez,img_face
    global cap_ktp, arr_ktp, ktp_rez, img_ktp
    results = model(frame, stream=True)
    capture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0].item())
            cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255,255,0),t=2,l=20)
            cvzone.putTextRect(frame, f'{names[cls]}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1, colorR=(0, 0, 0),offset=3)
            if cls == 0 and box.conf >= 0.6:
                print(box.conf)
                cap_face = capture[y1:y1 + h,x1:x1 + w]
                if box.conf > 0.94: save_wajah = cap_face
                # else:
                #     if len(save_wajah)==0: save_wajah =cap_face
            if len(save_wajah) > 0:
                arr_face = Image.fromarray(save_wajah)
                face_rez = arr_face.resize((150, 150), Image.LANCZOS)
                img_face = ImageTk.PhotoImage(face_rez)
                face.configure(image=img_face)

            if cls == 1 and box.conf >= 0.5:
                cap_ktp = capture[y1:y1 + h,x1:x1 + w]
                if box.conf > 0.95: save_ktp = cap_ktp
                else:
                    if len(save_ktp)==0: save_ktp =cap_ktp
            if len(save_ktp) > 0:
                arr_ktp = Image.fromarray(save_ktp)
                ktp_rez = arr_ktp.resize((150, 150), Image.LANCZOS)
                img_ktp = ImageTk.PhotoImage(ktp_rez)
                ktp.configure(image=img_ktp)



def run_camera():
    global frame,img_array, video
    global camscl, camres, camera, cap, runct, gbr
    if runct ==1:
        # cap = cv2.VideoCapture("video/widodo.mp4")
        cap = cv2.VideoCapture(1)
        cap.set(3, 800)
        cap.set(4, 640)
    elif runct ==0:
        camscl = Image.open("bg.jpg")
        camres = camscl.resize((300, 380), Image.LANCZOS)
        camera = ImageTk.PhotoImage(camres)
        gbr.configure(image=camera)
        cap =None
        runct = 1
    # -----------------------------------------------------
    if cap is not None:
        runct =2
        success, frame = cap.read()
        if success == True:
            frame = cv2.resize(frame, (300, 380))
            get_recognition()
            video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(video)
            camera = ImageTk.PhotoImage(image=img_array)
            gbr.configure(image=camera)
            gbr.after(1, run_camera)
        else:
            cap.release()
            cv2.destroyAllWindows()

# Variable GLobal
model = YOLO("./yolo/train6/last.pt")
names = ["Wajah", "KTP Elektronik"]
runct = 0
save_wajah=[]
save_ktp=[]

if __name__ == "__main__":
    app = ttk.Window(
        title="Face Recognition",
        themename="solar",
        resizable=(False, False),
    )
    setup(app)
    run_camera()
    app.mainloop()