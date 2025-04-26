import cv2
import dlib
import numpy as np
import pygame
import mediapipe as mp
import threading
from tkinter import Tk, Label, Button, StringVar, Frame
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from threading import Thread

# ------------------------------------------------------------------------------
# Config & State
# ------------------------------------------------------------------------------
CONFIG = {
    "thresh":     0.27,
    "blink_time": 0.15,
    "drowsy_time":2.0,
    "awake_time": 2.0,
    "alarm_path": "alarm.wav"
}

ALARM_ON     = False
STOP_ALARM   = False
AWAKE_TIMER  = 0
DROWSY_TIMER = 0
BLINK_COUNTER= 0
BLINK_DETECTED=False

shared_state = {
    "frame":      None,
    "status":     "Initializing...",
    "blinks":     0,
    "angles":     (0.0, 0.0, 0.0),
    "running":    False,
    # Phase 2 crops (for future model input)
    "left_eye":   None,
    "right_eye":  None,
    "mouth":      None
}

# ------------------------------------------------------------------------------
# Alarm
# ------------------------------------------------------------------------------
pygame.mixer.init()
def play_alarm(path):
    global ALARM_ON, STOP_ALARM
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(-1)
    while ALARM_ON:
        if STOP_ALARM:
            pygame.mixer.music.stop()
            STOP_ALARM = False
            ALARM_ON   = False
            break

# ------------------------------------------------------------------------------
# MediaPipe & Dlib Setup
# ------------------------------------------------------------------------------
mp_face_mesh     = mp.solutions.face_mesh
face_mesh        = mp_face_mesh.FaceMesh(refine_landmarks=True)
dlib_detector    = dlib.get_frontal_face_detector()
dlib_predictor   = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE_IDX_MEDIAPIPE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX_MEDIAPIPE  = [263, 387, 385, 362, 380, 373]
MOUTH_IDX_MEDIAPIPE      = [61, 291, 13, 14, 78, 308, 82, 312]
HEAD_IDX_MEDIAPIPE       = [4, 152, 33, 133, 263, 362, 61, 291, 168, 13, 14]
MODEL_MEDIAPIPE = np.array([
    (0.0, 0.0, 0.0), (0.0, -330, -65), (-165, 170, -135), (-50, 170, -135),
    (165, 170, -135), (50, 170, -135), (-150, -150, -125), (150, -150, -125),
    (0, 50, -130), (0, -25, -120), (0, -50, -120)
], dtype="double")

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def rotation_vector_to_euler_angles(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    sy       = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rmat[2,1], rmat[2,2])
        y = np.arctan2(-rmat[2,0], sy)
        z = np.arctan2(rmat[1,0], rmat[0,0])
    else:
        x = np.arctan2(-rmat[1,2], rmat[1,1])
        y = np.arctan2(-rmat[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def eye_aspect_ratio(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A+B)/(2.0*C) if C!=0 else 0

def crop_eye_and_mouth(frame, landmarks, img_size=224, margin=0.2):
    """Crop left eye, right eye, and mouth from frame using MediaPipe landmarks."""
    h, w = frame.shape[:2]
    def get_crop(idxs):
        pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idxs])
        x,y,bw,bh = cv2.boundingRect(pts)
        px,py = int(bw*margin), int(bh*margin)
        x1,y1 = max(x-px,0), max(y-py,0)
        x2,y2 = min(x+bw+px,w), min(y+bh+py,h)
        crop = frame[y1:y2, x1:x2]
        if crop.size==0:
            return np.zeros((img_size,img_size,3), dtype=np.uint8)
        return cv2.resize(crop, (img_size,img_size))
    left  = get_crop(LEFT_EYE_IDX_MEDIAPIPE)
    right = get_crop(RIGHT_EYE_IDX_MEDIAPIPE)
    mouth = get_crop(MOUTH_IDX_MEDIAPIPE)
    return left, right, mouth

# ------------------------------------------------------------------------------
# Main Detection Loop
# ------------------------------------------------------------------------------
def detect():
    global ALARM_ON, STOP_ALARM, AWAKE_TIMER, DROWSY_TIMER, BLINK_COUNTER, BLINK_DETECTED
    cap       = cv2.VideoCapture(0)
    prev_open = True

    while shared_state["running"]:
        ret, frame = cap.read()
        if not ret: break
        shared_state["frame"] = frame.copy()
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        eye_open  = True
        pitch,yaw,roll = 0.0,0.0,0.0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # --- Phase 2: crop eye & mouth
            le_crop, re_crop, m_crop = crop_eye_and_mouth(frame, lm)
            shared_state["left_eye"]  = le_crop
            shared_state["right_eye"] = re_crop
            shared_state["mouth"]     = m_crop

            # existing EAR-based logic
            le_pts = [(lm[i].x*frame.shape[1], lm[i].y*frame.shape[0]) for i in LEFT_EYE_IDX_MEDIAPIPE]
            re_pts = [(lm[i].x*frame.shape[1], lm[i].y*frame.shape[0]) for i in RIGHT_EYE_IDX_MEDIAPIPE]
            ear = (eye_aspect_ratio(le_pts) + eye_aspect_ratio(re_pts))/2
            eye_open = ear >= CONFIG["thresh"]

            # head-pose
            img_pts = np.array([(lm[i].x*frame.shape[1], lm[i].y*frame.shape[0]) for i in HEAD_IDX_MEDIAPIPE], dtype="double")
            cam_M   = np.array([[frame.shape[1],0,frame.shape[1]/2],
                                [0,frame.shape[1],frame.shape[0]/2],
                                [0,0,1]], dtype="double")
            _, rvec, _ = cv2.solvePnP(MODEL_MEDIAPIPE, img_pts, cam_M, np.zeros((4,1)))
            pitch,yaw,roll = rotation_vector_to_euler_angles(rvec)

        # blink detection
        if not eye_open and prev_open:
            BLINK_DETECTED = True
        elif eye_open and not prev_open and BLINK_DETECTED:
            BLINK_COUNTER += 1
            BLINK_DETECTED = False
        prev_open = eye_open

        # drowsiness logic
        if eye_open:
            DROWSY_TIMER = 0
            AWAKE_TIMER += 1
            if AWAKE_TIMER*CONFIG["blink_time"]>=CONFIG["awake_time"] and ALARM_ON:
                STOP_ALARM = True
        else:
            AWAKE_TIMER = 0
            DROWSY_TIMER += 1
            if DROWSY_TIMER*CONFIG["blink_time"]>=CONFIG["drowsy_time"] and not ALARM_ON:
                ALARM_ON = True
                Thread(target=play_alarm, args=(CONFIG["alarm_path"],)).start()

        shared_state.update({
            "status": "DROWSY" if ALARM_ON else "Awake",
            "blinks": BLINK_COUNTER,
            "angles": (pitch,yaw,roll)
        })

    cap.release()
    shared_state["frame"] = None

# ------------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------------
class DrowsinessGUI:
    def __init__(self, root):
        self.root       = root
        self.root.title("Drowsiness & Head Pose Detection")
        self.root.geometry("1000x720")
        self.video_label= Label(self.root)
        self.video_label.pack()

        self.status_var= StringVar()
        self.blinks_var= StringVar()
        self.pose_var  = StringVar()

        Label(self.root, textvariable=self.status_var, font=("Arial",14)).pack()
        Label(self.root, textvariable=self.blinks_var, font=("Arial",14)).pack()
        Label(self.root, textvariable=self.pose_var, font=("Arial",14)).pack()

        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Start", command=self.start, width=15, bg="green", fg="white").grid(row=0,column=0,padx=10)
        Button(btn_frame, text="Stop",  command=self.stop,  width=15, bg="red",   fg="white").grid(row=0,column=1,padx=10)
        Button(btn_frame, text="Exit",  command=self.root.quit, width=15).grid(row=0,column=2,padx=10)

        self.update_gui()

    def start(self):
        if not shared_state["running"]:
            shared_state["running"] = True
            threading.Thread(target=detect, daemon=True).start()

    def stop(self):
        shared_state["running"] = False

    def update_gui(self):
        # display video
        if shared_state["frame"] is not None:
            img = cv2.cvtColor(shared_state["frame"], cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        # update labels
        self.status_var.set(f"Status: {shared_state['status']}")
        self.blinks_var.set(f"Blinks: {shared_state['blinks']}")
        p,y,r = shared_state["angles"]
        self.pose_var.set(f"Pitch: {p:.1f}°, Yaw: {y:.1f}°, Roll: {r:.1f}°")

        self.root.after(10, self.update_gui)

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    root = Tk()
    app  = DrowsinessGUI(root)
    root.mainloop()
