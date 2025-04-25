import cv2
import dlib
import numpy as np
import pygame
from threading import Thread
from scipy.spatial import distance as dist
import mediapipe as mp

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG = {
    "thresh": 0.27,        # Eye aspect ratio threshold
    "blink_time": 0.15,    # Blink time threshold in seconds
    "drowsy_time": 2.0,    # Drowsy time threshold in seconds
    "awake_time": 2.0,     # Awake time threshold in seconds to auto-stop the alarm
    "alarm_path": "alarm.wav",  # Path to alarm sound file
}

# Alarm / State Variables
ALARM_ON      = False
STOP_ALARM    = False
AWAKE_TIMER   = 0
DROWSY_TIMER  = 0
BLINK_COUNTER = 0
BLINK_DETECTED= False

# ------------------------------------------------------------------------------
# Initialize pygame for alarm
# ------------------------------------------------------------------------------
pygame.mixer.init()

# ------------------------------------------------------------------------------
# 1) Mediapipe Setup (Face Mesh)
# ------------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks for EAR (MediaPipe)
LEFT_EYE_IDX_MEDIAPIPE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX_MEDIAPIPE = [263, 387, 385, 362, 380, 373]

# Extended head-pose landmarks (MediaPipe)
HEAD_POSE_IDX_EXTENDED = {
    "nose_tip":             4,
    "chin":                 152,
    "left_eye_outer_corner": 33,
    "left_eye_inner_corner": 133,
    "right_eye_outer_corner": 263,
    "right_eye_inner_corner": 362,
    "left_mouth_corner":    61,
    "right_mouth_corner":   291,
    "nose_bridge_mid":      168,
    "lips_upper_mid":       13,
    "lips_lower_mid":       14,
}
HEAD_POSE_IDX_LIST_MEDIAPIPE = list(HEAD_POSE_IDX_EXTENDED.values())

MODEL_POINTS_EXTENDED_MEDIAPIPE = np.array([
    (0.0,   0.0,    0.0),      # Nose tip
    (0.0,  -330.0, -65.0),     # Chin
    (-165.0, 170.0, -135.0),   # Left eye outer corner
    (-50.0,  170.0, -135.0),   # Left eye inner corner
    (165.0,  170.0, -135.0),   # Right eye outer corner
    (50.0,   170.0, -135.0),   # Right eye inner corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0,  -150.0, -125.0),  # Right mouth corner
    (0.0,    50.0,  -130.0),   # Nose bridge
    (0.0,   -25.0,  -120.0),   # Lips upper mid
    (0.0,   -50.0,  -120.0)    # Lips lower mid
], dtype="double")

# ------------------------------------------------------------------------------
# 2) Dlib Setup
# ------------------------------------------------------------------------------
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("C:/Users/jaswa/OneDrive/Desktop/ML/drowsiness_detection/shape_predictor_68_face_landmarks.dat")

# Eye landmarks for EAR (Dlib’s 68-point model)
LEFT_EYE_IDX_DLIB  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX_DLIB = [42, 43, 44, 45, 46, 47]

# Head-pose landmarks (Dlib’s 68-point model):
# Typically: Nose tip=30, Chin=8, Left eye corner=36, Right eye corner=45, 
# Left mouth=48, Right mouth=54 ( minimal version ).
HEAD_POSE_IDX_DLIB = [30, 8, 36, 45, 48, 54]

MODEL_POINTS_DLIB = np.array([
    (0.0,    0.0,     0.0),   # Nose tip
    (0.0,  -330.0,   -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0,  170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0,  -150.0, -125.0)  # Right mouth corner
], dtype="double")

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def play_alarm(sound_path):
    global ALARM_ON, STOP_ALARM
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play(-1)  # loop
    while ALARM_ON:
        if STOP_ALARM:
            pygame.mixer.music.stop()
            STOP_ALARM = False
            ALARM_ON   = False
            break

def rotation_vector_to_euler_angles(rotation_vec):
    """
    Convert a rotation vector (Rodrigues) to Euler angles (pitch, yaw, roll) in degrees.
    """
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_mat[2,1], rotation_mat[2,2])
        yaw   = np.arctan2(-rotation_mat[2,0], sy)
        roll  = np.arctan2(rotation_mat[1,0], rotation_mat[0,0])
    else:
        pitch = np.arctan2(-rotation_mat[1,2], rotation_mat[1,1])
        yaw   = np.arctan2(-rotation_mat[2,0], sy)
        roll  = 0

    pitch = np.degrees(pitch)
    yaw   = np.degrees(yaw)
    roll  = np.degrees(roll)
    return pitch, yaw, roll

# ------------------------------------------------------------------------------
# --- 1) MediaPipe-based EAR & Head Pose
# ------------------------------------------------------------------------------
def eye_aspect_ratio_mediapipe(landmarks, w, h, indices):
    """
    EAR using MediaPipe normalized landmarks -> pixel coords -> Euclidean dist.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

def get_eye_open_status_mediapipe(landmarks, w, h):
    """
    Returns True if average EAR (left+right) >= threshold.
    """
    left_ear = eye_aspect_ratio_mediapipe(landmarks, w, h, LEFT_EYE_IDX_MEDIAPIPE)
    right_ear= eye_aspect_ratio_mediapipe(landmarks, w, h, RIGHT_EYE_IDX_MEDIAPIPE)
    avg_ear  = (left_ear + right_ear) / 2.0
    return avg_ear >= CONFIG["thresh"]

def detect_head_pose_mediapipe(landmarks, frame):
    """
    Extended SolvePnP for MediaPipe. Returns (rotation_vec, translation_vec, (pitch, yaw, roll)).
    """
    h, w = frame.shape[:2]
    image_points = []
    for idx in HEAD_POSE_IDX_LIST_MEDIAPIPE:
        x = landmarks[idx].x * w
        y = landmarks[idx].y * h
        image_points.append((x, y))
    image_points = np.array(image_points, dtype="double")

    # Approx camera matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, r_vec, t_vec = cv2.solvePnP(
        MODEL_POINTS_EXTENDED_MEDIAPIPE, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, (None, None, None)
    pitch, yaw, roll = rotation_vector_to_euler_angles(r_vec)
    return r_vec, t_vec, (pitch, yaw, roll)

# ------------------------------------------------------------------------------
# --- 2) Dlib-based EAR & Head Pose
# ------------------------------------------------------------------------------
def eye_aspect_ratio_dlib(landmarks_dlib):
    """
    EAR using Dlib’s 68 landmark indexing directly.
    """
    # Convert dlib shape to list of (x, y)
    points = [(p.x, p.y) for p in landmarks_dlib.parts()]

    left_pts = [points[i] for i in LEFT_EYE_IDX_DLIB]
    right_pts= [points[i] for i in RIGHT_EYE_IDX_DLIB]

    left_ear = _ear_from_points(left_pts)
    right_ear= _ear_from_points(right_pts)
    avg_ear  = (left_ear + right_ear) / 2.0
    return avg_ear

def _ear_from_points(eye_pts):
    """
    Helper to compute EAR from a 6-point array for one eye.
    """
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def get_eye_open_status_dlib(landmarks_dlib):
    """
    Returns True if Dlib-based EAR >= threshold.
    """
    avg_ear = eye_aspect_ratio_dlib(landmarks_dlib)
    return avg_ear >= CONFIG["thresh"]

def detect_head_pose_dlib(landmarks_dlib, frame):
    """
    SolvePnP for dlib’s 68 landmarks -> minimal 6 points approach.
    Returns (rotation_vec, translation_vec, (pitch, yaw, roll)).
    """
    h, w = frame.shape[:2]
    points = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in HEAD_POSE_IDX_DLIB]
    image_points = np.array(points, dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, r_vec, t_vec = cv2.solvePnP(
        MODEL_POINTS_DLIB, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, (None, None, None)
    pitch, yaw, roll = rotation_vector_to_euler_angles(r_vec)
    return r_vec, t_vec, (pitch, yaw, roll)

# ------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------
def detect_drowsiness_mediapipe_dlib():
    global ALARM_ON, STOP_ALARM, AWAKE_TIMER, DROWSY_TIMER, BLINK_COUNTER, BLINK_DETECTED

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible!")
        return

    print("[INFO] Starting Drowsiness + Head Pose (Mediapipe + Dlib)...")
    prev_eye_open = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera!")
            break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1) Attempt MediaPipe
        mediapipe_results = mp_face_mesh_detector.process(frame_rgb)
        used_mediapipe = False
        used_dlib = False

        if mediapipe_results.multi_face_landmarks:
            # We found a face with MediaPipe
            face_landmarks = mediapipe_results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            # Eye status via MediaPipe
            eye_open = get_eye_open_status_mediapipe(landmarks_list, w, h)

            # Head pose via MediaPipe
            _, _, angles = detect_head_pose_mediapipe(landmarks_list, frame)

            used_mediapipe = True
        else:
            # 2) Fallback to Dlib if MediaPipe fails
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = dlib_detector(gray, 0)
            if len(faces) > 0:
                face_rect = faces[0]
                landmarks_dlib = dlib_predictor(gray, face_rect)

                # Eye status via Dlib
                eye_open = get_eye_open_status_dlib(landmarks_dlib)

                # Head pose via Dlib
                _, _, angles = detect_head_pose_dlib(landmarks_dlib, frame)

                used_dlib = True
            else:
                eye_open = True  # default
                angles = (None, None, None)

        if not used_mediapipe and not used_dlib:
            # No face detected by either approach
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Blink detection
            if not eye_open and prev_eye_open:
                BLINK_DETECTED = True
            elif eye_open and not prev_eye_open and BLINK_DETECTED:
                BLINK_COUNTER += 1
                BLINK_DETECTED = False

            prev_eye_open = eye_open

            # Drowsiness logic
            if eye_open:
                DROWSY_TIMER = 0
                AWAKE_TIMER += 1
                if AWAKE_TIMER * CONFIG["blink_time"] >= CONFIG["awake_time"] and ALARM_ON:
                    STOP_ALARM = True
            else:
                AWAKE_TIMER = 0
                DROWSY_TIMER += 1
                if (DROWSY_TIMER * CONFIG["blink_time"]) >= CONFIG["drowsy_time"] and not ALARM_ON:
                    ALARM_ON = True
                    Thread(target=play_alarm, args=(CONFIG["alarm_path"],)).start()

            # Display
            if ALARM_ON:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Awake", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Blinks: {BLINK_COUNTER}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Drowsiness & Head Pose (Mediapipe + Dlib)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    detect_drowsiness_mediapipe_dlib()
