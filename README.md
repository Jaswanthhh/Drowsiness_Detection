
# 💤 Drowsiness & Head Pose Detection System (Hybrid Approach)

This project implements a real-time driver drowsiness detection system using a **hybrid approach** that combines:

- 🔍 **Facial landmark detection** using MediaPipe and Dlib
- 🧠 **Eye state classification** via **transfer learning** (MobileNetV2)
- 📐 **Head pose estimation** using traditional PnP methods
- 🖥️ **Tkinter GUI** to visualize live webcam feed, alert status, head angles, and blink counter

---

## 🔧 Features

- ✅ Real-time drowsiness detection
- ✅ GUI-based display (no OpenCV pop-ups)
- ✅ Alerts with alarm sounds (via Pygame)
- ✅ Head pose estimation: Pitch, Yaw, Roll
- ✅ Blink counting
- ✅ Eye state prediction using deep learning (transfer learning)

---

## 🧠 Architecture

```txt
[ Webcam ]
    ↓
[ Face Landmark Detection (MediaPipe / Dlib) ]
    ├─> EAR Calculation (for blinking)
    ├─> Head Pose Estimation (SolvePnP)
    └─> Eye Region Cropping → CNN (MobileNetV2)
                            → Open / Closed Eye Prediction
                                ↓
                       Drowsiness Decision
                                ↓
                     Tkinter GUI + Alarm Alert
```

---

## 📁 Folder Structure

```txt
├── eye_dataset/                   # Dataset folder for training eye classifier
│   ├── open/
│   └── closed/
├── eye_state_model.h5            # Trained eye state classifier (MobileNetV2)
├── driver8.py                    # Main detection + GUI integration script
├── TRAIN.py                      # Transfer learning script (for CNN training)
├── alarm.wav                     # Alarm sound for drowsy alerts
├── shape_predictor_68...dat      # Dlib facial landmark model
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/your-username/drowsiness-detection-hybrid.git
cd drowsiness-detection-hybrid
```

### 2️⃣ Setup Virtual Environment

```bash
python -m venv drowsy-env
.\drowsy-env\Scriptsctivate       # On Windows
pip install -r requirements.txt
```

> ⚠️ If `dlib` fails, install Visual Studio C++ Build Tools and use:
> ```bash
> pip install dlib‑19.24.0‑cp311‑cp311‑win_amd64.whl
> ```

### 3️⃣ Run the Detection GUI

```bash
python driver8.py
```

---

## 🧪 Training Eye State Classifier (Optional)

If you'd like to retrain the CNN:

1. Place eye images in:
   ```
   eye_dataset/open/
   eye_dataset/closed/
   ```

2. Run:

```bash
python TRAIN.py
```

> This creates: `eye_state_model.h5`

---

## 🖼️ GUI Preview

```
┌─────────────────────────────┐
│     [ Live Webcam Feed ]    │
│  Status: DROWSY             │
│  Blinks: 10                 │
│  Pitch: 3.2°, Yaw: 1.1°,    │
│  Roll: 0.6°                 │
│                             │
│  [Start] [Stop] [Exit]      │
└─────────────────────────────┘
```
## 🔗 Download Required Files

| File | Link |
|------|------|
| shape_predictor_68_face_landmarks.dat | [Download from Dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| eye_state_model.h5 | [Google Drive](https://drive.google.com/your-link-here) |
| Data-Set | [Download from kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new?resource=download). |

---

## 📦 Requirements

- Python 3.10 or 3.11
- TensorFlow < 2.19
- OpenCV
- Mediapipe
- Dlib
- Pillow
- Pygame

Install via:

```bash
pip install -r requirements.txt
```

---

## 📢 Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe)
- [Dlib](http://dlib.net/)
- [Closed Eyes in the Wild Dataset (CEW)](https://www.kaggle.com/datasets/tonyshe/cew)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

---
## Contact for Doubts
- Mail Jaswanth2jaswanth@gmail.com
