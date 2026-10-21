# How to Run — FaceX Whole-Person Tracking

> **Week focus:** Whole-person tracking and face-association software that maintains temporal identity continuity across face loss through an honest three-state model (Confirmed → Temporary → Unknown) with exponential confidence decay.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Dependencies](#2-install-dependencies)
3. [Download ML Models](#3-download-ml-models)
4. [First-Time Setup: Enroll Your Face](#4-first-time-setup-enroll-your-face)
5. [Running Whole-Person Tracking (Main Focus)](#5-running-whole-person-tracking-main-focus)
6. [Running Simulated Scenarios (No Camera Needed)](#6-running-simulated-scenarios-no-camera-needed)
7. [Evaluating Simulation Results](#7-evaluating-simulation-results)
8. [CPU Performance Benchmark](#8-cpu-performance-benchmark)
9. [Other Modules (Parts 1 and 2)](#9-other-modules-parts-1-and-2)
10. [Running Tests](#10-running-tests)
11. [Understanding the Three-State Model](#11-understanding-the-three-state-model)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

- **Python:** 3.11 or 3.12 (64-bit). Check with `python --version`.
- **Webcam:** Any USB or built-in webcam (index 0 by default).
- **OS:** Windows, Linux, or macOS with a graphical desktop session.
- **Internet:** Required once to download the ML model weights (~260 MB for ArcFace).
- **CPU-only:** No GPU required — everything runs on CPU.

---

## 2. Install Dependencies

Install all packages **globally** so they are available system-wide:

```powershell
pip install "numpy>=1.26,<2" "opencv-contrib-python>=4.10,<4.12" "onnxruntime>=1.18,<2" "mediapipe==0.10.21" "paho-mqtt>=1.6,<2"
```

> If you already have `opencv-python` or `opencv-python-headless` installed,
> remove them first to avoid conflicts:
> ```powershell
> pip uninstall -y opencv-python opencv-python-headless
> pip install "opencv-contrib-python>=4.10,<4.12"
> ```

Verify the install:

```powershell
python -c "import cv2, numpy, onnxruntime, mediapipe; print('All OK')"
```

---

## 3. Download ML Models

Run from the project root (`Face-Locking/`):

```powershell
python init_project.py --download-models
```

This downloads and verifies:

| Model | Purpose | Size |
|---|---|---|
| `models/detector_yunet.onnx` | Face detection + 5 landmarks | ~233 KB |
| `models/embedder_arcface.onnx` | 512-D face embedding (ArcFace ResNet100) | ~261 MB |

Re-running is safe — already-verified models are reused.

---

## 4. First-Time Setup: Enroll Your Face

Before tracking, enroll your face so the system knows who to follow:

```powershell
python -m src.enroll --name "YourName" --samples 8
```

**During enrollment:**
1. Keep your face centered, well-lit, and fully visible.
2. Wait for the **Ready** prompt, then press **Space** to capture each sample.
3. Slightly vary your expression and head angle between captures.
4. Press **Q** or **Escape** to cancel without saving.

After all 8 samples are captured, the identity is saved automatically to `data/db/face_db.json`.

To re-enroll (overwrite existing data):
```powershell
python -m src.enroll --name "YourName" --replace
```

---

## 5. Running Whole-Person Tracking (Main Focus)

This is the **core Week 3 module** — it tracks your whole body, not just your face.

### Basic launch (webcam):

```powershell
python -m src.person_tracking --target "YourName"
```

### Full command with common options:

```powershell
python -m src.person_tracking --target "YourName" ^
    --camera 0 ^
    --person-backend hog ^
    --detect-every 3 ^
    --half-life 3.0 ^
    --threshold 0.45 ^
    --margin 0.05
```

### What you will see on screen:

| Overlay | Meaning |
|---|---|
| Green box + `CONFIRMED` | Face recognised recently; full tracking active |
| Yellow box + `TEMPORARY` | Face not visible; body tracked by motion, confidence decaying |
| No box / `UNKNOWN` | Identity unconfirmed or confidence expired |
| Confidence number `0.0–1.0` | Current identity confidence (decays over time when face hidden) |

Press **Q** or **Escape** to quit.

### Key parameters explained:

| Parameter | Default | Purpose |
|---|---|---|
| `--target` | *(required)* | Exact enrolled name to follow |
| `--camera` | `0` | Camera index (try `1` if default fails) |
| `--person-backend` | `hog` | Body detector: `hog` (no download), `none` (face-derived body), `yolo` |
| `--half-life` | `3.0` | Seconds for confidence to halve when face is hidden |
| `--detect-every` | `3` | Run body detector every N frames (larger = faster) |
| `--threshold` | `0.45` | ArcFace cosine similarity threshold |
| `--no-display` | *(flag)* | Run headless (video file mode) |
| `--mqtt-host` | *(optional)* | MQTT broker host for hardware integration |

### Test real-world scenarios:

| Scenario | What to do | Expected behaviour |
|---|---|---|
| **Head turn** | Look away from camera | State switches to `TEMPORARY`, box stays on body |
| **Occlusion** | Walk behind an object | Kalman filter predicts position; confidence decays |
| **Path crossing** | Another person walks in front | Confidence decays faster during overlap; swap is refused |
| **Return** | Come back into view | Face re-confirmed → `CONFIRMED` state restored |

---

## 6. Running Simulated Scenarios (No Camera Needed)

The simulator renders scripted scenarios — no webcam or models required.

### List all available scenarios:

```powershell
python -m src.person_tracking.sim --list
```

Available scenarios: `head_turn`, `long_turn`, `occlusion`, `long_occlusion`,
`path_crossing`, `crossing_same_colour`, `leave_return`, `strangers_only`, `crowd`

### Run a scenario with live display:

```powershell
python -m src.person_tracking.sim --scenario head_turn --show
python -m src.person_tracking.sim --scenario path_crossing --show
python -m src.person_tracking.sim --scenario occlusion --show
```

### Save an annotated video:

```powershell
python -m src.person_tracking.sim --scenario head_turn --show --save data/reports/head_turn.mp4
```

### Save a summary sheet:

```powershell
python -m src.person_tracking.sim --scenario occlusion --sheet data/reports/occlusion.png
```

---

## 7. Evaluating Simulation Results

Runs each scenario across multiple random seeds and scores against ground truth:

```powershell
python -m src.person_tracking.sim.evaluate --seeds 5
```

This reports per-scenario metrics including:
- **Coverage** (% of frames with correct tracking)
- **Face-loss coverage** (% correct when face is not visible)
- **Wrong-hold %** (false identity locks)
- **Ghost lock seconds** (time locked on the wrong person)

Reports are saved to `data/reports/`.

---

## 8. CPU Performance Benchmark

Measures per-stage timings on your machine:

```powershell
python -m src.person_tracking.benchmark
```

Typical results on a mid-range CPU at 640×480:
- YuNet detection: ~18 ms
- HOG body detector: ~5 ms (every 3rd frame)
- Tracker step: ~1 ms
- ArcFace R100: ~108 ms per face

---

## 9. Other Modules (Parts 1 and 2)

These are earlier stages of the project:

```powershell
# Part 1 — Face recognition
python -m src.recognize --camera 0 --threshold 0.45 --margin 0.05

# Part 2 — Single face lock (face must stay visible)
python -m src.face_tracking --target "YourName"

# Live camera preview
python -m src.camera

# YuNet landmarks + aligned face preview
python -m src.landmarks

# Live embedding dimension check
python -m src.embed
```

---

## 10. Running Tests

```powershell
# All tests (no camera or models needed)
python -m unittest discover -s tests -v

# Part 3 (person tracking) tests only
python -m unittest tests.test_person_tracking -v
```

All 10 tests should pass. Two smoke tests check live model inference and skip if models are absent.

---

## 11. Understanding the Three-State Model

```
UNKNOWN  --(face match + enough evidence)-->  CONFIRMED  --(no face, decay < 0.50)-->  TEMPORARY
   ^                                               ^                                         |
   |                                               +-------(new face match → c=0.70)--------+
   +----(decay < 0.20, different person, swap)---------------------------------------------+
```

**Confidence formula:** `c(t) = c₀ × 0.5^(t / half_life)`

| Situation | Confidence decay multiplier |
|---|---|
| Normal face loss | 1× (half-life controls speed) |
| Crossing another person | 2.5× faster |
| Coasting (body undetected) | 1.5× faster |
| Predicted outside frame | 4× faster |

**Thresholds (defaults):**

| Threshold | Value | Role |
|---|---|---|
| `--confirm` | `0.70` | Minimum confidence to enter `CONFIRMED` |
| Drop | `0.50` | Falls to `TEMPORARY` below this |
| `--unknown` | `0.20` | Falls to `UNKNOWN` below this |

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named cv2` | `pip install opencv-contrib-python` |
| `No module named mediapipe` | `pip install mediapipe==0.10.21` |
| Camera won't open | Close Teams/Zoom, try `--camera 1`, check OS camera permissions |
| Always shows `UNKNOWN` | Re-enroll with better lighting; check `--threshold 0.45` |
| HOG body detector misses you | Switch to `--person-backend none` (uses face-derived body boxes) |
| Slow (< 5 FPS) | Increase `--detect-every 5`, ensure only one face in view |
| Model mismatch error | Re-enroll: `python -m src.enroll --name "YourName" --replace` |
| No display window | Must run in a desktop session (not SSH without X forwarding) |
| Missing model files | `python init_project.py --download-models` |

---

## Project Structure Quick Reference

```
Face-Locking/
├── init_project.py              # Model downloader + directory setup
├── requirements.txt             # Python dependencies
├── models/                      # YuNet + ArcFace ONNX weights (after download)
├── data/
│   ├── db/face_db.json          # Enrolled face database
│   ├── logs/                    # Tracking session CSV logs
│   └── reports/                 # Simulation evaluation reports
└── src/
    ├── enroll.py                # Webcam face enrollment
    ├── recognize.py             # Live face recognition (Part 1)
    ├── face_tracking.py         # Single-face lock (Part 2)
    └── person_tracking/         # Whole-person tracking (Part 3 — Week focus)
        ├── __main__.py          # Entry point: python -m src.person_tracking
        ├── tracker.py           # Core three-state tracker
        ├── kalman.py            # Constant-velocity Kalman filter
        ├── identity.py          # Confidence + state transitions
        ├── face_body.py         # Face-to-body spatial mapping
        ├── centering.py         # Target centring output
        └── sim/                 # Scriptable simulator (no camera)
```
