# Facial Emotion Recognition System

Real-time emotion detection from a webcam stream and batch accuracy analysis
on a labelled image dataset.  Built with OpenCV, DeepFace, and Matplotlib.

---

## Project structure

```
AIPROMPT/
├── main.py                   ← entry point (webcam & batch modes)
├── results.csv               ← batch output (auto-generated)
├── confusion_matrix.png      ← auto-generated after batch run
├── per_emotion_accuracy.png  ← auto-generated after batch run
│
├── detector/
│   ├── preprocess.py         ← grayscale → CLAHE → resize (Partner A)
│   ├── face_detector.py      ← Haar Cascade loader & detector (Partner A)
│   └── webcam_utils.py       ← FPS counter, screenshot saver (Partner A)
│
├── emotion/
│   ├── classifier.py         ← DeepFace wrapper, returns all 7 scores (Partner B)
│   ├── visualization.py      ← colour-coded boxes + mini bar chart (Partner B)
│   └── analysis.py           ← batch pipeline + confusion matrix (Partner B)
│
├── dataset/
│   ├── angry/   (*.jpg / *.jpeg)
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── tests/
    ├── test_classifier.py
    ├── test_visialization.py
    └── test_analysis.py
```

---

## Requirements

| Package | Version tested |
|---------|---------------|
| Python  | 3.10 – 3.13   |
| opencv-python | ≥ 4.8 |
| deepface | ≥ 0.0.93 |
| tensorflow *or* torch | ≥ 2.x |
| matplotlib | ≥ 3.7 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |

Install everything at once:

```bash
pip install opencv-python deepface matplotlib scikit-learn pandas
# DeepFace will pull in tensorflow automatically.
# If you prefer PyTorch, install torch separately.
```

> **macOS / Apple Silicon:** replace `opencv-python` with `opencv-python-headless`
> and add `pip install tf-keras` if you hit import errors.

---

## Quick start

### 1 · Clone / unzip the project

```bash
unzip AIPROMPT_3.zip
cd AIPROMPT
```

### 2 · Webcam mode (default)

```bash
python main.py --mode webcam
```

A window titled **"Emotion Detection — Webcam"** opens.

| Key | Action |
|-----|--------|
| `S` | Save screenshot (`Screenshot_<timestamp>.png`) |
| `Q` | Quit — a **matplotlib summary chart** appears showing emotion counts and distribution for the whole session |

The summary PNG is also saved automatically as `session_summary_<timestamp>.png`.

### 3 · Batch mode

Place labelled images in `dataset/<emotion>/` subfolders
(angry, disgust, fear, happy, sad, surprise, neutral) then run:

```bash
python main.py --mode batch --dataset dataset --output_csv results.csv
```

Outputs produced:

| File | Contents |
|------|----------|
| `results.csv` | Per-image predictions and confidence scores |
| `confusion_matrix.png` | 7 × 7 confusion matrix |
| `per_emotion_accuracy.png` | Bar chart of per-emotion accuracy |

Console will also print overall accuracy and per-emotion breakdown.

---

## Running the tests

```bash
# from the AIPROMPT/ directory
python -m tests.test_classifier
python -m tests.test_visialization
python -m tests.test_analysis
```

`test_classifier` and `test_visialization` require a webcam window
(they call `cv2.imshow`); run them on a machine with a display.

---

## How the pipeline works

```
Camera frame
    │
    ▼
face_detector.py   ← Haar Cascade (scaleFactor=1.2, minNeighbors=8)
    │
    ▼  (every 5th frame)
preprocess.py      ← BGR→Gray → CLAHE(clip=2.0, tile=8×8) → resize 48×48
    │
    ▼
classifier.py      ← DeepFace.analyze(actions=["emotion"])
                      returns dominant_emotion + all 7 confidence scores
    │
    ▼
visualization.py   ← colour-coded bounding box
                      label + confidence %
                      mini bar chart per face (overlaid on frame)
    │
    ▼  (after Q is pressed)
show_session_summary()  ← matplotlib bar + pie chart, saved as PNG
```

---

## Dataset layout (batch mode)

```
dataset/
├── happy/
│   ├── 000001.jpg
│   └── ...
├── sad/
│   └── ...
└── (angry / disgust / fear / surprise / neutral)
```

Each subfolder name **must** exactly match one of the 7 emotion labels.
Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.

---

## Notes & known limitations

- **Lighting:** CLAHE normalises contrast, but very dark or backlit scenes
  will still reduce detection accuracy.
- **Single frontal face:** The classifier is optimised for one frontal face;
  profile faces or heavy occlusion may not be detected.
- **DeepFace first run:** On the first run DeepFace downloads model weights
  (~600 MB).  Make sure you have an internet connection.
- **macOS display issue:** If `cv2.imshow` freezes, run the script from a
  terminal (not an IDE) and make sure the Python executable has camera permissions.