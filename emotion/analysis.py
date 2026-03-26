import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from emotion.classifier import classify_emotion

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def collect_image_paths(dataset_path):
    data = []

    for emotion in EMOTIONS:
        emotion_folder = os.path.join(dataset_path, emotion)

        if not os.path.isdir(emotion_folder):
            continue

        for file_name in os.listdir(emotion_folder):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(emotion_folder, file_name)
                data.append((image_path, emotion))

    return data


def extract_main_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = img[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, (224, 224))
    return face_crop


def analyze_dataset(dataset_path, output_csv="results.csv"):
    rows = []
    image_data = collect_image_paths(dataset_path)

    for image_path, actual_emotion in image_data:
        img = cv2.imread(image_path)

        if img is None:
            rows.append({
                "image_path": image_path,
                "actual_emotion": actual_emotion,
                "predicted_emotion": None,
                "confidence": 0.0,
                "success": False,
                "error": "Image could not be loaded"
            })
            continue

        face_crop = extract_main_face(img)

        if face_crop is None:
            rows.append({
                "image_path": image_path,
                "actual_emotion": actual_emotion,
                "predicted_emotion": None,
                "confidence": 0.0,
                "success": False,
                "error": "No face detected"
            })
            continue

        result = classify_emotion(face_crop)

        rows.append({
            "image_path": image_path,
            "actual_emotion": actual_emotion,
            "predicted_emotion": result["dominant_emotion"],
            "confidence": result["confidence"],
            "success": result["success"],
            "error": result["error"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def compute_overall_accuracy(df):
    valid_df = df[(df["success"] == True) & (df["predicted_emotion"].notna())]

    if len(valid_df) == 0:
        return 0.0

    correct = (valid_df["actual_emotion"] == valid_df["predicted_emotion"]).sum()
    return correct / len(valid_df)


def compute_per_emotion_accuracy(df):
    results = {}
    valid_df = df[(df["success"] == True) & (df["predicted_emotion"].notna())]

    for emotion in EMOTIONS:
        emotion_df = valid_df[valid_df["actual_emotion"] == emotion]

        if len(emotion_df) == 0:
            results[emotion] = 0.0
            continue

        correct = (emotion_df["actual_emotion"] == emotion_df["predicted_emotion"]).sum()
        results[emotion] = correct / len(emotion_df)

    return results


def compute_detection_success_rate(df):
    if len(df) == 0:
        return 0.0

    successful = (df["success"] == True).sum()
    return successful / len(df)


def plot_confusion_matrix(df, save_path="confusion_matrix.png"):
    valid_df = df[(df["success"] == True) & (df["predicted_emotion"].notna())]

    if len(valid_df) == 0:
        print("No valid predictions to plot confusion matrix.")
        return

    y_true = valid_df["actual_emotion"]
    y_pred = valid_df["predicted_emotion"]

    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Emotion Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_per_emotion_accuracy(per_emotion_acc, save_path="per_emotion_accuracy.png"):
    emotions = list(per_emotion_acc.keys())
    accuracies = list(per_emotion_acc.values())

    plt.figure(figsize=(8, 5))
    plt.bar(emotions, accuracies)
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.ylabel("Accuracy")
    plt.title("Per-Emotion Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def print_summary(df):
    overall_acc = compute_overall_accuracy(df)
    per_emotion_acc = compute_per_emotion_accuracy(df)
    detection_rate = compute_detection_success_rate(df)

    print("\n===== BATCH ANALYSIS SUMMARY =====")
    print(f"Total images: {len(df)}")

    successful = (df["success"] == True).sum()
    print(f"Successful predictions: {successful}")
    print(f"Detection / processing success rate: {detection_rate:.2%}")
    print(f"Overall accuracy: {overall_acc:.2%}")

    print("\nPer-emotion accuracy:")
    for emotion, acc in per_emotion_acc.items():
        print(f"{emotion}: {acc:.2%}")

    return overall_acc, per_emotion_acc