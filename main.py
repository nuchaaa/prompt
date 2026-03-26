import cv2 as cv
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector.preprocess import preprocess
from detector.face_detector import load_face_cascade, detect_faces
from detector.webcam_utils import calculate_fps, save_screenshot

from emotion.classifier import classify_emotion
from emotion.visualization import draw_face_result

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def show_session_summary(emotion_counts: dict, session_seconds: float):
    emotions = EMOTIONS
    counts = [emotion_counts.get(e, 0) for e in emotions]
    total = sum(counts)

    colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f", "#3498db", "#e67e22", "#95a5a6"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Webcam Session Summary  |  {total} detections  |  {session_seconds:.0f}s",
        fontsize=14, fontweight="bold"
    )

    bars = axes[0].bar(emotions, counts, color=colors, edgecolor="white")
    axes[0].set_title("Detection count per emotion")
    axes[0].set_xlabel("Emotion")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.3, str(cnt),
                         ha="center", va="bottom", fontsize=9)

    if total > 0:
        non_zero = [(e, c, col) for e, c, col in zip(emotions, counts, colors) if c > 0]
        axes[1].pie([x[1] for x in non_zero],
                    labels=[x[0] for x in non_zero],
                    colors=[x[2] for x in non_zero],
                    autopct="%1.1f%%", startangle=140)
    else:
        axes[1].text(0.5, 0.5, "No faces detected", ha="center", va="center")
        axes[1].axis("off")
    axes[1].set_title("Emotion distribution (%)")

    plt.tight_layout()
    save_path = f"session_summary_{int(time.time())}.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Session summary saved → {save_path}")


def main():
    face_cascade = load_face_cascade()

    if face_cascade.empty():
        print("Error: Could not find XML file")
        return

    capt = cv.VideoCapture(0)

    if not capt.isOpened():
        print("Error: Could not open camera")
        return

    prev_time = 0
    start_time = time.time()
    frame_cnt = 0
    n = 5
    last_faces = []
    emotion_counts = defaultdict(int)

    print("Controls: 'S' for Screenshot, 'Q' to Quit")

    while True:
        r, frame = capt.read()
        if not r:
            break

        frame = cv.flip(frame, 1)
        frame_cnt += 1

        if frame_cnt % n == 0:
            faces = detect_faces(frame, face_cascade)

            if len(faces) > 0:
                last_faces = faces

        for (x, y, w, h) in last_faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(
                frame,
                "User Face",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1
            )

            face = frame[y:y + h, x:x + w]

            if face.size > 0:
                processed_face = preprocess(face)

                preview = cv.cvtColor(processed_face, cv.COLOR_GRAY2BGR)
                preview = cv.resize(preview, (80, 80))
                frame[10:90, 10:90] = preview

                cv.putText(
                    frame,
                    "Input to AI",
                    (10, 105),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1
                )

                result = classify_emotion(face)

                if result["success"]:
                    draw_face_result(frame, x, y, w, h, result)
                    emotion_counts[result["dominant_emotion"]] += 1
                else:
                    print("Emotion Error:", result["error"])
                    cv.putText(
                        frame,
                        "Emotion Error",
                        (x, y + h + 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        fps, prev_time = calculate_fps(prev_time)

        cv.putText(
            frame,
            f"FPS: {fps}",
            (frame.shape[1] - 120, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

        cv.imshow("Emotion Detection System", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_screenshot(frame)

    capt.release()
    cv.destroyAllWindows()

    show_session_summary(dict(emotion_counts), time.time() - start_time)


if __name__ == "__main__":
    main()