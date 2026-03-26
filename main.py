import cv2 as cv
import time

from detector.preprocess import preprocess
from detector.face_detector import load_face_cascade, detect_faces
from detector.webcam_utils import calculate_fps, save_screenshot

from emotion.classifier import classify_emotion
from emotion.visualization import draw_face_result


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
    frame_cnt = 0
    n = 5
    last_faces = []

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

            face = frame[y:y+h, x:x+w]

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
                else:
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


if __name__ == "__main__":
    main()