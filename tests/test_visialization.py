import os
import cv2
from emotion.classifier import classify_emotion
from emotion.visualization import draw_face_result


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "test_face.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image -> {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print("No face detected.")
        return

    # Берём самое большое лицо
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Slightly shrink detected face box for cleaner visualization
    shrink_x = int(w * 0.12)
    shrink_y = int(h * 0.10)

    x = x + shrink_x
    y = y + shrink_y
    w = w - 2 * shrink_x
    h = h - 2 * shrink_y

    # Safety clamp
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img.shape[1] - x))
    h = max(1, min(h, img.shape[0] - y))

    face_crop = img[y:y+h, x:x+w]
    if face_crop.size == 0:
        print("Face crop is empty.")
        return

    result = classify_emotion(face_crop)

    if result["success"]:
        print("Dominant emotion:", result["dominant_emotion"])
        print("Confidence:", result["confidence"])
        draw_face_result(img, x, y, w, h, result)
    else:
        print("Classification error:", result["error"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    max_display_width = 1000
    if img.shape[1] > max_display_width:
        scale = max_display_width / img.shape[1]
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_w, new_h))

    cv2.imshow("Visualization Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()