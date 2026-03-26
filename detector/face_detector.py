import cv2 as cv


def load_face_cascade():
    """
    Load Haar Cascade model for face detection.
    """
    return cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(frame, face_cascade):
    """
    Detect faces on the frame using Partner A settings.
    Returns faces in format: (x, y, w, h)
    """
    gray_small = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(100, 100)
    )

    return faces