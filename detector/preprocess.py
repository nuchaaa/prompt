import cv2 as cv


def preprocess(frame, t_size=(48, 48)):
    """
    Partner A: Data preparation for the model.
    Steps: Grayscale -> CLAHE -> Resize to 48x48.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(gray)
    resized = cv.resize(res, t_size, interpolation=cv.INTER_AREA)
    return resized