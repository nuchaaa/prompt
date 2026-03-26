import cv2
import os
from emotion.classifier import classify_emotion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "test_face.jpg")

img = cv2.imread(image_path)
print("Image path:", image_path)
print("Image loaded:", img is not None)

result = classify_emotion(img)
print(result)