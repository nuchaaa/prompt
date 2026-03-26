from deepface import DeepFace
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def classify_emotion(face_img):
    result_template = {
        "success": False,
        "dominant_emotion": None,
        "confidence": 0.0,
        "scores": {emotion: 0.0 for emotion in EMOTIONS},
        "error": None
    }

    try:
        if face_img is None:
            result_template["error"] = "face_img is None"
            return result_template

        if not isinstance(face_img, np.ndarray):
            result_template["error"] = "face_img must be a numpy array"
            return result_template

        if face_img.size == 0:
            result_template["error"] = "face_img is empty"
            return result_template

        analysis = DeepFace.analyze(
            img_path=face_img,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(analysis, list):
            analysis = analysis[0]

        emotion_scores = analysis.get("emotion", {})
        dominant_emotion = analysis.get("dominant_emotion", None)

        scores = {}
        for emotion in EMOTIONS:
            scores[emotion] = float(emotion_scores.get(emotion, 0.0))

        confidence = 0.0
        if dominant_emotion in scores:
            confidence = scores[dominant_emotion]

        return {
            "success": True,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "scores": scores,
            "error": None
        }

    except Exception as e:
        result_template["error"] = str(e)
        return result_template