import cv2

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "neutral": (180, 180, 180)
}

EMOTIONS_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def get_emotion_color(emotion):
    return EMOTION_COLORS.get(emotion, (255, 255, 255))


def draw_emotion_box(image, x, y, w, h, result):
    emotion = result.get("dominant_emotion", "unknown")
    confidence = result.get("confidence", 0.0)
    color = get_emotion_color(emotion)

    # face box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    # bigger label
    label = f"{emotion}: {confidence:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_x = x
    label_y = max(40, y - 12)

    cv2.rectangle(
        image,
        (label_x, label_y - text_h - 12),
        (label_x + text_w + 12, label_y + baseline - 4),
        color,
        -1
    )

    cv2.putText(
        image,
        label,
        (label_x + 6, label_y - 6),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

    return image


def choose_chart_position(image, x, y, w, h, chart_width, chart_height, margin=15):
    img_h, img_w = image.shape[:2]

    # right
    right_x = x + w + margin
    right_y = y
    if right_x + chart_width <= img_w:
        chart_x = right_x
        chart_y = min(max(0, right_y), max(0, img_h - chart_height))
        return chart_x, chart_y

    # left
    left_x = x - chart_width - margin
    left_y = y
    if left_x >= 0:
        chart_x = left_x
        chart_y = min(max(0, left_y), max(0, img_h - chart_height))
        return chart_x, chart_y

    # below
    below_x = x
    below_y = y + h + margin

    chart_x = min(max(0, below_x), max(0, img_w - chart_width))
    chart_y = min(max(0, below_y), max(0, img_h - chart_height))
    return chart_x, chart_y


def draw_mini_bar_chart(
    image,
    x,
    y,
    scores,
    face_w=None,
    face_h=None,
    chart_width=180,
    bar_height=18,
    gap=10
):
    num_bars = len(EMOTIONS_ORDER)
    chart_height = num_bars * (bar_height + gap) + 20
    panel_width = chart_width + 110

    if face_w is not None and face_h is not None:
        chart_x, chart_y = choose_chart_position(
            image, x, y, face_w, face_h, panel_width, chart_height
        )
    else:
        chart_x, chart_y = x, y

    # background panel
    cv2.rectangle(
        image,
        (chart_x, chart_y),
        (chart_x + panel_width, chart_y + chart_height),
        (35, 35, 35),
        -1
    )

    cv2.rectangle(
        image,
        (chart_x, chart_y),
        (chart_x + panel_width, chart_y + chart_height),
        (255, 255, 255),
        2
    )

    for i, emotion in enumerate(EMOTIONS_ORDER):
        score = float(scores.get(emotion, 0.0))
        score = max(0.0, min(score, 100.0))
        color = get_emotion_color(emotion)

        row_y = chart_y + 12 + i * (bar_height + gap)

        # emotion text bigger
        cv2.putText(
            image,
            emotion,
            (chart_x + 8, row_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        filled_w = int((score / 100.0) * chart_width)
        filled_w = min(filled_w, chart_width)

        # filled bar
        cv2.rectangle(
            image,
            (chart_x + 95, row_y),
            (chart_x + 95 + filled_w, row_y + bar_height),
            color,
            -1
        )

        # bar outline
        cv2.rectangle(
            image,
            (chart_x + 95, row_y),
            (chart_x + 95 + chart_width, row_y + bar_height),
            (255, 255, 255),
            2
        )

    return image


def draw_face_result(image, x, y, w, h, result):
    draw_emotion_box(image, x, y, w, h, result)
    draw_mini_bar_chart(
        image=image,
        x=x,
        y=y,
        scores=result["scores"],
        face_w=w,
        face_h=h
    )
    return image


def draw_multiple_faces(image, face_results):
    for x, y, w, h, result in face_results:
        draw_face_result(image, x, y, w, h, result)
    return image