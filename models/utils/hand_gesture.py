import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def classify_gesture(landmarks):
    thumb_tip = landmarks[4].x
    index_tip = landmarks[8].x
    pinky_tip = landmarks[20].x

    # Basic rule-based gesture classification
    if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:
        return "✋ Open Hand"
    elif landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y:
        return "✊ Fist"
    elif landmarks[12].y < landmarks[10].y and landmarks[8].y > landmarks[6].y:
        return "✌️ Peace"
    else:
        return "🤷 Unknown"

def detect_gesture(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        gesture = "No hand detected"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            gesture = classify_gesture(hand_landmarks.landmark)
            annotated = image.copy()
            mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return gesture, annotated
        return gesture, image
